import imutils
import numpy as np
import gc

from extraFunc import *

min_hessian = 800
retry = [1, 3, 5, 7]


def surf_algorithm(object_image, scene_image):
    surf = cv.xfeatures2d.SURF_create(min_hessian)
    (k1, d1) = surf.detectAndCompute(object_image, None)
    (k2, d2) = surf.detectAndCompute(scene_image, None)
    m = cv.FlannBasedMatcher()
    return k1, d1, k2, d2, object_image, scene_image, m


def sift_algorithm(object_image, scene_image):
    sift = cv.xfeatures2d.SIFT_create(min_hessian)
    (k1, d1) = sift.detectAndCompute(object_image, None)
    (k2, d2) = sift.detectAndCompute(scene_image, None)
    m = cv.FlannBasedMatcher()
    return k1, d1, k2, d2, object_image, scene_image, m


def brisk_algorithm(object_image, scene_image, scale=1):
    sift = cv.BRISK_create(5, 3, scale)
    (k1, d1) = sift.detectAndCompute(object_image, None)
    (k2, d2) = sift.detectAndCompute(scene_image, None)
    m = cv.BFMatcher(cv.NORM_HAMMING)
    return k1, d1, k2, d2, object_image, scene_image, m


def calculate(kps1, des1, kps2, des2, img_object, img_scene, matcher, turn):
    matches = matcher.knnMatch(des1, des2, k=2)
    img2 = img_scene.copy()

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    obj = []
    scene = []
    for i in range(len(good_matches)):
        obj.append(kps1[good_matches[i].queryIdx].pt)
        scene.append(kps2[good_matches[i].trainIdx].pt)
    obj = np.asarray(obj)
    scene = np.asarray(scene)

    if len(obj) > 0 and len(scene) > 0:
        H, mask = cv.findHomography(obj, scene, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        if H is not None:
            h_o, w_o = img_object.shape
            obj_corner = np.float32([[0, 0], [0, h_o - 1], [w_o - 1, h_o - 1], [w_o - 1, 0]]).reshape(-1, 1, 2)
            scene_corner = cv.perspectiveTransform(obj_corner, H)

            x_cords = []
            y_cords = []
            h_s, w_s = img_scene.shape
            for i in scene_corner:
                x_cords.append(i[0][0])
                y_cords.append(i[0][1])
                if i[0][0] < 0:
                    i[0][0] = 0
                if i[0][1] < 0:
                    i[0][1] = 0
                if i[0][0] > w_s:
                    i[0][0] = w_s - 3
                if i[0][1] > h_s:
                    i[0][1] = h_s - 3

            area_match = calc_area(x_cords, y_cords)

            if turn == 1 and area_match < 1 or area_match > w_s * h_s:
                predict = False
            else:
                matches_mask_t = [m for m in matches_mask]
                for i in range(len(matches_mask)):
                    if matches_mask_t[i]:
                        point = tuple(scene[i])
                        if cv.pointPolygonTest(scene_corner, point, False) < 0:
                            matches_mask_t[i] = 0
                if area_match > 100 and sum(matches_mask_t)/sum(matches_mask) > 0.91:
                    predict = True
                else:
                    predict = False

            cv.polylines(img2, [np.int32(scene_corner)], True, 255, 3)
        else:
            predict = False

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           matchesMask=matches_mask,  # draw only inliers
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img3 = cv.drawMatches(img_object, kps1, img2, kps2, good_matches, None, **draw_params)
        img3 = resize_with_aspect_ratio(img3, 1800)
    else:
        predict = False
        img3 = None

    return predict, img3


def remove_time(img):
    # img = cv.bilateralFilter(img, 11, 17, 17)
    edged = cv.Canny(img, 30, 200)
    nts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(nts)

    h_img, w_img = img.shape

    screenCnt = []
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        if w_img/160 <= w <= w_img/25 and 0.35 <= w/h <= 1:
            screenCnt.append((x, y, w, h))

    valid_y = []
    for i in screenCnt:
        if i[1] > 0.8*h_img:
            valid_y.append(i)

    if len(valid_y) > 0:
        predict = True
        p_y = [i[1] for i in valid_y]
        min_y = max(set(p_y), key=p_y.count)

        valid = []
        for i in valid_y:
            if i[1] == min_y:
                valid.append(i)

        if len(valid) > 1:
            max_w = valid[0][2]
            max_h = valid[0][3]
            min_x = valid[0][0]
            max_x = valid[0][0]
            for i in valid:
                if min_x > i[0]:
                    min_x = i[0]
                if max_x < i[0]:
                    max_x = i[0]
                if max_w < i[2]:
                    max_w = i[2]
                if max_h < i[3]:
                    max_h = i[3]
            max_y = min_y + max_h
            max_x = max_x + max_w

            img = cv.rectangle(img, (min_x, min_y), (max_x, max_y), 0, -1)
    else:
        predict = False
    return predict, img


def image_registration(path_to_object_image, path_to_scene_image, algorithm=1):
    img_object = cv.imread(path_to_object_image, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(path_to_scene_image, cv.IMREAD_GRAYSCALE)

    h_o, w_o = img_object.shape
    h_s, w_s = img_scene.shape

    if algorithm == 3 or algorithm == 4:
        if w_o > 1920:
            img_object = resize_with_aspect_ratio(img_object, 1920)
        if w_s > 1920:
            img_scene = resize_with_aspect_ratio(img_scene, 1920)
        h_o, w_o = img_object.shape
        h_s, w_s = img_scene.shape

    turn = 1
    predict = False
    img = None

    try:
        if algorithm == 1 or algorithm == "sift":
            kps1, des1, kps2, des2, obj, scene, matcher = sift_algorithm(img_object, img_scene)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
            gc.collect()
            if not predict:
                kps1, des1, kps2, des2, obj, scene, matcher = sift_algorithm(img_scene, img_object)
                predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
                gc.collect()

        if algorithm == 2 or algorithm == "surf":
            kps1, des1, kps2, des2, obj, scene, matcher = surf_algorithm(img_object, img_scene)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
            turn += 1
            gc.collect()
            if not predict:
                kps1, des1, kps2, des2, obj, scene, matcher = surf_algorithm(img_scene, img_object)
                predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
                gc.collect()

        if algorithm == 3 or algorithm == "brisk":
            kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(img_object, img_scene)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
            turn += 1
            gc.collect()
            if not predict:
                kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(img_scene, img_object)
                predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
                gc.collect()

        if algorithm == 4 or algorithm == "brisk_improve":
            kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(img_object, img_scene, 3)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
            turn += 1
            gc.collect()
            if not predict:
                kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(img_scene, img_object, 3)
                predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
                gc.collect()
    except (ValueError, Exception):
        predict = False

    if img is None:
        if h_s < h_o:
            img_scene = np.vstack((img_scene, np.zeros((h_o-h_s, w_s))))
        if h_o < h_s:
            img_object = np.vstack((img_object, np.zeros((h_s-h_o, w_o))))
        img = np.hstack((img_object, img_scene))
        img = resize_with_aspect_ratio(img, 1800)

    gc.collect()
    return predict, img
