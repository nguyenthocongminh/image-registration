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

            if turn in retry and area_match < 1 or area_match > w_s * h_s:
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
        img3 = np.hstack((img_object, img_scene))
        img3 = resize_with_aspect_ratio(img3, 1800)

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


def image_registration(path_to_object_image, path_to_scene_image):
    img_object = cv.imread(path_to_object_image, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(path_to_scene_image, cv.IMREAD_GRAYSCALE)

    h_o, w_o = img_object.shape
    h_s, w_s = img_scene.shape

    if w_o*h_o > w_s*h_s:
        img_object = resize_with_aspect_ratio(img_object, w_s)
    if w_s*h_s > w_o*h_o:
        img_scene = resize_with_aspect_ratio(img_scene, w_o)

    h_o, w_o = img_object.shape
    h_s, w_s = img_scene.shape

    mirror_object = img_object.copy()
    mirror_scene = img_scene.copy()
    mirror_object = resize_with_aspect_ratio(mirror_object, 800)
    mirror_scene = resize_with_aspect_ratio(mirror_scene, 800)

    if w_o > 1920:
        img_object = resize_with_aspect_ratio(img_object, 1920)
    if w_s > 1920:
        img_scene = resize_with_aspect_ratio(img_scene, 1920)

    img_object_array = [
        img_object,
        mirror_object
    ]
    img_scene_array = [
        img_scene,
        mirror_scene
    ]

    predict = False
    img = []

    for i in [0, 1]:
        _obj = img_object_array[i]
        _sce = img_scene_array[i]

        if img_object is None or img_scene is None:
            print("Error reading images")
            return None

        turn = 1
        while not predict and turn <= 8:
            try:
                if turn == 1:
                    kps1, des1, kps2, des2, obj, scene, matcher = sift_algorithm(_obj, _sce)
                elif turn == 2:
                    kps1, des1, kps2, des2, obj, scene, matcher = sift_algorithm(_sce, _obj)
                elif turn == 3:
                    kps1, des1, kps2, des2, obj, scene, matcher = surf_algorithm(_obj, _sce)
                elif turn == 4:
                    kps1, des1, kps2, des2, obj, scene, matcher = surf_algorithm(_sce, _obj)
                elif turn == 5:
                    kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(_obj, _sce)
                elif turn == 6:
                    kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(_sce, _obj)
                elif turn == 7:
                    kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(_obj, _sce, 3)
                else:
                    kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(_sce, _obj, 3)

                predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher, turn)
            except:
                predict = False
            turn += 1
            gc.collect()
        tmp_img = img.copy()
        if predict:
            break
        img = tmp_img
        gc.collect()

    return predict, img
