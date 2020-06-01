import imutils
import numpy as np
import gc

from extraFunc import *

min_hessian = 800
max_width = 1920


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


def calculate(kps1, des1, kps2, des2, img_object, img_scene, matcher):
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

            if area_match < 1:
                predict = False
            else:
                matches_mask_t = [m for m in matches_mask]
                for i in range(len(matches_mask)):
                    if matches_mask_t[i]:
                        point = tuple(scene[i])
                        if cv.pointPolygonTest(scene_corner, point, False) < 0:
                            matches_mask_t[i] = 0
                if sum(matches_mask_t) / sum(matches_mask) > 0.91:
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
    height, width, dimension = img.shape
    mask = cv.threshold(img.copy(), 210, 255, cv.THRESH_BINARY)[1][:, :, 0]
    mask = cv.dilate(mask, np.ones((5, 5)))
    for i in range(0, height):
        for j in range(0, int((width * 3.5) / 6)):
            mask[i][j] = 0
    for i in range(0, int((height * 5) / 6)):
        for j in range(0, width):
            mask[i][j] = 0
    dst = cv.inpaint(img, mask, inpaintRadius=7, flags=cv.INPAINT_TELEA)
    return dst


def validateSceneObject(img_object, img_scene):
    object_gray = cv.cvtColor(img_object, cv.COLOR_BGR2GRAY)
    scene_gray = cv.cvtColor(img_scene, cv.COLOR_BGR2GRAY)
    h_o, w_o = object_gray.shape
    h_s, w_s = scene_gray.shape
    count_black_object = h_o * w_o - cv.countNonZero(object_gray)
    count_black_scene = h_s * w_s - cv.countNonZero(scene_gray)
    return count_black_scene >= count_black_object


def image_registration(path_to_object_image, path_to_scene_image, algorithm=1):
    img_object = cv.imread(path_to_object_image, cv.IMREAD_COLOR)
    img_scene = cv.imread(path_to_scene_image, cv.IMREAD_COLOR)

    if not validateSceneObject(img_object, img_scene):
        temp = img_scene
        img_scene = img_object
        img_object = temp
        del temp
        gc.collect()

    h_o, w_o, d_o = img_object.shape
    h_s, w_s, d_s = img_scene.shape

    if w_o > max_width:
        img_object = resize_with_aspect_ratio(img_object, max_width)
    if w_s > max_width:
        img_scene = resize_with_aspect_ratio(img_scene, max_width)
    h_o, w_o, d_o = img_object.shape
    h_s, w_s, d_s = img_scene.shape

    img_object = cv.cvtColor(remove_time(img_object), cv.COLOR_BGR2GRAY)
    img_scene = cv.cvtColor(remove_time(img_scene), cv.COLOR_BGR2GRAY)

    predict = False
    img = None

    try:
        if algorithm == 1 or algorithm == "sift":
            kps1, des1, kps2, des2, obj, scene, matcher = sift_algorithm(img_object, img_scene)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher)
            gc.collect()

        if algorithm == 2 or algorithm == "surf":
            kps1, des1, kps2, des2, obj, scene, matcher = surf_algorithm(img_object, img_scene)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher)
            gc.collect()

        if algorithm == 3 or algorithm == "brisk":
            kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(img_object, img_scene)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher)
            gc.collect()

        if algorithm == 4 or algorithm == "brisk_improve":
            kps1, des1, kps2, des2, obj, scene, matcher = brisk_algorithm(img_object, img_scene, 3)
            predict, img = calculate(kps1, des1, kps2, des2, obj, scene, matcher)
            gc.collect()
    except (ValueError, Exception):
        predict = False

    if img is None:
        if h_s < h_o:
            img_scene = np.vstack((img_scene, np.zeros((h_o - h_s, w_s))))
        if h_o < h_s:
            img_object = np.vstack((img_object, np.zeros((h_s - h_o, w_o))))
        img = np.hstack((img_object, img_scene))
        img = resize_with_aspect_ratio(img, 1800)

    gc.collect()
    return predict, img
