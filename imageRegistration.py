import cv2 as cv
import numpy as np
from extraFunc import *

min_hessian = 800


def orb_algorithm(object_image, scene_image):
    orb = cv.ORB_create(min_hessian)
    (k1, d1) = orb.detectAndCompute(object_image, None)
    (k2, d2) = orb.detectAndCompute(scene_image, None)
    return k1, d1, k2, d2, object_image, scene_image


def surf_algorithm(object_image, scene_image):
    surf = cv.xfeatures2d.SURF_create(min_hessian)
    (k1, d1) = surf.detectAndCompute(object_image, None)
    (k2, d2) = surf.detectAndCompute(scene_image, None)
    return k1, d1, k2, d2, object_image, scene_image


def sift_algorithm(object_image, scene_image):
    sift = cv.xfeatures2d.SIFT_create(min_hessian)
    (k1, d1) = sift.detectAndCompute(object_image, None)
    (k2, d2) = sift.detectAndCompute(scene_image, None)
    return k1, d1, k2, d2, object_image, scene_image


def calculate(kps1, des1, kps2, des2, img_object, img_scene, turn):
    matcher = cv.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    obj = np.float32([kps1[m.queryIdx].pt for m in good_matches])
    scene = np.float32([kps2[m.trainIdx].pt for m in good_matches])

    H, mask = cv.findHomography(obj, scene, cv.RANSAC)
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

        if turn < 3 and area_match < 1 or area_match > w_s * h_s:
            decision = False
        else:
            matches_mask_t = [m for m in matches_mask]
            for i in range(len(matches_mask)):
                if matches_mask_t[i]:
                    point = tuple(scene[i])
                    if cv.pointPolygonTest(scene_corner, point, False) < 0:
                        matches_mask_t[i] = 0
            if area_match > 1 and matches_mask == matches_mask_t:
                decision = True
            else:
                decision = False

        img_scene = cv.polylines(img_scene, [np.int32(scene_corner)], True, 255, 3)
    else:
        decision = False

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       matchesMask=matches_mask,  # draw only inliers
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img3 = cv.drawMatches(img_object, kps1, img_scene, kps2, good_matches, None, **draw_params)
    img3 = resize_with_aspect_ratio(img3, 1800)

    return decision, img3


def image_registration(path_to_object_image, path_to_scene_image, turn=1):
    img_object = cv.imread(path_to_object_image, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(path_to_scene_image, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print("Error reading images")
        return None

    if turn == 1:
        kps1, des1, kps2, des2, obj, scene = sift_algorithm(img_object, img_scene)
    elif turn == 3:
        kps1, des1, kps2, des2, obj, scene = sift_algorithm(img_scene, img_object)
    elif turn == 2:
        kps1, des1, kps2, des2, obj, scene = surf_algorithm(img_object, img_scene)
    else:
        kps1, des1, kps2, des2, obj, scene = surf_algorithm(img_scene, img_object)

    decision, img = calculate(kps1, des1, kps2, des2, obj, scene, turn)
    if not decision and turn < 4:
        turn += 1
        return image_registration(path_to_object_image, path_to_scene_image, turn)

    return decision, img
