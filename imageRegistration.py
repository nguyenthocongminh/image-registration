import cv2 as cv
import numpy as np


def calc_area(x_cords, y_cords):
    area = 0
    for x in range(2):
        v1, v2, v3 = 0, x + 1, x + 2
        tr_area = abs(0.5 * (x_cords[v1] * (y_cords[v2] - y_cords[v3]) +
                             x_cords[v2] * (y_cords[v3] - y_cords[v1]) +
                             x_cords[v3] * (y_cords[v1] - y_cords[v2])))
        area += tr_area
    return area


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


cvSIFT = 1
cvSURF = 2


def image_registration(path_to_object_image, path_to_scene_image, retry=True, filter_type=cvSIFT):
    img_object = cv.imread(path_to_object_image, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(path_to_scene_image, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print("Error reading images")
        return None

    min_hessian = 800
    if filter_type == cvSIFT:
        sift = cv.xfeatures2d.SIFT_create(min_hessian)
        (kps1, des1) = sift.detectAndCompute(img_object, None)
        (kps2, des2) = sift.detectAndCompute(img_scene, None)
    else:
        surf = cv.xfeatures2d.SURF_create(min_hessian)
        (kps1, des1) = surf.detectAndCompute(img_object, None)
        (kps2, des2) = surf.detectAndCompute(img_scene, None)

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

        if retry and (area_match < 1 or area_match > w_s * h_s):
            return image_registration(path_to_scene_image, path_to_object_image, retry=False)

        matches_mask_t = [m for m in matches_mask]
        for i in range(len(matches_mask)):
            if matches_mask_t[i]:
                point = tuple(scene[i])
                if cv.pointPolygonTest(scene_corner, point, False) < 0:
                    matches_mask_t[i] = 0
        if area_match > 1 and matches_mask == matches_mask_t:
            decision = True
        else:
            if filter_type == cvSIFT:
                return image_registration(path_to_object_image, path_to_scene_image, retry=True, filter_type=cvSURF)
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
