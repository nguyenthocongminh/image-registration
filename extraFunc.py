import cv2 as cv


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