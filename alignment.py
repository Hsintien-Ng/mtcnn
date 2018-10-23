# -*- coding: utf-8 -*-
import math
import numpy as np
from PIL import Image


def crop_face(image, bboxes, points):
    """
    crop face from the original image and 
    :param image: an original image.
           bboxes: the detected bounding boxes of original coordinate.
           points: the detected landmark of original coordinate.
    :return: 
    """
    flat = np.ones((1))
    if type(image) == type(flat):
        image = Image.fromarray(image)
    if bboxes is not None and len(bboxes) > 0:
        bb = bboxes[0]
        lm = points[0]
        lex, ley = lm[0], lm[5]
        rex, rey = lm[1], lm[6]
        nex, ney = lm[2], lm[7]
        lmx, lmy = lm[3], lm[8]
        rmx, rmy = lm[4], lm[9]

        # # rotate the cropped image.
        # angle = calculate_angle(lex, ley, rex, rey)
        # image, lex, ley, rex, rey, lmx, lmy, rmx, rmy \
        #     = image_rotate(image, angle, lex, ley, rex, rey, lmx, lmy, rmx, rmy)

        # base parameters
        eye_width = rex - lex
        ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0  # the center of two eyes
        mouth_width = rmx - lmx
        mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0  # the center of the mouth
        em_height = mcy - ecy  # the height between eye center and mouth center
        fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0  # the center of the face

        # pure face
        if eye_width > em_height:
            alpha = eye_width
        else:
            alpha = em_height
        # print("eye_wid", eye_width)
        # print("em_height", em_height)
        # print("alpha:", alpha)
        g_beta = 2.0
        g_left = fcx - alpha / 2.0 * g_beta
        g_upper = fcy - alpha / 2.0 * g_beta
        g_right = fcx + alpha / 2.0 * g_beta
        g_lower = fcy + alpha / 2.0 * g_beta
        g_face = image.crop([g_left, g_upper, g_right, g_lower])

        landmark = [lex, ley, rex, rey, lmx, lmy, rmx, rmy]

        return g_face, image, [[g_left, g_upper, g_right, g_lower]], landmark


def calculate_angle(elx, ely, erx, ery):
    """
    calculate the rotate angle of image according to eyes.
    :param elx: coordinate x of left eye
    :param ely: coordinate y of left eye
    :param erx: coordinate x of right eye
    :param ery: coordinate y of right eye
    :return: the rotate angle of the face
    """
    dx = erx - elx
    dy = ery - ely
    angle = math.atan(dy / dx) * 180 / math.pi
    return angle


def image_rotate(img, angle, elx, ely, erx, ery, mlx, mly, mrx, mry, expand=1):
    """
    rotate the image.
    :param img: the original image
    :param angle: rotate angle
    :param elx: coordinate x of left eye
    :param ely: coordinate y of left eye
    :param erx: coordinate x of right eye
    :param ery: coordinate y of right eye
    :param mlx: coordinate x of left mouth
    :param mly: coordinate y of left mouth
    :param mrx: coordinate x of right mouth
    :param mry: coordinate y of right mouth
    :param expand: a flat of expansion
    :return: the rotated image and rotated coordinates of keypoints
    """
    width, height = img.size
    img = img.rotate(angle, expand=expand)

    # expand = 0
    # expand = 1
    elx, ely = pos_transform_resize(angle, elx, ely, width, height)
    erx, ery = pos_transform_resize(angle, erx, ery, width, height)
    mlx, mly = pos_transform_resize(angle, mlx, mly, width, height)
    mrx, mry = pos_transform_resize(angle, mrx, mry, width, height)
    return img, ely, elx, ery, erx, mly, mlx, mry, mrx


def pos_transform_resize(angle, x, y, w, h):
    """
    return new coordinates after rotating the image with expandison.
    :param angle: the rotate angle
    :param x: coordinate x of any point in the original image
    :param y: coordinate y of any point in the original image
    :param w: width of the original image
    :param h: height of the original image
    :return: transformed coordinate (y, x)
    """
    angle = angle * math.pi / 180
    matrix = [
        -math.sin(angle), math.cos(angle), 0.0,
        math.cos(angle), math.sin(angle), 0.0
    ]
    def transform(x, y, matrix=matrix):
        a, b, c, d, e, f = matrix
        return a * x + b * y + c, d * x + e * y + f

    # calculate output size
    xx = []
    yy = []
    for x_, y_ in ((0, 0), (w, 0), (w, h), (0, h)):
        x_, y_ = transform(x_, y_)
        xx.append(x_)
        yy.append(y_)
    ww = int(math.ceil(max(xx)) - math.floor(min(xx)))
    hh = int(math.ceil(max(yy)) - math.floor(min(yy)))

    # adjust center
    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = ww / 2.0 - cx
    matrix[5] = hh / 2.0 - cy

    tx, ty = transform(x, y)
    return tx, ty
