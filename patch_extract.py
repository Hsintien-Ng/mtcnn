# -*- coding: utf-8 -*-
import time
import caffe
import cv2
import numpy as np
import os
import sys
from alignment import crop_face
from PIL import Image
from face_detect import detect_face


def progress_bar(current, total):
    max_arrow = 50
    num_arrow = int(current * max_arrow / total)
    num_line = max_arrow - num_arrow
    bar = '[' + '>' * num_arrow + '-' * num_line + '] ' \
        + '%d/%d' % (current, total)
    if current < total - 1:
        bar += '\r'
    else:
        bar += '\n'
    sys.stdout.write(bar)
    sys.stdout.flush()


def extract_patch(img, bboxes, landmarks):
    """
    extract 5 patches from the original image according to its landmarks
    left eye, right eye, left cheek, right cheek, mouth
    :param img: the rotated image before cropping
    :param bboxes: [lux, luy, rbx, rby]
    :param landmarks: [lex, ley, rex, rey, lmx, lmy, rmx, rmy]
    :return: 
    """
    flat = np.ones((1))
    if type(img) == type(flat):
        img = Image.fromarray(img)
    if bboxes is not None and len(bboxes) > 0:
        # landmarks
        lm = landmarks
        lex, ley = lm[0], lm[1]
        rex, rey = lm[2], lm[3]
        lmx, lmy = lm[4], lm[5]
        rmx, rmy = lm[6], lm[7]
        lcx, lcy = (lex + lmx) / 2., (ley + lmy) / 2.
        rcx, rcy = (rex + rmx) / 2., (rey + rmy) / 2.
        cmx, cmy = (lmx + rmx) / 2., (lmy + rmy) / 2.

        # bboxes
        bb = bboxes[0]
        lux, luy, rbx, rby = bb
        h = rbx - lux
        w = rby - luy
        if h < 0 and w < 0:
            print('h and w must be positive')
        scale = [[0.2, 0.2], [0.2, 0.2], [0.15, 0.15],
                 [0.15, 0.15], [0.3, 0.2]]
        beta = 1.3
        le_left = lex - w * scale[0][0] / beta
        le_upper = ley - h * scale[0][1] / beta
        le_right = lex + w * scale[0][0] / beta
        le_bottom = ley + h * scale[0][1] / beta
        re_left = rex - w * scale[1][0] / beta
        re_upper = rey - h * scale[1][1] / beta
        re_right = rex + w * scale[1][0] / beta
        re_bottom = rey + h * scale[1][1] / beta
        lc_left = lcx - w * scale[2][0] / beta
        lc_upper = lcy - h * scale[2][1] / beta
        lc_right = lcx + w * scale[2][0] / beta
        lc_bottom = lcy + h * scale[2][1] / beta
        rc_left = rcx - w * scale[3][0] / beta
        rc_upper = rcy - h * scale[3][1] / beta
        rc_right = rcx + w * scale[3][0] / beta
        rc_bottom = rcy + h * scale[3][1] / beta
        cm_left = cmx - w * scale[4][0] / beta
        cm_upper = cmy - h * scale[4][1] / beta
        cm_right = cmx + w * scale[4][0] / beta
        cm_bottom = cmy + h * scale[4][1] / beta
        le_bbox = [le_left, le_upper, le_right, le_bottom]
        re_bbox = [re_left, re_upper, re_right, re_bottom]
        lc_bbox = [lc_left, lc_upper, lc_right, lc_bottom]
        rc_bbox = [rc_left, rc_upper, rc_right, rc_bottom]
        cm_bbox = [cm_left, cm_upper, cm_right, cm_bottom]
        le_img = img.crop(le_bbox)
        re_img = img.crop(re_bbox)
        lc_img = img.crop(lc_bbox)
        rc_img = img.crop(rc_bbox)
        cm_img = img.crop(cm_bbox)

        img_crop = [le_img, re_img, lc_img, rc_img, cm_img]
        bbox_crop = [le_bbox, re_bbox, lc_bbox, rc_bbox, cm_bbox]
        return (img_crop, bbox_crop)




if __name__ == '__main__':
    inputFolder = os.path.join('/', 'mnt', 'disk50', 'datasets', 'MORPH', 'Patches_split')
    imgori = os.path.join('/', 'mnt', 'disk50', 'datasets', 'MORPH', 'Images_ori')
    face = os.path.join(inputFolder, 'face')

    imgNames = os.listdir(imgori)
    faceNames = os.listdir(face)

    total = len(imgNames)
    print(len(imgNames), len(faceNames))
    leftlist = []
    for i, imgName in enumerate(imgNames):
        if imgName[:-4] + '.png' not in faceNames:
            leftlist.append(imgName)
        progress_bar(i, total)
    print(leftlist)
    if len(leftlist) > 0:

        inputFolder = os.path.join('/', 'mnt', 'disk50', 'datasets', 'MORPH', 'Images_ori')
        outputFolder = os.path.join('/', 'mnt', 'disk50', 'datasets', 'MORPH', 'Patches_split')
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        split = ['left_eye', 'right_eye', 'left_cheek', 'right_cheek', 'mouth']
        if not os.path.exists(os.path.join(outputFolder, 'face')):
            os.makedirs(os.path.join(outputFolder, 'face'))
        for i in range(len(split)):
            if not os.path.exists(os.path.join(outputFolder, split[i])):
                os.makedirs(os.path.join(outputFolder, split[i]))

        # caffe setting
        caffe_model_path = "./model"
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        caffe.set_mode_cpu()
        PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
        RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
        ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)

        # read image
        # img_names = os.listdir(inputFolder)
        img_names = leftlist
        total = len(img_names)
        imgNone = []
        for k, img_name in enumerate(img_names):
            img = cv2.imread(os.path.join(inputFolder, img_name))
            if img is not None:
                img_matlab = img.copy()
                tmp = img_matlab[:, :, 2].copy()
                img_matlab[:, :, 2] = img_matlab[:, :, 0]
                img_matlab[:, :, 0] = tmp

                bboxes, landmarks = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)

                if len(bboxes) > 0 and len(landmarks) > 0:
                    face, img_rot, bboxes, landmarks = crop_face(img, bboxes, landmarks)
                    img_crop, bbox_crop = extract_patch(img_rot, bboxes, landmarks)
                    for i in range(len(img_crop)):
                        img_crop[i] = np.array(img_crop[i])

                    for i in range(len(split)):
                        cv2.imwrite(os.path.join(outputFolder, split[i], img_name[:-4] + '.png'), img_crop[i])
                    cv2.imwrite(os.path.join(outputFolder, 'face', img_name[:-4] + '.png'), np.array(face))
                    # cv2.imwrite(os.path.join(outputFolder, 'img_rot', img_name[:-4] + '.png'), np.array(img_rot))
                else:
                    print(img_name + ' has no face.')
            else:
                print(img_name)
            progress_bar(k, total)


