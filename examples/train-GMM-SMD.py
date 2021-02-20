# TODO: argparse
# import argparse
import os, sys

import cv2 as cv
import numpy as np
from scipy.io import loadmat

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../")
from pipeline.background_subtraction.GMM import GMM_BGS
from pipeline.general import Horizon

dataset = sys.argv[1]

train_videos = [
    "MVI_1474_VIS.avi",
    "MVI_1484_VIS.avi",
    "MVI_1486_VIS.avi",
    "MVI_1582_VIS.avi",
    "MVI_1612_VIS.avi",
    "MVI_1626_VIS.avi",
    "MVI_1627_VIS.avi",
    "MVI_1640_VIS.avi"
]

model = GMM_BGS(n_components=15)

def generateBackgroundMask(annotations):
    mask = np.full(img.shape, 255, dtype=np.uint8)
    for anno in annotations:
        anno = [int(value) for value in anno]
        pt1 = (anno[0], anno[1])
        pt2 = (anno[0] + anno[2], anno[1] + anno[3])
        mask = cv.rectangle(mask, pt1, pt2, color=0, thickness=cv.FILLED)
    return mask

def getHorizon(annotation):
    anno = np.hstack(annotation)[0]
    return Horizon(Horizon.POINT_K, ((anno[0], anno[1]), - anno[3] / anno[2]))

img_list = []
bg_mask_list = []
horizon_list = []

for filename in train_videos:

    video = cv.VideoCapture(os.path.join(dataset, "VIS_Onshore/Videos", filename))
    horizon_gt = loadmat(os.path.join(dataset, "VIS_Onshore/HorizonGT", filename[:-4]+"_HorizonGT.mat"))["structXML"][0]
    object_gt = loadmat(os.path.join(dataset, "VIS_Onshore/ObjectGT", filename[:-4]+"_ObjectGT.mat"))["structXML"][0]

    for index in range(min(int(video.get(cv.CAP_PROP_FRAME_COUNT)), horizon_gt.shape[0], object_gt.shape[0])):
        ret, img = video.read()
        if not ret:
            break
        if index % 20 == 0:
            img_list.append(img)
            bg_mask_list.append(generateBackgroundMask(object_gt[index][6]))
            horizon_list.append(getHorizon(horizon_gt[index]))

            debug = img * 0.7 + bg_mask_list[-1] * 0.3
            horizon_list[-1].render(debug)
            cv.imshow("debug", debug)
            cv.waitKey(10)


dataset = (img_list, bg_mask_list, horizon_list)

model.fit(dataset)

