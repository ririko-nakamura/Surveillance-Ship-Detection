# TODO: argparse
# import argparse
import os, sys

import cv2 as cv
import numpy as np
from scipy.io import loadmat

from pipeline.background_subtraction.GMM import GMM_BGS

dataset = sys.argv[1]

train_videos = [
    "MVI_1448_VIS_Haze.avi",
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

def generateBackgroundMask():
    mask = np.fill(img.shape, 255, dtype=np.uint8)
    for bbox in annotations:
        pt1 = []
        pt2 = []
        mask = cv.rectangle(mask, pt1, pt2, color=0, thickness=cv.FILLED)

def getHorizon():
    pass

img_list = []
bg_mask_list = []
horizon_list = []

for filename in train_videos:

    video = cv.VideoCapture(os.path.join(dataset, "VIS_Onshore/Videos", filename))
    horizon_gt = loadmat(os.path.join(dataset, "VIS_Onshore/HorizonGT", filename[:-4]+"_HorizonGT.mat"))["structXML"][0]
    object_gt = loadmat(os.path.join(dataset, "VIS_Onshore/ObjectGT", filename[:-4]+"_ObjectGT.mat"))

    for index in range(min(horizon_gt.shape, object_gt.shape, video.frames)):
        ret, img = video.read()
        if not ret:
            break
        if index % 20 == 0:
            img_list.append(img)
            bg_mask_list.append(generateBackgroundMask())
            horizon_list.append(getHorizon())

