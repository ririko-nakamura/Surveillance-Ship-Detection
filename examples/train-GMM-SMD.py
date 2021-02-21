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
# Evenly sample n_frame*sample_rate frames in a video
sample_rate = 0.01
# Sample n image patches for each frame
n_samples = 20
sample_size = 50

train_videos = [
    "MVI_1609_VIS.avi",
    "MVI_1610_VIS.avi",
    "MVI_1478_VIS.avi",
    "MVI_1479_VIS.avi",
    "MVI_1481_VIS.avi",
    "MVI_1482_VIS.avi",
    "MVI_1584_VIS.avi",
    "MVI_1613_VIS.avi",
    "MVI_1614_VIS.avi",
    "MVI_1615_VIS.avi",
    "MVI_1617_VIS.avi",
    "MVI_1619_VIS.avi",
    "MVI_1620_VIS.avi",
    "MVI_1583_VIS.avi",
    "MVI_1622_VIS.avi",
    "MVI_1623_VIS.avi",
    "MVI_1587_VIS.avi",
    "MVI_1624_VIS.avi",
    "MVI_1625_VIS.avi",
    "MVI_1592_VIS.avi",
    "MVI_1644_VIS.avi",
    "MVI_1645_VIS.avi",
    "MVI_1646_VIS.avi"
]

model = GMM_BGS(n_components=15)

def generateBackgroundMask(shape, annotations):
    mask = np.full(shape, 0, dtype=np.uint8)
    for i in range(n_samples):
        x = np.random.randint(sample_size//2, shape[1] - sample_size//2)
        y = np.random.randint(sample_size//2, shape[0] - sample_size//2)
        pt1 = (x - sample_size//2, y - sample_size//2)
        pt2 = (x + sample_size//2, y + sample_size//2)
        mask = cv.rectangle(mask, pt1, pt2, color=255, thickness=cv.FILLED)
    for anno in annotations:
        anno = [int(value) for value in anno]
        pt1 = (anno[0], anno[1])
        pt2 = (anno[0] + anno[2], anno[1] + anno[3])
        mask = cv.rectangle(mask, pt1, pt2, color=0, thickness=cv.FILLED)
    return mask

def getHorizon(annotation):
    anno = np.hstack(annotation)[0]
    return Horizon(Horizon.POINT_K, ((anno[0], anno[1]), - anno[2] / anno[3]))

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
        if index % int(1/sample_rate) == 0:
            img = cv.medianBlur(img, 3)
            img_list.append(img)
            bg_mask_list.append(generateBackgroundMask(img_list[-1].shape[0:2], object_gt[index][6]))
            horizon_list.append(getHorizon(horizon_gt[index]))

            # Visualize mask
            mask = np.array([bg_mask_list[-1] for i in range(3)]).transpose(1, 2, 0)
            debug = (img * 0.7 + mask * 0.3).astype(np.uint8)
            cv.imshow("debug", cv.resize(debug, (640, 360)))
            cv.waitKey(1)


dataset = zip(img_list, bg_mask_list, horizon_list)

model.fit(dataset)

model.save_model('GMM.model')



