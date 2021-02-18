import os, sys

import cv2 as cv
import numpy as np
import pickle as pkl

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../")
# Configure the pipeline here
from pipeline.horizon_detection.MSCM_LiFe import HorizonDetector
from pybgs import TwoPoints as BackgroundSubtractor
from pipeline.foreground_segamentation.general import ForegroundSegamentor

#dataset = "D:/Datasets/SMD/VIS_Onshore/test"
dataset = "data/VIS_Onshore/test/%d.jpg"

# horizonDetector = HorizonDetector(10)

i = 0

video = cv.VideoCapture(dataset)

#for f in os.listdir(dataset):
if True:

    subtractor = BackgroundSubtractor()
    segamentor = ForegroundSegamentor()

    #video = cv.VideoCapture()
    #video.open(os.path.join(dataset, f))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    #writer = cv.VideoWriter(os.path.join(dataset, f[:-4]+".mp4"), fourcc, 30, (1920, 1080), True)
    writer = cv.VideoWriter("result.mp4", fourcc, 30, (1920, 1080), True)
    
    cur_video_detections = []

    flag, img = video.read()
    while flag:

        if i % 200 == 0:
            subtractor = BackgroundSubtractor()
            segamentor = ForegroundSegamentor()

        #img = cv.resize(img, (1600, 900))
        mask = subtractor.apply(img)
        detections = segamentor.apply(mask)
        cur_video_detections.append(detections)
        for det in detections:
            img = cv.rectangle(img, (det.min_x, det.min_y), (det.max_x, det.max_y), (0, 255, 0))
        writer.write(img)
        cv.imshow("detections", img)
        cv.waitKey(10)
        flag, img = video.read()

        i = i + 1

    with open("result.pkl", 'wb') as pkldump:
    #with open(os.path.join(dataset, f[:-4]+".pkl"), 'wb') as pkldump:
        pkl.dump(cur_video_detections, pkldump)
    with open("result.pkl", 'rb') as pkldump:
    #with open(os.path.join(dataset, f[:-4]+".pkl"), 'rb') as pkldump:
        cur_video_detections = pkl.load(pkldump)
        print(cur_video_detections)