import os, sys

import cv2
import numpy as np
import pybgs

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../")
from pipeline.horizon_detection.MSCM_LiFe import HorizonDetector

dataset = "D:/Datasets/SMD/VIS_Onshore/Videos"

class detection:

    def __init__(self):
        self.max_x = -1
        self.min_x = 65535
        self.max_y = -1
        self.min_y = 65535

    def area(self):
        if self.max_x < self.min_x or self.max_y < self.min_y:
            return 0
        else:
            return (self.max_x - self.min_x) * (self.max_y - self.min_y)

# horizonDetector = HorizonDetector(10)

# test median blur function and multiscale filter images build
if False:
    sample = np.arange(20*20).reshape((20, 20))
    sample = horizonDetector.filterMultiscaleImages(sample)
    for i in range(len(sample)):
        print(sample[i])

if False:
    for i in range(1000, 2001, 100):
        image = cv2.imread("./data/VIS_Onshore/test/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        horizon = horizonDetector.detect(image)
        horizon.render(image)
        cv2.imwrite("./results/test_{}_horizon_detection.jpg".format(i), image)

for f in os.listdir(dataset):

    subtractor = pybgs.TwoPoints()

    video = cv2.VideoCapture()
    video.open(os.path.join(dataset, f))

    flag, img = video.read()
    while flag:
        img = cv2.resize(img, (960, 540))
        mask = subtractor.apply(img)
        cv2.imshow("video", img)
        cv2.imshow("foreground mask", mask)
        cv2.waitKey(10)
        flag, img = video.read()

        '''
        # Morphplogical operations to do foreground segamentation

        # Wrapping mask
        raw_detections = []
        for i in range(max_mask_index):
            raw_detections.append(detection())
        for x in range(cols):
            for y in range(rows):
                raw_detections[mask[y, x]].max_x = max(raw_detections[mask[y, x]].max_x, x)
                raw_detections[mask[y, x]].min_x = min(raw_detections[mask[y, x]].min_x, x)
                raw_detections[mask[y, x]].max_y = max(raw_detections[mask[y, x]].max_y, y)
                raw_detections[mask[y, x]].min_y = min(raw_detections[mask[y, x]].min_y, y)

        # Filter out detections with a small area
        detections = []
        for detection in raw_detections:
            if detection.area() >= AREA_THRESHOLD:
                detections.append(detection)
        '''