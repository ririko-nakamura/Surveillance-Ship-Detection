import os, sys

import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../")
from pipeline.horizon_detection.MSCM_LiFe import HorizonDetector

horizonDetector = HorizonDetector(10)

# test median blur function and multiscale filter images build
if False:
    sample = np.arange(20*20).reshape((20, 20))
    sample = horizonDetector.filterMultiscaleImages(sample)
    for i in range(len(sample)):
        print(sample[i])

if True:
    for i in range(1000, 2001, 100):
        image = cv2.imread("./data/VIS_Onshore/test/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
        horizon = horizonDetector.detect(image)
        horizon.render(image)
        cv2.imwrite("./results/test_{}_horizon_detection.jpg".format(i), image)