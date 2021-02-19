import os, sys

import cv2 as cv

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/../")
from helpers import bbox
from pipeline.horizon_detection.IVA import IVAHorizonDetector
sys.path.append("HOG-SVM-python")
from object_detector import *


# Preparation: load models.
horizon_detector = IVAHorizonDetector()
clf = joblib.load("./HOG-SVM-python/data/models/svm.model")

# Preparation: load dataset.
dataset = sys.argv[1]
test_imgs = os.listdir(dataset)
test_imgs.sort(key= lambda x:int(x[:-4]))

# Main process
index = -1
for img_fname in test_imgs:

    index = index + 1
    if index % 100 != 0:
        continue

    img = cv.imread(os.path.join(dataset, img_fname), cv.IMREAD_GRAYSCALE)
    blurred_img = cv.medianBlur(img, 11)
    horizon_det, _ = horizon_detector.detect(img)
    detections = nms(detect(clf, img))
    for (x_tl, y_tl, _, w, h) in detections:
        if not horizon_det.checkSuppress(bbox((x_tl, y_tl, w, h))):
            cv.rectangle(img, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
        else:
            cv.rectangle(img, (x_tl, y_tl), (x_tl+w,y_tl+h), (255, 0, 0), thickness=2)
    horizon_det.render(img)
    #clone = cv2.resize(clone, (960, 540))
    #cv2.imshow("Final Detections after applying NMS", clone)
    #cv2.waitKey()
    filename, _ = os.path.splitext(img_fname)
    cv2.imwrite('./results/'+filename+"_result.jpg", img)

