import os, sys
import pickle as pkl

import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from helpers import bbox
from pipeline.horizon_detection.MSCMLiFe import MSCMLiFeHorizonDetector
sys.path.append("HOG-SVM-python")
from object_detector import *


# Preparation: load models.
horizon_detector = MSCMLiFeHorizonDetector(range(4, 10))
clf = joblib.load("./HOG-SVM-python/data/models/svm.model")

# Preparation: load dataset.
dataset = os.path.join(sys.argv[1], 'test')
test_imgs = os.listdir(dataset)
test_imgs.sort(key= lambda x:int(x[:-4]))

# Main process
index = 0
for img_fname in test_imgs:

    index = index + 1
    if index % 20 != 1:
        continue

    img = cv.imread(os.path.join(dataset, img_fname), cv.IMREAD_GRAYSCALE)
    horizon_det, _ = horizon_detector.detect(img)
    if horizon_det is None:
        continue

    detections = detect(clf, img)

    flag = horizon_det.render(img)

    final_dets = []
    for (x_tl, y_tl, _, w, h) in detections:
        det = bbox((x_tl, y_tl, w, h))
        if flag and not horizon_det.checkSuppress(det):
            cv.rectangle(img, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
        elif not flag and (w != 100 or h != 40):
            cv.rectangle(img, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
        else:
            cv.rectangle(img, (x_tl, y_tl), (x_tl+w,y_tl+h), (255, 0, 0), thickness=2)
        final_dets.append([x_tl, y_tl, w, h])
    
    filename, _ = os.path.splitext(img_fname)
    cv2.imwrite('./results/'+filename+"_mscm_life_result.jpg", img)
    with open('./results/'+filename+"_mscm_life_detections.pkl", "wb") as f:
        pkl.dump(final_dets, f)
    with open('./results/'+filename+"_mscm_life_horizon.pkl", "wb") as f:
        pkl.dump(horizon_det, f)        

    # 思路：拟合bbox底边的坐标

    # TODO: If no horizon, supress all det with standard size

