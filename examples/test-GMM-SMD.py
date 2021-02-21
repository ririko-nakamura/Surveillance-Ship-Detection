import os, sys
import pickle as pkl

import cv2 as cv
import numpy as np

threshold = 0.5
sample_rate = 0.01

dataset = sys.argv[1]

test_videos = [
    "MVI_1474_VIS.avi",
    "MVI_1484_VIS.avi",
    "MVI_1486_VIS.avi",
    "MVI_1582_VIS.avi",
    "MVI_1612_VIS.avi",
    "MVI_1626_VIS.avi",
    "MVI_1627_VIS.avi",
    "MVI_1640_VIS.avi"
]

with open("GMM.model", "rb") as f:
    model = pkl.load(f)

i = 0

for filename in test_videos:

    video = cv.VideoCapture(os.path.join(dataset, "VIS_Onshore/Videos", filename))

    for index in range(int(video.get(cv.CAP_PROP_FRAME_COUNT))):

        i = i + 1
        if i % int(1 / sample_rate) != 1:
            continue
        
        ret, img = video.read()
        if not ret:
            break
        img = cv.medianBlur(img, 3)
        probs = model.predict_proba(img.reshape(-1, 3))

        mask = []
        for distribution in probs:
            max_prob = distribution.max()
            max_index = np.asarray(abs(distribution - max_prob) <= 1e-5).nonzero()[0][0]
            mask.append(max_index if max_prob > threshold else -1)

        mask = np.array(mask).reshape(img.shape[0:2])

        mask = mask + 1
        mask = (mask / mask.max() * 255).astype(np.uint8)
        cv.imshow("mask", cv.resize(mask, (640, 360)))
        cv.waitKey(1)

     