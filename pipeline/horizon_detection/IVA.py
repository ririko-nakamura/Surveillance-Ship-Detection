import cv2
import numpy as np

#sys.path.append('C:/Workspace/Surveillance-Ship-Detection/pipeline/horizon-detection')
from ..general import Horizon

class IVAHorizonDetector:

    def detect(self, inputImage):

        # Array of (x, y'(x))
        maxVaritionPoints = []
        totalVarition = 0
        for x in range(inputImage.shape[1]):
            maxVarition = -1
            maxVaritionY = -1
            for y in range(1, inputImage.shape[0]):
                varition = abs(int(inputImage[y, x]) - int(inputImage[y-1, x]))
                if varition > maxVarition:
                    maxVarition = varition
                    maxVaritionY = y
                totalVarition = totalVarition + varition
            maxVaritionPoints.append((x, maxVaritionY))

        horizonLine = cv2.fitLine(np.array(maxVaritionPoints), cv2.DIST_L2, 0, 0.01, 0.01)
        point = (int(horizonLine[2, 0]), int(horizonLine[3, 0]))
        k = - horizonLine[1, 0] / horizonLine[0, 0]
        horizonLine = Horizon(Horizon.POINT_K, (point, k))

        return horizonLine, totalVarition / inputImage.shape[1] / (inputImage.shape[0] - 1)
    

