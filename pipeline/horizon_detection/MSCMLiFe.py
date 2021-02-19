import math
import threading

import cv2
import numpy as np

from .IVA import IVAHorizonDetector
from ..general import Horizon

class MSCMLiFeHorizonDetector:

    def __init__(self, scales):
        self.scales = scales
        self.IVADetector = IVAHorizonDetector()

    def detect(self, inputImage):
        filteredImages = self.filterMultiscaleImages(inputImage)
        meanFilteredImage = np.mean(filteredImages, axis=0)

        # Calculate horizon candidates
        houghCandidates = []
        houghScores = []
        IVACandidates = []
        IVAScores = []
        for s in range(len(filteredImages)):
            # Calculate Hough transform candidates
            #visiblization = filteredImages[s].copy()
            edgeMap = cv2.Canny(filteredImages[s], 10, 100)
            lines = cv2.HoughLinesWithAccumulator(edgeMap, 1, np.pi / 100, 100)
            for i in range(min(10, len(lines))):
                houghCandidates.append(Horizon(Horizon.ROU_THETA, (lines[i, 0, 0], lines[i, 0, 1])))
                houghScores.append(lines[i, 0, 2])
                #houghCandidates[-1].render(visiblization, 255)
            # Calculate IVA candidates
            line, score = self.IVADetector.detect(filteredImages[s])
            IVACandidates.append(line)
            IVAScores.append(score)
            #line.render(visiblization, 0)
            #cv2.imshow("detections", visiblization)
            #cv2.waitKey(1000)

        # Evaluate each pair of (houghCandidate, IVACandidate)
        bestDecision = None
        bestScore = -1
        for n in range(len(houghCandidates)):
            for s in range(len(IVACandidates)):
                score = HorizonDetector.evaluateDetectionPair(
                    houghCandidates[n],
                    IVACandidates[s], 
                    houghScores[n],
                    IVAScores[s],
                    filteredImages[s].shape[1],
                    filteredImages[s].shape[0]
                )
                if score > bestScore:
                    bestDecision = houghCandidates[n]
                    bestScore = score

        return bestDecision

    # TODO: Optimize median blur algorithm
    def filterMultiscaleImages(self, inputImage):
        # Helper function to apply median blur to an image
        # It copies the original image and blurs it
        def medianBlur(inputArray, kernelSize):
            assert(len(inputArray.shape) == len(kernelSize))
            # Helper function to do median blur in a ROI
            # Ranges are in [min, max)
            def ROIMedian(inputArray, xMin, xMax, yMin, yMax):
                buffer = []
                for x in range(xMin, xMax):
                    for y in range(yMin, yMax):
                        buffer.append(inputArray[x, y])
                buffer.sort()
                return buffer[len(buffer)//2]
            # medianBlur() procedure
            array = inputArray.copy()
            rows = array.shape[0]
            cols = array.shape[1]
            for x in range(rows):
                for y in range(cols):
                    xMin = max(0, x - 2*kernelSize[0])
                    xMax = min(rows, x + 2*kernelSize[0] + 1)
                    yMin = max(0, y - 2*kernelSize[1])
                    yMax = min(cols, y + 2*kernelSize[1] + 1)
                    array[x, y] = ROIMedian(array, xMin, xMax, yMin, yMax)
            return array

        # filterMultiscaleImages() procedure
        filteredImages = []
        for s in range(self.scales):
            filteredImage = medianBlur(inputImage, [2*s, 0])
            filteredImages.append(filteredImage)
        return filteredImages

    @staticmethod
    def evaluateDetectionPair(n, s, Hn, Ss, maxX, maxY):
        goodness = Hn * Ss
        yN, alphaN = n.toYAlpha(maxX)
        yS, alphaS = s.toYAlpha(maxX)
        promixity = (1 - pow((yN - yS) / maxY, 2)) * pow(math.cos(alphaN - alphaS), 2)
        return goodness * promixity

