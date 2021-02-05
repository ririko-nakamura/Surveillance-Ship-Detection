import cv2
import numpy as np

class HorizonDetector:

    def __init__(self, scales):
        self.scales = scales

    def detect(self, inputImage):
        filteredImages = self.filterMultiscaleImages(inputImage)
        meanFilteredImage = np.mean(filteredImages, axis=0)

        horizonCandidates = []
        for s in range(len(filteredImages)):
            edgeMap = cv2.Canny(filteredImages[s], 10, 100)
            lines = np.array([])
            cv2.HoughLines(edgeMap, 1, np.pi / 100, 150, lines)
            print(lines)
            print(lines.shape)
            horizonCandidates.extend(lines[0:10])

    
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
            filteredImages.append(medianBlur(inputImage, [2*s, 0]))
        return filteredImages


if __name__ == "__main__":

    horizonDetector = HorizonDetector(4)

    # test median blur function and multiscale filter images build
    if False:
        sample = np.arange(20*20).reshape((20, 20))
        sample = horizonDetector.filterMultiscaleImages(sample)
        for i in range(len(sample)):
            print(sample[i])

    # 
    if True:
        image = cv2.imread("Malta - Birzebbuga - Harbour (San Gorg) 01 ies.jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (0, 0), None, 0.25, 0.25)
        horizonDetector.detect(image)