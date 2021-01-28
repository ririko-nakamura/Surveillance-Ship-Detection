import cv2
import nuumpy as np

class HorizonDetector:

    maxScaleIndex = 10

    def __init__(self):
        pass

    def detect(self, inputImage):
        filteredImages = self.filterMultiscaleImages(inputImage)

    
    # TODO: Optimize median blur algorithm
    def filterMultiscaleImages(self, inputImage):
        # Helper function to apply median blur to an iamge
        def medianBlur(inputImage, kernelSize):
            # Helper function to do median blur in a ROI
            # Ranges are in [min, max)
            def ROIMedian(image, xMin, xMax, yMin, yMax):
                buffer = []
                for x in range(xMin, xMax):
                    for y in range(yMin, yMax):
                        buffer.append(image[x, y])
                buffer.sort()
                return buffer[len(buffer)/2]
            # medianBlur() procedure
            image = inputImage.copy()
            for x in range(image.cols):
                for y in range(image.rows):
                    xMin = max(0, x - 2*kernelSize[0])
                    xMax = min(image.rows, x + 2*kernelSize[0])
                    yMin = max(0, y - 2*kernelSize[1])
                    yMax = min(image.cols, y + 2*kernelSize[1])
                    image[x, y] = ROIMedian(image, x, x, yMin, yMax)
            return image

        # filterMultiscaleImages() procedure
        filteredImages = []
        for s in range(HorizonDetector.maxScaleIndex+1):
            filteredImages.append(medianBlur(inputImage, [0, 2*s]))
        return filteredImages