import math

import cv2 as cv
import numpy as np

# TODO: extend cv2.Line

# All angles are in radians.
class Horizon:

    # Types of horizon line annotations

    # OpenCV HoughLines output format
    ROU_THETA = 1
    # Any point on the line and the line's slope value
    POINT_K = 2

    def __init__(self, annoType, params):
        if annoType == Horizon.ROU_THETA:
            self.point = (params[0] * math.cos(params[1]), params[0] * math.sin(params[1]))
            self.k = -math.tan(np.pi / 2 - params[1])   
        elif annoType == Horizon.POINT_K:
            self.point = params[0]
            self.k = params[1]
        else:
            assert annoType <= Horizon.POINT_K and annoType >= 0, "Invalid annotation type"

    def y(self, x):
        delta_x = x - self.point[0]
        delta_y = delta_x * self.k
        return int(self.point[1] + delta_y)

    def render(self, img, color=(255, 0, 0)):
        A = (0, self.y(0))
        B = (img.shape[1] - 1, self.y(img.shape[1] - 1))
        if abs(A[1]) > 1e4 or abs(B[1]) > 1e4:
            return False
        cv.line(img, A, B, color)
        return True

    def checkSuppress(self, det):
        Ay = self.y(det.min_x)
        if Ay >= det.min_y and Ay <= det.max_y:
            return False
        By = self.y(det.max_x)
        if By >= det.min_y and By <= det.max_y:
            return False
        return True

    def toYAlpha(self, maxX):
        return(self.y(maxX // 2), -math.atan(self.k))

if __name__ == "__main__":
    l1 = Horizon(Horizon.ROU_THETA, ((200 + 300 * math.tan(np.pi / 6)) * math.cos(np.pi / 6), np.pi / 3))
    l2 = Horizon(Horizon.POINT_K, ((300, 200), 0.5))
    # SMD Annotation
    l3 = Horizon(Horizon.POINT_K, ((300, 200), math.tan(np.pi / 6)))
    img = np.zeros((400, 600, 3), np.uint8)
    l1.render(img)
    l2.render(img)
    l3.render(img)
    cv.imshow("debug", img)
    cv.waitKey()