import cv2
import numpy as np
import scipy.sparse as ss

def houghLines(image):

    houghImage = np.array((180, ), dtype=np.uint8)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[y, x] != 0:
                for theta in range(-90, 90):
                    rho = x * np.cos(theta/180*np.pi) + y * np.sin(theta/180*np.pi)
                    houghImage.

            

if __name__ == "__main__":

    img = np.zeros([300, 300]).astype(np.uint8)
    cv2.line(img, (0, 0), (100,100), 255)
    cv2.imshow('',img)

    cv2.imshow('hough',hough_img)
    cv2.waitKey()
