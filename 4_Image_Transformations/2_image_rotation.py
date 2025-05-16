import cv2 as cv
import numpy as np

img = cv.imread('cat_large.jpg')
cv.imshow('Cat', img)

# print(img.shape[:2])
# print(img.shape)
# print(type(img.shape))

def rotate(image, angle, rotPoint=None):
    (height, width) = image.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(image, rotMat, dimensions)


rotated = rotate(img, 45)
cv.imshow('rotated', rotated)

cv.waitKey(0)