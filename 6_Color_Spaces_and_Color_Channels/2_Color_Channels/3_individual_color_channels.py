import cv2 as cv
import numpy as np

img = cv.imread('kashmir.jpg')
cv.imshow('kashmir', img)

b, g, r = cv.split(img)

blank = np.zeros(img.shape[:2], dtype='uint8')

blue = cv.merge((b, blank, blank))
green = cv.merge((blank, g, blank))
red = cv.merge((blank, blank, r))

cv.imshow('Blue Channel', blue)
cv.imshow('Green Channel', green)
cv.imshow('Red Channel', red)

cv.waitKey(0)
