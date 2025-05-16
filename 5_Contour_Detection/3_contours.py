# contours with blur 

import cv2 as cv 
import numpy as np

img = cv.imread('kashmir.jpg')
cv.imshow('Kashmir', img)


gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray_img', gray_img)

ret, thresh = cv.threshold(gray_img, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

canny_img = cv.Canny(thresh, 125, 175)
cv.imshow('canny_img', canny_img)

contours, hierarchies = cv.findContours(canny_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# print(contours, type(contours))
# print(hierarchies, type(hierarchies))
print(len(contours))

cv.waitKey(0)