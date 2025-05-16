# Visualize contour image 
import cv2 as cv 
import numpy as np 

img = cv.imread('kashmir.jpg')
cv.imshow('Kashmir', img)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blank_img = np.zeros((img.shape), dtype = 'uint8')
cv.imshow('blank_img', blank_img)

ret, thresh = cv.threshold(gray_img, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.drawContours(blank_img, contours, -1, (0, 0, 255), 1)
cv.imshow('Drawn Contours', blank_img)

cv.waitKey()