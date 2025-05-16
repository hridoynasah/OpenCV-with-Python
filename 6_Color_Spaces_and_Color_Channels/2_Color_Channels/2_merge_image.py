import cv2 as cv
import numpy as np

img = cv.imread('kashmir.jpg')
cv.imshow('kashmir', img)

# split image color channels 
b, g, r = cv.split(img)

cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

# merge image color channels 
merged = cv.merge((b, g, r))
cv.imshow('Merged', merged)


cv.waitKey(0)