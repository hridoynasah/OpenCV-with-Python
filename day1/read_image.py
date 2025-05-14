import cv2 as cv 
import numpy as np

# read image 
# returns the image as a matrix of pixels(numpy.ndarray)
img = cv.imread('cat_small.jpg')

# print(type(img))  # output: <class 'numpy.ndarray'>
# print(img.shape, img.ndim) # output: (183, 275, 3) 3

cv.imshow('Black Cat', img)

cv.waitKey(0) 