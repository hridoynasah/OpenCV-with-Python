import cv2 as cv 
import numpy as np

kashmir = cv.imread('photos/Kashmir.jpg')
cv.imshow('Kashmir', kashmir)

# Blur the image -> ksize must be even (higher the ksize, higher blur)
kashmir_blur = cv.GaussianBlur(kashmir, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blur Kashmir', kashmir_blur)

# high blur 
kashmir_blur_2 = cv.GaussianBlur(kashmir, (11, 11), cv.BORDER_DEFAULT)
cv.imshow('Blur Kashmir 2', kashmir_blur_2)

cv.waitKey(0)