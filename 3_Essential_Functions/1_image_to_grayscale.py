import cv2 as cv 
import numpy as np

img = cv.imread('photos/cat_large.jpg')

print(img.shape)

cv.imshow('Cat_BGR', img)

# Converting the image to gray scale

gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray_Cat', gray_image)

# Converting the image to HSV scale 
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV_image', hsv_image)

cv.waitKey()