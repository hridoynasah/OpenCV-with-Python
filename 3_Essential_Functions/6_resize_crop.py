import numpy as np 
import cv2 as cv 

img = cv.imread('photos/kashmir.jpg')
cv.imshow('Kashmir', img)

print(f'{img.shape}')

# resize (ignoring the aspect ratio)
resized1 = cv.resize(img, (400, 400))
cv.imshow('resized1', resized1)
# resize (ignoring the aspect ratio)
resized2 = cv.resize(img, (400, 400),  interpolation=cv.INTER_AREA)
cv.imshow('resized2', resized2)
# resize (ignoring the aspect ratio)
resized3 = cv.resize(img, (400, 400),  interpolation=cv.INTER_LINEAR)
cv.imshow('resized3', resized3)

# cropping image 
cropped = img[50:250, 250:450]
cv.imshow('Cropped', cropped)

cv.waitKey(0)