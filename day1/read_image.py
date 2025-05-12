import cv2 as cv

img = cv.imread('large_cat.jpg')

cv.imshow('Cat', img)
cv.waitKey(0)

