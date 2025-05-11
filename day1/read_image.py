import cv2 as cv

img = cv.imread('llama.jpg')

cv.imshow('Cat', img)
cv.waitKey(0)
