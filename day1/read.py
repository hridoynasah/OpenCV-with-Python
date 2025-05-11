import cv2 as cv

img = cv.imread('\cat_1.jpg')

cv.imshow('Cat', img)
cv.waitKey(0)
