import cv2 as cv

# by default BGR
img = cv.imread('kashmir.jpg')
cv.imshow('BGR', img)

# BGR to Grayscale (Grayscale to BGR possible)
grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', grayscale)

# Grayscale to BGR
bgr = cv.cvtColor(grayscale, cv.COLOR_GRAY2BGR)
cv.imshow('Grayscale ---> BGR', bgr)

cv.waitKey(0)