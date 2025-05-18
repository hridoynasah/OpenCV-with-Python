import cv2 as cv

img = cv.imread('kashmir.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gray', gray)


ret , thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
print(ret, thresh)

cv.imshow('Binary Threshold', thresh)

ret_inv , thresh_inv = cv.threshold(gray, 50, 155, cv.THRESH_BINARY_INV)

cv.imshow('Binary Threshold Inverse', thresh_inv)

cv.waitKey()