import cv2 as cv

# by default BGR
img = cv.imread('kashmir.jpg')
cv.imshow('BGR', img)

# BGR to hsv (HSV to BGR possible)
hue_saturated_value = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Hue Saturated Value', hue_saturated_value)

# BGR to hsv (HSV to BGR possible)
bgr = cv.cvtColor(hue_saturated_value, cv.COLOR_HSV2BGR)
cv.imshow('Hue Saturated Value ---> BGR', bgr)

cv.waitKey(0)