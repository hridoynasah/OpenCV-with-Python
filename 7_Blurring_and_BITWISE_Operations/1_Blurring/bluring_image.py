import cv2 as cv

img = cv.imread('kashmir.jpg')

# Average blur
avg = cv.blur(img, (5, 5))
cv.imshow('Average Blur', avg)

# Gaussian blur
gauss = cv.GaussianBlur(img, (5, 5), 0)
cv.imshow('Gaussian Blur', gauss)

# Median blur
median = cv.medianBlur(img, 5)
cv.imshow('Median Blur', median)

# Bilateral blur
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)
cv.destroyAllWindows()