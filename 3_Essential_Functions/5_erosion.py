import cv2 as cv 

img = cv.imread('photos/kashmir.jpg')
cv.imshow('Kashmir', img)

# blur the image 
blur_img = cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)
cv.imshow('Kahsmir blur', blur_img)

# Canny edges 
canny = cv.Canny(blur_img, 125, 175)
cv.imshow('Canny', canny)

# dilation (scaling up edges)
dilated = cv.dilate(canny, (3, 3), iterations = 3)
cv.imshow('Dilated', dilated)

# Erosion (scaling down edges)
erosion = cv.erode(dilated, (3,3), iterations= 3)
cv.imshow('Eroded', erosion)

cv.waitKey(0)