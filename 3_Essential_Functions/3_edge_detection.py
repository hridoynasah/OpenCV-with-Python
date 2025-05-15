import cv2 as cv

kashmir = cv.imread('photos/kashmir.jpg')
cv.imshow('Kashmir', kashmir)

# Edge dectection
canny = cv.Canny(kashmir,125, 175)
cv.imshow('Canny Edges', canny)

# blur the image 
blur_kashmir = cv.GaussianBlur(kashmir, (9,9), cv.BORDER_DEFAULT)

blur_canny = cv.Canny(blur_kashmir, 125, 175)
cv.imshow('Blur canny edges', blur_canny)

cv.waitKey()