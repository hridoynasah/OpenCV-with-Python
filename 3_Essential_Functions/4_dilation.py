import cv2 as cv 

img = cv.imread('photos/kashmir.jpg')
cv.imshow('Kashmir', img)

# Canny edges 
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)

# Dilate an image (Increase the thickness of canny edges)
# more iterations more thick 
dilated = cv.dilate(canny, (3, 3), iterations = 7)
cv.imshow('Dilated', dilated)

cv.waitKey()