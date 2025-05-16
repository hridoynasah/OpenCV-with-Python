import cv2 as cv 

img = cv.imread('cat_large.jpg')
cv.imshow('Cat', img)

flipped = cv.flip(img, 1) 
cv.imshow('Flipped', flipped)

cv.waitKey(0)