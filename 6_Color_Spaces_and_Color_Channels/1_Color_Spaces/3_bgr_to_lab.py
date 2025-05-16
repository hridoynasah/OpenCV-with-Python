import cv2 as cv

# by default BGR
img = cv.imread('kashmir.jpg')
cv.imshow('BGR', img)

# BGR to lab (lab to BGR possible)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('lab', lab)

cv.waitKey(0)