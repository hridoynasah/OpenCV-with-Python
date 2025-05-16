import cv2 as cv
import matplotlib.pyplot as plt

# by default BGR
img = cv.imread('kashmir.jpg')
cv.imshow('BGR', img)

# it can't give the actual image seems it refers RGB
plt.imshow(img)
plt.show()