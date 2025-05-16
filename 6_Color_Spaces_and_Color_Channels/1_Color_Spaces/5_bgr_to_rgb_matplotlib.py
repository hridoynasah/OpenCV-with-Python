import cv2 as cv
import matplotlib.pyplot as plt

# by default BGR
img = cv.imread('kashmir.jpg')
cv.imshow('BGR', img)

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('lab', rgb)

# it can't display the actual, seems it refers RGB
plt.imshow(rgb)
plt.show()
