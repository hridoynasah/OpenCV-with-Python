import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

blank_img = np.zeros((600, 600), dtype = 'uint8')
cv.imshow('Blank image', blank_img)

# x-axis (100 to 500)
# y-axis (100 to 500)
rectangle = cv.rectangle(blank_img.copy(), (100, 100), (500, 500), color = 255, thickness=-1)
cv.imshow('Rectangle', rectangle)

circle = cv.circle(blank_img.copy(), (300, 300), 250, color = 255, thickness= -1)
cv.imshow('Circle', circle)

# Bitwise NOT operation
bit_not_r = cv.bitwise_not(rectangle)
cv.imshow('Bitwise XOR', bit_not_r)

# Bitwise NOT operation
bit_not_c = cv.bitwise_not(circle)
cv.imshow('Bitwise XOR', bit_not_c)

plt.imshow(bit_not_r, cmap = 'gray')
plt.show()

cv.waitKey(0)