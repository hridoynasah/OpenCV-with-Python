import cv2 as cv 
import numpy as np

blank_image = np.zeros((500, 500, 3), dtype = 'uint8')

blank_image[125:250, 250:375] = 0, 0, 255
cv.imshow('Green', blank_image)

cv.waitKey(0)