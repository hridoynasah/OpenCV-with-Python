# draw a line

import cv2 as cv 
import numpy as np

blank_image = np.zeros((500, 500, 3), dtype = 'uint8')

# cv.line(blank_image, (100, 100), (300, 300), (255, 255, 255), thickness = 3)
cv.line(blank_image, (0, 0), (250, 250), (255, 255, 255), thickness = 3)

cv.imshow('line', blank_image)
cv.waitKey(0)