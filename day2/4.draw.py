# draw a circle 

import cv2 as cv 
import numpy as np

blank_image = np.zeros((500, 500, 3), dtype = 'uint8')

# cv.circle(blank_image, (250, 250), 30,(0, 255, 0), thickness= 3)
# cv.circle(blank_image, (100, 100), 50, (255, 0, 0), thickness= 3)
cv.circle(blank_image, (150, 150), 50, (255, 255, 255), thickness= -1) # Filled circle 

cv.imshow('Circle', blank_image)
cv.waitKey(0)