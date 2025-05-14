import cv2 as cv 
import numpy as np

blank_image = np.zeros((1000, 1000, 3), dtype = 'uint8')

cv.putText(blank_image, 'Hello, my name is Hridoy', (0, 225), cv.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), thickness= 2)
cv.imshow('Text', blank_image)
cv.waitKey(0)