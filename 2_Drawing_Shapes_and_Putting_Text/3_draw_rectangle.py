# draw a rectangle 
import cv2 as cv 
import numpy as np 

blank_image = np.zeros((500, 500, 3), dtype = 'uint8')

# cv.rectangle(blank_image,
#               (0,0), # where to start 
#               (250, 250), # where to end
#               (0, 255, 0), # color channel
#               thickness=3)
# cv.imshow('Rectangle', blank_image)

# cv.rectangle(blank_image, (150,150), (250, 250), (0, 255, 0), thickness=3)
# cv.imshow('Rectangle', blank_image)

# filled rectangle 
# cv.rectangle(blank_image, (0,0), (250,500), (0, 255, 0), thickness=cv.FILLED)
# cv.imshow('Filled_rectangle', blank_image)

# filled rectangle 
# cv.rectangle(blank_image, (0,0), (500,250), (0, 255, 0), thickness=-1)
# cv.imshow('Filled_rectangle', blank_image)

cv.rectangle(blank_image, (0, 0), (blank_image.shape[1]//2, blank_image.shape[0]//2), (0, 255, 0), thickness=-1)
cv.imshow('Filled Rectangle', blank_image)


cv.waitKey(0)