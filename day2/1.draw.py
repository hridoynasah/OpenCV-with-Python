import cv2 as cv
import numpy as np

# create a blank image 
blank_image = np.zeros((500, 500, 3), dtype = 'uint8')
cv.imshow('Blank',blank_image)


# print(blank_image.shape) # output: (500, 500)
# print(blank_image.ndim) # output: 2
# print(blank_image[:])

# # Paint the image a certain color 
# blank_image[:] = 0, 255, 0 # color channels (green)
# cv.imshow('Green',blank_image)

# Paint the image a certain color 
# blank_image[:] = 255, 0, 0 # color channels (blue)
# cv.imshow('Blue',blank_image)

# # Paint the image a certain color 
# blank_image[:] = 0, 0, 255 # color channels (Red)
# cv.imshow('Red',blank_image)

# # Paint the image a certain color 
# blank_image[:] = 255, 255, 255 # color channels (white)
# cv.imshow('white',blank_image)


# Paint the image a certain color 
blank_image[:] = 0, 0, 0 # color channels (black)
cv.imshow('Black',blank_image)

cv.waitKey(0)