import cv2 as cv 

def rescaleFrame(frame, scale = 0.75):
    # reshaping the height by 75%
    height = int(frame.shape[0] * scale) 
    # reshaping the width by 75%
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)


img = cv.imread('llama.jpg')

resized_img = rescaleFrame(img, 0.1)

cv.imshow('Original image', img)
cv.waitKey(0)

cv.imshow('Resized image', resized_img)
cv.waitKey(0)