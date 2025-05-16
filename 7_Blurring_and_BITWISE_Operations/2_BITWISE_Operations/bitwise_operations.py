import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

# Bitwise operations
bit_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bit_and)

bit_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bit_or)

bit_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bit_xor)

bit_not = cv.bitwise_not(rectangle)
cv.imshow('Bitwise NOT', bit_not)

cv.waitKey(0)
cv.destroyAllWindows()