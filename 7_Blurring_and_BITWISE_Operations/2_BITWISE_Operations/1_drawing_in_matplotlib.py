import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create a blank grayscale image (400x400)
blank = np.zeros((400, 400), dtype='uint8')

# Display the blank image using OpenCV
cv.imshow('Blank', blank)

# Draw a rectangle on a copy of the blank image
# x axis 50 --> 350 & y axis 100 --> 300
rectangle = cv.rectangle(blank.copy(), 
                         (50, 100), # (x, y)
                         (350, 300), # (x, y) 
                         color=255, 
                         thickness=-1)

cv.imshow('Rectangle', rectangle)

circle = cv.circle(blank.copy(), (200, 200), 200 ,color = 255, thickness= 2)
cv.imshow('Circle', circle)

# Display the rectangle image using Matplotlib
plt.imshow(rectangle, cmap='gray')
plt.title('Rectangle')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

# Display the circle image using Matplotlib 
plt.imshow(circle, cmap = 'gray')
plt.title('Rectangle')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()



# Wait for a key press and close OpenCV windows
cv.waitKey(0)
cv.destroyAllWindows()