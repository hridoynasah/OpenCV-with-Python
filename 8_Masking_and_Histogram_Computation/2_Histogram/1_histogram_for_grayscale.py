import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('kashmir.jpg')
cv.imshow('Kashmir', img)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image',gray_img)

# grayscale histogram 
gray_hist = cv.calcHist([gray_img], [0], None, [256], [0, 256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

# cv.waitKey(0)