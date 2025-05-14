import cv2 as cv 

# reading a video 
# create a video capture object to read video frames 
capture = cv.VideoCapture('video_3.mp4')

# videos are read frame by frame using loop

while True:
    isTrue, frame = capture.read()
    cv.imshow('Jet Plane Video', frame)

    if cv.waitKey(20) & 0xFF == ord('k'):
        break

capture.release() # frees the video capture object
cv.destroyAllWindows() # closes all OpenCV windows created by cv.imshow()