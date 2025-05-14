import cv2 as cv 

def rescaleFrame(frame, scale = 0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

capture = cv.VideoCapture('videos/video_1.mp4')

while True:
    isTrue, frame = capture.read()
    resized_video = rescaleFrame(frame, 0.3)
    cv.imshow('Original_Video', frame)
    cv.imshow('Resized_Video', resized_video)

    if cv.waitKey(30) & 0xFF == ord('k'):
        break

capture.release()
cv.destroyAllWindows()