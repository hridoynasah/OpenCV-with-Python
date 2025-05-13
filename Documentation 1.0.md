#### Part 1: Introduction

# Introduction to Python and OpenCV Course

## Overview
This course provides a comprehensive introduction to computer vision using the OpenCV library in Python. It covers everything from basic image and video processing to advanced techniques like face detection, face recognition, and building a deep learning model for character classification.

## Course Structure
- **Basics**: Learn to read images/videos, manipulate them with transformations, draw shapes, and apply essential OpenCV functions.
- **Advanced**: Explore color spaces, bitwise operations, masking, histograms, edge detection, and thresholding.
- **Faces**: Implement face detection and recognition using Haar cascades and OpenCV's built-in recognizer.
- **Capstone**: Build a deep learning model to classify characters from *The Simpsons* using TensorFlow and Keras.

## Prerequisites
- **Python**: Ensure Python 3.7 or higher is installed. Check with:
  ```bash
  python --version
  ```
  Download the latest version from [python.org](https://www.python.org) if needed.
- Basic familiarity with Python programming.

## Resources
- All code and materials are available on the instructor’s GitHub page (link in the video description).
- Subscribe to the instructor’s channel for updates.

## What is Computer Vision?
Computer vision is a field of deep learning focused on extracting insights from images and videos. OpenCV is a powerful library available in Python, C++, and Java, designed for computer vision tasks.

## Next Steps
In the next section, we’ll install the necessary packages and set up the environment for the course.


---

#### Part 2: Installing OpenCV and Caer

# Installing OpenCV and Caer

## Overview
This section covers the installation of OpenCV and Caer, essential packages for the course. OpenCV is the primary library for computer vision, while Caer is a utility package to streamline workflows.

## Prerequisites
- Python 3.7 or higher installed.
- pip (Python package manager).

## Installation Steps
1. **Install OpenCV**:
   - Use the command:
     ```bash
     pip install opencv-contrib-python
     ```
   - **Explanation**: The `opencv-contrib-python` package includes the main OpenCV module and additional community-contributed modules, providing full functionality. Avoid installing only `opencv-python`, as it lacks the contrib modules.

2. **Install NumPy**:
   - NumPy is automatically installed as a dependency of OpenCV, but you can ensure it’s installed with:
     ```bash
     pip install numpy
     ```
   - **Explanation**: NumPy is a scientific computing package used for matrix and array manipulations, essential for image processing tasks in OpenCV.

3. **Install Caer**:
   - Use the command:
     ```bash
     pip install caer
     ```
   - **Explanation**: Caer is a custom package created by the instructor, offering helper functions to speed up computer vision workflows. It’s used primarily in the final video for building a deep learning model but is recommended to install now to avoid future setup issues.
   - **Note**: Explore or contribute to Caer’s codebase on the instructor’s GitHub (link in the description).

## Verification
- Check Python version:
  ```bash
  python --version
  ```
- Ensure OpenCV is installed:
  ```python
  import cv2
  print(cv2.__version__)
  ```

## Next Steps
With the environment set up, the next section will cover reading images and videos in OpenCV.


---

#### Part 3: Reading Images & Video

# Reading Images and Video in OpenCV

## Overview
This tutorial explains how to read and display images and videos using OpenCV, a foundational skill for computer vision tasks.

## Reading Images
1. **Function**: `cv2.imread(path)`
   - **Purpose**: Reads an image from a specified file path and returns it as a matrix of pixels.
   - **Parameters**:
     - `path`: String path to the image file (e.g., `'photos/cat.jpg'`). Can be relative (if in the current directory) or absolute.
   - **Example**:
     ```python
     import cv2 as cv
     img = cv.imread('photos/cat.jpg')
     ```
   - **Explanation**: The image is loaded as a NumPy array, where each element represents a pixel’s color values (BGR format for color images).

2. **Displaying the Image**: `cv2.imshow(window_name, image)`
   - **Purpose**: Displays the image in a new window.
   - **Parameters**:
     - `window_name`: String name for the display window (e.g., `'Cat'`).
     - `image`: The image matrix (e.g., `img`).
   - **Example**:
     ```python
     cv.imshow('Cat', img)
     ```
   - **Note**: Use `cv2.waitKey(0)` to keep the window open until a key is pressed:
     ```python
     cv.waitKey(0)
     ```
     - `0`: Waits indefinitely for a key press.
     - Non-zero values (e.g., `20`): Waits for the specified milliseconds.

3. **Handling Large Images**:
   - If an image’s dimensions exceed the screen size (e.g., 2400x1600), it may go off-screen. OpenCV lacks built-in scaling for display, but resizing techniques are covered in the next video.

4. **Error Handling**:
   - If the image path is incorrect, OpenCV raises a `-215 Assertion Failed` error, indicating the file couldn’t be found.

## Reading Videos
1. **Function**: `cv2.VideoCapture(source)`
   - **Purpose**: Creates a video capture object to read video frames.
   - **Parameters**:
     - `source`: Either an integer (e.g., `0` for webcam) or a file path (e.g., `'videos/dog.mp4'`).
     - Integer values:
       - `0`: Default webcam.
       - `1`, `2`, etc.: Additional cameras.
   - **Example**:
     ```python
     capture = cv.VideoCapture('videos/dog.mp4')
     ```

2. **Reading Frames**:
   - Videos are read frame by frame in a loop using `capture.read()`:
     ```python
     while True:
         isTrue, frame = capture.read()
         cv.imshow('Video', frame)
     ```
   - **Returns**:
     - `isTrue`: Boolean indicating if the frame was read successfully.
     - `frame`: The current video frame as a matrix.
   - **Note**: When the video ends, `capture.read()` returns `False`, causing an error unless handled.

3. **Controlling Playback**:
   - Use `cv2.waitKey(milliseconds)` to control frame display duration and allow user input:
     ```python
     if cv.waitKey(20) & 0xFF == ord('d'):
         break
     ```
     - `20`: Displays each frame for 20ms.
     - `0xFF == ord('d')`: Breaks the loop if the ‘d’ key is pressed.

4. **Cleanup**:
   - Release the capture object and close windows:
     ```python
     capture.release()
     cv.destroyAllWindows()
     ```

5. **Error Handling**:
   - A `-215 Assertion Failed` error occurs if the video file path is incorrect or no more frames are available.

## Example Code
```python
import cv2 as cv

# Read and display image
img = cv.imread('photos/cat.jpg')
cv.imshow('Cat', img)
cv.waitKey(0)

# Read and display video
capture = cv.VideoCapture('videos/dog.mp4')
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
```

## Key Points
- Images are read as pixel matrices and displayed in windows.
- Videos are processed frame by frame, requiring a loop and cleanup.
- Use relative paths for files in the project directory to avoid errors.
- Handle large images and video end conditions to prevent issues.

## Next Steps
The next section covers resizing and rescaling images and video frames to manage large media files effectively.


---

#### Part 4: Resizing and Rescaling Frames

# Resizing and Rescaling Frames in OpenCV

## Overview
This tutorial explains how to resize and rescale images and video frames in OpenCV to reduce computational strain and fit display constraints.

## Why Resize/Rescale?
- **Computational Efficiency**: Large media files (e.g., high-resolution images/videos) require significant processing power. Resizing reduces the data size.
- **Display Compatibility**: Prevents images/videos from exceeding screen dimensions.

## Rescaling Images and Videos
1. **Custom Function**: `rescaleFrame(frame, scale=0.75)`
   - **Purpose**: Scales an image or video frame by a specified factor.
   - **Parameters**:
     - `frame`: The input image or video frame (NumPy array).
     - `scale`: Scaling factor (e.g., `0.75` for 75% of original size, `0.2` for 20%).
   - **Implementation**:
     ```python
     def rescaleFrame(frame, scale=0.75):
         width = int(frame.shape[1] * scale)  # frame.shape[1] is width
         height = int(frame.shape[0] * scale)  # frame.shape[0] is height
         dimensions = (width, height)
         return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
     ```
     - **Explanation**:
       - `frame.shape`: Returns `(height, width, channels)` for the frame.
       - Calculates new dimensions by multiplying original width and height by `scale`.
       - `cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)`:
         - Resizes the frame to the new `dimensions` (width, height).
         - `cv.INTER_AREA`: Interpolation method suitable for shrinking images (covered later).

2. **Example for Images**:
   ```python
   img = cv.imread('photos/cat.jpg')
   resized_img = rescaleFrame(img, scale=0.75)
   cv.imshow('Resized Image', resized_img)
   cv.waitKey(0)
   ```
   - Scales the image to 75% of its original size.

3. **Example for Videos**:
   ```python
   capture = cv.VideoCapture('videos/dog.mp4')
   while True:
       isTrue, frame = capture.read()
       if not isTrue:
           break
       resized_frame = rescaleFrame(frame, scale=0.2)
       cv.imshow('Resized Video', resized_frame)
       if cv.waitKey(20) & 0xFF == ord('d'):
           break
   capture.release()
   cv.destroyAllWindows()
   ```
   - Scales each video frame to 20% of its original size.

## Resizing Live Video
1. **Function**: `capture.set(property_id, value)`
   - **Purpose**: Adjusts properties of a video capture device (e.g., webcam).
   - **Parameters**:
     - `property_id`: Integer representing the property (e.g., `3` for width, `4` for height, `10` for brightness).
     - `value`: The desired value for the property.
   - **Example**:
     ```python
     def changeRes(capture, width, height):
         capture.set(3, width)
         capture.set(4, height)
     ```
     - Sets the resolution of a live video feed (e.g., webcam).
   - **Note**: Only works for live video (e.g., webcam), not pre-recorded video files.

2. **Usage**:
   ```python
   capture = cv.VideoCapture(0)  # Webcam
   changeRes(capture, 640, 480)
   while True:
       isTrue, frame = capture.read()
       if not isTrue:
           break
       cv.imshow('Webcam', frame)
       if cv.waitKey(20) & 0xFF == ord('d'):
           break
   capture.release()
   cv.destroyAllWindows()
   ```
   - Sets webcam resolution to 640x480.

## Key Points
- **RescaleFrame**: A versatile function for scaling both images and videos, using `cv.resize`.
- **capture.set**: Specific to live video, adjusts resolution directly on the capture device.
- Always downscale to smaller dimensions than the original, as upscaling beyond a camera’s capability (e.g., 720p to 1080p) is not supported.
- Interpolation methods (e.g., `cv.INTER_AREA`) affect resizing quality, covered in later videos.

## Next Steps
The next section explores drawing shapes and adding text to images in OpenCV.


---

#### Part 5: Drawing Shapes & Putting Text

# Drawing Shapes and Putting Text in OpenCV

## Overview
This tutorial covers how to draw shapes (rectangles, circles, lines) and add text to images in OpenCV, useful for annotations and visualizations.

## Setup
- Import OpenCV and NumPy:
  ```python
  import cv2 as cv
  import numpy as np
  ```

## Creating a Blank Image
1. **Function**: `np.zeros(shape, dtype)`
   - **Purpose**: Creates a blank image (black) for drawing.
   - **Parameters**:
     - `shape`: Tuple of `(height, width, channels)` (e.g., `(500, 500, 3)` for a 500x500 RGB image).
     - `dtype`: Data type (e.g., `np.uint8` for 8-bit unsigned integers, standard for images).
   - **Example**:
     ```python
     blank = np.zeros((500, 500, 3), dtype='uint8')
     cv.imshow('Blank', blank)
     cv.waitKey(0)
     ```
   - **Explanation**: Creates a 500x500 RGB image, all pixels initialized to zero (black).

## Painting the Image
1. **Method**: Array slicing and assignment
   - **Purpose**: Sets pixel values to a specific color.
   - **Example**:
     ```python
     blank[:] = 0, 255, 0  # Green (BGR format)
     cv.imshow('Green', blank)
     cv.waitKey(0)
     ```
     - **Explanation**: Sets all pixels to green (0, 255, 0 in BGR).
   - **Partial Painting**:
     ```python
     blank[200:300, 300:400] = 0, 0, 255  # Red square
     cv.imshow('Red Square', blank)
     cv.waitKey(0)
     ```
     - Paints a 100x100 pixel region red.

## Drawing Shapes
1. **Rectangle**: `cv2.rectangle(image, pt1, pt2, color, thickness)`
   - **Purpose**: Draws a rectangle on the image.
   - **Parameters**:
     - `image`: The target image.
     - `pt1`: Top-left corner (x, y).
     - `pt2`: Bottom-right corner (x, y).
     - `color`: BGR tuple (e.g., `(0, 255, 0)` for green).
     - `thickness`: Border thickness (pixels). Use `cv.FILLED` or `-1` to fill the rectangle.
   - **Example**:
     ```python
     cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)
     cv.imshow('Rectangle', blank)
     cv.waitKey(0)
     ```
     - Draws a green rectangle from (0, 0) to (250, 250).
   - **Filled Rectangle**:
     ```python
     cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=cv.FILLED)
     cv.imshow('Filled Rectangle', blank)
     cv.waitKey(0)
     ```
     - Fills a rectangle covering half the image.

2. **Circle**: `cv2.circle(image, center, radius, color, thickness)`
   - **Purpose**: Draws a circle on the image.
   - **Parameters**:
     - `image`: The target image.
     - `center`: Center point (x, y).
     - `radius`: Circle radius (pixels).
     - `color`: BGR tuple.
     - `thickness`: Border thickness. Use `-1` for filled.
   - **Example**:
     ```python
     cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=3)
     cv.imshow('Circle', blank)
     cv.waitKey(0)
     ```
     - Draws a red circle with center at (250, 250) and radius 40.

3. **Line**: `cv2.line(image, pt1, pt2, color, thickness)`
   - **Purpose**: Draws a line between two points.
   - **Parameters**:
     - `image`: The target image.
     - `pt1`: Starting point (x, y).
     - `pt2`: Ending point (x, y).
     - `color`: BGR tuple.
     - `thickness`: Line thickness.
   - **Example**:
     ```python
     cv.line(blank, (100, 100), (300, 400), (255, 255, 255), thickness=3)
     cv.imshow('Line', blank)
     cv.waitKey(0)
     ```
     - Draws a white line from (100, 100) to (300, 400).

## Adding Text
1. **Function**: `cv2.putText(image, text, org, fontFace, fontScale, color, thickness)`
   - **Purpose**: Writes text on the image.
   - **Parameters**:
     - `image`: The target image.
     - `text`: String to display.
     - `org`: Bottom-left corner of the text (x, y).
     - `fontFace`: Font type (e.g., `cv.FONT_HERSHEY_TRIPLEX`, `cv.FONT_HERSHEY_COMPLEX`).
     - `fontScale`: Font size scaling factor (e.g., `1.0`).
     - `color`: BGR tuple.
     - `thickness`: Text thickness.
   - **Example**:
     ```python
     cv.putText(blank, 'Hello, my name is Jason', (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
     cv.imshow('Text', blank)
     cv.waitKey(0)
     ```
     - Writes green text starting at (0, 225).
   - **Note**: Adjust `org` to prevent text from going off-screen for large images.

## Example Code
```python
import cv2 as cv
import numpy as np

# Create blank image
blank = np.zeros((500, 500, 3), dtype='uint8')

# Paint green
blank[:] = 0, 255, 0
cv.imshow('Green', blank)

# Draw rectangle
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=cv.FILLED)
cv.imshow('Filled Rectangle', blank)

# Draw circle
cv.circle(blank, (250, 250), 40, (0, 0, 255), thickness=-1)
cv.imshow('Filled Circle', blank)

# Draw line
cv.line(blank, (100, 100), (300, 400), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)

# Add text
cv.putText(blank, 'Hello', (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Use blank images for practice or draw on existing images.
- BGR color format: (Blue, Green, Red).
- Thickness of `-1` or `cv.FILLED` fills shapes.
- Adjust coordinates and font scale to fit text within image boundaries.

## Next Steps
The next section introduces five essential OpenCV functions for image processing.


---

#### Part 6: 5 Essential Functions in OpenCV

# 5 Essential Functions in OpenCV

## Overview
This tutorial covers five fundamental OpenCV functions commonly used in computer vision projects: converting to grayscale, blurring, edge detection, dilation, erosion, resizing, and cropping.

## Setup
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/boston.jpg')
cv.imshow('Original', img)
cv.waitKey(0)
```

## 1. Converting to Grayscale
1. **Function**: `cv2.cvtColor(image, code)`
   - **Purpose**: Converts an image between color spaces (e.g., BGR to grayscale).
   - **Parameters**:
     - `image`: Input image.
     - `code`: Conversion code (e.g., `cv.COLOR_BGR2GRAY` for BGR to grayscale).
   - **Example**:
     ```python
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('Gray', gray)
     cv.waitKey(0)
     ```
   - **Explanation**: Converts a BGR image to grayscale, reducing it to a single channel that represents pixel intensity, not color.

## 2. Blurring an Image
1. **Function**: `cv2.GaussianBlur(image, ksize, sigmaX)`
   - **Purpose**: Applies a Gaussian blur to reduce noise (e.g., from poor lighting or sensor issues).
   - **Parameters**:
     - `image`: Input image.
     - `ksize`: Kernel size as a tuple (width, height). Must be odd (e.g., `(3, 3)`, `(7, 7)`).
     - `sigmaX`: Standard deviation in X direction (use `0` for default).
   - **Example**:
     ```python
     blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
     cv.imshow('Blur', blur)
     cv.waitKey(0)
     ```
     - **Explanation**: Blurs the image using a Gaussian filter. Larger kernels (e.g., `(7, 7)`) increase blur intensity.
   - **Note**: Other blurring techniques exist but are covered in the advanced section.

## 3. Edge Detection
1. **Function**: `cv2.Canny(image, threshold1, threshold2)`
   - **Purpose**: Detects edges in an image using the Canny edge detector.
   - **Parameters**:
     - `image`: Input image (grayscale recommended).
     - `threshold1`, `threshold2`: Hysteresis thresholds for edge detection.
   - **Example**:
     ```python
     can NYT = cv.Canny(img, 125, 175)
     cv.imshow('Canny Edges', canny)
     cv.waitKey(0)
     ```
     - **Explanation**: Identifies edges by detecting intensity gradients. Lower thresholds detect more edges; blurring the image first reduces noise and edge count.
     - **Tip**: Apply blur before Canny to reduce unwanted edges:
       ```python
       blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
       canny = cv.Canny(blur, 125, 175)
       ```

## 4. Dilation and Erosion
1. **Dilation**: `cv2.dilate(image, kernel, iterations)`
   - **Purpose**: Thickens edges in an image (e.g., Canny edges).
   - **Parameters**:
     - `image`: Input image (e.g., Canny edge map).
     - `kernel`: Structuring element (e.g., `np.ones((3, 3))`).
     - `iterations`: Number of dilation applications.
   - **Example**:
     ```python
     dilated = cv.dilate(canny, (3, 3), iterations=1)
     cv.imshow('Dilated', dilated)
     cv.waitKey(0)
     ```
     - **Explanation**: Expands edge pixels, making them thicker. More iterations or larger kernels increase thickness.

2. **Erosion**: `cv2.erode(image, kernel, iterations)`
   - **Purpose**: Shrinks dilated edges, potentially recovering the original structure.
   - **Parameters**: Same as `dilate`.
   - **Example**:
     ```python
     eroded = cv.erode(dilated, (3, 3), iterations=1)
     cv.imshow('Eroded', eroded)
     cv.waitKey(0)
     ```
     - **Explanation**: Reduces edge thickness. May not perfectly restore the original edges but can approximate them with matching parameters.

## 5. Resizing and Cropping
1. **Resizing**: `cv2.resize(image, dsize, interpolation)`
   - **Purpose**: Changes image dimensions.
   - **Parameters**:
     - `image`: Input image.
     - `dsize`: Desired size as `(width, height)`.
     - `interpolation`: Method (e.g., `cv.INTER_AREA` for shrinking, `cv.INTER_LINEAR` or `cv.INTER_CUBIC` for enlarging).
   - **Example**:
     ```python
     resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
     cv.imshow('Resized', resized)
     cv.waitKey(0)
     ```
     - **Explanation**:
       - `cv.INTER_AREA`: Best for shrinking.
       - `cv.INTER_CUBIC`: Slower but higher quality for enlarging.
       - Ignores aspect ratio unless calculated manually.

2. **Cropping**: Array slicing
   - **Purpose**: Extracts a region of interest from the image.
   - **Example**:
     ```python
     cropped = img[50:200, 200:400]
     cv.imshow('Cropped', cropped)
     cv.waitKey(0)
     ```
     - **Explanation**: Selects pixels from rows 50 to 200 and columns 200 to 400, creating a new image.

## Example Code
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/boston.jpg')

# Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge detection
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Dilation
dilated = cv.dilate(canny, (3, 3), iterations=3)
cv.imshow('Dilated', dilated)

# Erosion
eroded = cv.erode(dilated, (3, 3), iterations=3)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

# Crop
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Grayscale conversion simplifies images for processing.
- Gaussian blur reduces noise, aiding edge detection.
- Canny edge detection is sensitive to noise; blur first for cleaner results.
- Dilation and erosion modify edge thickness, useful for structural analysis.
- Resizing and cropping adjust image size and focus on specific regions.

## Next Steps
The next section explores image transformations like translation and rotation.


---

#### Part 7: Image Transformations

# Image Transformations in OpenCV

## Overview
This tutorial covers basic image transformations in OpenCV: translation, rotation, resizing, flipping, and cropping. These techniques manipulate image position, orientation, and size.

## Setup
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')
cv.imshow('Original', img)
cv.waitKey(0)
```

## 1. Translation
1. **Function**: Custom `translate(image, x, y)`
   - **Purpose**: Shifts an image along the x and y axes (up, down, left, right).
   - **Parameters**:
     - `image`: Input image.
     - `x`: Pixels to shift horizontally (positive: right, negative: left).
     - `y`: Pixels to shift vertically (positive: down, negative: up).
   - **Implementation**:
     ```python
     def translate(image, x, y):
         transMat = np.float32([[1, 0, x], [0, 1, y]])
         dimensions = (image.shape[1], image.shape[0])
         return cv.warpAffine(image, transMat, dimensions)
     ```
     - **Explanation**:
       - `transMat`: 2x3 translation matrix specifying x and y shifts.
       - `cv.warpAffine(image, matrix, dsize)`: Applies an affine transformation (e.g., translation) to the image.
       - `dimensions`: Output size (width, height).

2. **Example**:
   ```python
   translated = translate(img, 100, 100)
   cv.imshow('Translated', translated)
   cv.waitKey(0)
   ```
   - Shifts image 100 pixels right and down.

## 2. Rotation
1. **Function**: Custom `rotate(image, angle, rotPoint=None)`
   - **Purpose**: Rotates an image by a specified angle around a point.
   - **Parameters**:
     - `image`: Input image.
     - `angle`: Rotation angle in degrees (positive: counterclockwise, negative: clockwise).
     - `rotPoint`: Rotation center (defaults to image center if `None`).
   - **Implementation**:
     ```python
     def rotate(image, angle, rotPoint=None):
         (height, width) = image.shape[:2]
         if rotPoint is None:
             rotPoint = (width // 2, height // 2)
         rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
         dimensions = (width, height)
         return cv.warpAffine(image, rotMat, dimensions)
     ```
     - **Explanation**:
       - `cv.getRotationMatrix2D(center, angle, scale)`: Creates a 2x3 rotation matrix.
       - `scale=1.0`: No scaling during rotation.
       - Black triangles appear in corners due to image boundaries.

2. **Example**:
   ```python
   rotated = rotate(img, 45)
   cv.imshow('Rotated', rotated)
   cv.waitKey(0)
   ```
   - Rotates image 45° counterclockwise around the center.
   - **Chaining Rotations**:
     ```python
     rotated_rotated = rotate(rotated, -45)
     ```
     - Equivalent to rotating the original by 0° (45° - 45°).

## 3. Resizing
1. **Function**: `cv2.resize(image, dsize, interpolation)`
   - **Purpose**: Changes image dimensions.
   - **Parameters**:
     - `image`: Input image.
     - `dsize`: Desired size (width, height).
     - `interpolation`: Method (e.g., `cv.INTER_AREA`, `cv.INTER_LINEAR`, `cv.INTER_CUBIC`).
   - **Example**:
     ```python
     resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
     cv.imshow('Resized', resized)
     cv.waitKey(0)
     ```
     - **Explanation**:
       - `cv.INTER_AREA`: Best for shrinking.
       - `cv.INTER_CUBIC`: High-quality but slower for enlarging.
       - Ignores aspect ratio unless calculated.

## 4. Flipping
1. **Function**: `cv2.flip(image, flipCode)`
   - **Purpose**: Flips an image vertically, horizontally, or both.
   - **Parameters**:
     - `image`: Input image.
     - `flipCode`:
       - `0`: Vertical flip (over x-axis).
       - `1`: Horizontal flip (over y-axis).
       - `-1`: Both vertical and horizontal flip.
   - **Example**:
     ```python
     flip = cv.flip(img, 1)
     cv.imshow('Horizontal Flip', flip)
     cv.waitKey(0)
     ```
     - Flips image horizontally.

## 5. Cropping
1. **Method**: Array slicing
   - **Purpose**: Extracts a region of interest.
   - **Example**:
     ```python
     cropped = img[200:400, 100:400]
     cv.imshow('Cropped', cropped)
     cv.waitKey(0)
     ```
     - Crops rows 200 to 400 and columns 100 to 400.

## Example Code
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')

# Translation
def translate(image, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (image.shape[1], image.shape[0])
    return cv.warpAffine(image, transMat, dimensions)

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)

# Rotation
def rotate(image, angle, rotPoint=None):
    (height, width) = image.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(image, rotMat, dimensions)

rotated = rotate(img, -90)
cv.imshow('Rotated', rotated)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flip
flip = cv.flip(img, -1)
cv.imshow('Flip', flip)

# Crop
cropped = img[200:400, 100:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Translation and rotation use `cv.warpAffine` with transformation matrices.
- Resizing and flipping are straightforward but require appropriate interpolation.
- Cropping uses array slicing for precise region selection.
- Black regions in rotation occur due to image boundaries.

## Next Steps
The next section covers contour detection for object boundary analysis.


---

#### Part 8: Contour Detection

# Contour Detection in OpenCV

## Overview
This tutorial explains how to detect contours (object boundaries) in images using OpenCV, a key technique for shape analysis and object detection.

## What are Contours?
- **Definition**: Contours are curves joining continuous points along an object’s boundary, often confused with edges but mathematically distinct.
- **Use Cases**: Shape analysis, object detection, and recognition.

## Setup
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')
cv.imshow('Original', img)
cv.waitKey(0)
```

## Preprocessing
1. **Grayscale Conversion**:
   ```python
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   cv.imshow('Gray', gray)
   ```
   - Simplifies the image for edge detection.

2. **Edge Detection** (Canny):
   ```python
   canny = cv.Canny(gray, 125, 175)
   cv.imshow('Canny Edges', canny)
   ```
   - Detects edges as the basis for contours.

3. **Alternative: Thresholding**:
   ```python
   ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
   cv.imshow('Threshold', thresh)
   ```
   - **Explanation**: Binarizes the image (pixels below 125 become 0, above become 255), creating a binary map for contour detection.

## Finding Contours
1. **Function**: `cv2.findContours(image, mode, method)`
   - **Purpose**: Identifies contours in a binary image (e.g., Canny or thresholded).
   - **Parameters**:
     - `image`: Binary image (e.g., Canny edges or thresholded).
     - `mode`:
       - `cv.RETR_LIST`: Returns all contours.
       - `cv.RETR_EXTERNAL`: Returns only external contours.
       - `cv.RETR_TREE`: Returns all hierarchical contours.
     - `method`:
       - `cv.CHAIN_APPROX_NONE`: Returns all contour points.
       - `cv.CHAIN_APPROX_SIMPLE`: Compresses contours to endpoints (e.g., a line’s two endpoints).
   - **Returns**:
     - `contours`: List of contour coordinates.
     - `hierarchy`: Hierarchical representation (optional).
   - **Example**:
     ```python
     contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
     print(f'{len(contours)} contours found')
     ```
     - Counts the number of contours detected.

2. **Reducing Contours**:
   - Apply blur before edge detection to reduce noise and contour count:
     ```python
     blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
     canny = cv.Canny(blur, 125, 175)
     contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
     print(f'{len(contours)} contours found')
     ```
     - Blurring significantly reduces the number of contours (e.g., from 2794 to 380).

3. **Using Thresholding**:
   ```python
   ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
   contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
   print(f'{len(contours)} contours found')
   ```
   - Thresholding may yield different contour counts (e.g., 839).

## Visualizing Contours
1. **Function**: `cv2.drawContours(image, contours, contourIdx, color, thickness)`
   - **Purpose**: Draws contours on an image.
   - **Parameters**:
     - `image`: Target image (use a blank or copy).
     - `contours`: List of contours.
     - `contourIdx`: Index of contour to draw (`-1` for all).
     - `color`: BGR tuple.
     - `thickness`: Contour thickness.
   - **Example**:
     ```python
     blank = np.zeros(img.shape, dtype='uint8')
     cv.drawContours(blank, contours, -1, (0, 255, 0), 1)
     cv.imshow('Contours', blank)
     cv.waitKey(0)
     ```
     - Draws all contours in green on a blank image.

## Example Code
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')
blank = np.zeros(img.shape, dtype='uint8')

# Preprocess
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)

# Find contours
contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found')

# Draw contours
cv.drawContours(blank, contours, -1, (0, 255, 0), 1)
cv.imshow('Contours', blank)

# Threshold alternative
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found via threshold')

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Contours are boundaries, not edges, used for object analysis.
- Preprocessing (grayscale, blur, Canny, or thresholding) is critical for accurate contour detection.
- `cv.RETR_LIST` and `cv.CHAIN_APPROX_SIMPLE` are common settings for general use.
- Blurring reduces noise and contour count.

## Next Steps
The next section begins advanced topics, starting with color spaces.


---

#### Part 9: Color Spaces

# Color Spaces in OpenCV

## Overview
This tutorial introduces color spaces in OpenCV, focusing on converting images between different color representations (e.g., BGR, grayscale, HSV, LAB, RGB).

## What are Color Spaces?
- **Definition**: A color space is a system for representing colors numerically. Each space organizes colors differently, suited for specific tasks.
- **Common Color Spaces**:
  - **BGR**: Default in OpenCV (Blue, Green, Red).
  - **Grayscale**: Single-channel intensity (no color).
  - **HSV**: Hue, Saturation, Value (intuitive for color selection).
  - **LAB**: Perceptually uniform, good for color differences.
  - **RGB**: Standard for displays (Red, Green, Blue).

## Converting Color Spaces
1. **Function**: `cv2.cvtColor(image, code)`
   - **Purpose**: Converts an image from one color space to another.
   - **Parameters**:
     - `image`: Input image.
     - `code`: Conversion code (e.g., `cv.COLOR_BGR2GRAY`, `cv.COLOR_BGR2HSV`).
   - **Examples**:
     ```python
     import cv2 as cv

     img = cv.imread('photos/cat.jpg')

     # BGR to Grayscale
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('Grayscale', gray)

     # BGR to HSV
     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
     cv.imshow('HSV', hsv)

     # BGR to LAB
     lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
     cv.imshow('LAB', lab)

     # BGR to RGB
     rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     cv.imshow('RGB', rgb)

     cv.waitKey(0)
     cv.destroyAllWindows()
     ```
     - **Explanation**:
       - **Grayscale**: Reduces to one channel, showing intensity.
       - **HSV**: Represents colors by hue (color type), saturation (intensity), and value (brightness). Useful for color-based segmentation.
       - **LAB**: Separates lightness (L) from color (A, B channels), ideal for perceptual uniformity.
       - **RGB**: Matches standard display formats but requires conversion from OpenCV’s BGR.

## Key Points
- OpenCV uses BGR by default, unlike most systems (RGB).
- HSV is intuitive for tasks like color filtering.
- Use `cv2.cvtColor` with the appropriate code for conversions.
- Display issues may arise with non-BGR images (e.g., HSV may look distorted).

## Next Steps
The next section explores splitting and merging color channels.


---

#### Part 10: Color Channels

# Color Channels in OpenCV

## Overview
This tutorial explains how to split and merge color channels in OpenCV, allowing manipulation of individual color components (e.g., Blue, Green, Red).

## What are Color Channels?
- **Definition**: A color image (e.g., BGR) consists of multiple channels, each representing a color component. For BGR, there are three channels: Blue, Green, Red.
- **Use Case**: Isolating or modifying specific colors.

## Splitting Channels
1. **Function**: `cv2.split(image)`
   - **Purpose**: Separates a multi-channel image into individual channels.
   - **Parameters**:
     - `image`: Input image (e.g., BGR).
   - **Returns**: Tuple of single-channel images (e.g., `(b, g, r)` for BGR).
   - **Example**:
     ```python
     import cv2 as cv
     import numpy as np

     img = cv.imread('photos/cat.jpg')
     b, g, r = cv.split(img)
     cv.imshow('Blue', b)
     cv.imshow('Green', g)
     cv.imshow('Red', r)
     cv.waitKey(0)
     ```
     - **Explanation**: Each channel is a grayscale image where intensity represents the color’s contribution. Bright areas indicate high presence of that color.

## Merging Channels
1. **Function**: `cv2.merge(channels)`
   - **Purpose**: Combines single-channel images into a multi-channel image.
   - **Parameters**:
     - `channels`: Tuple or list of channels (e.g., `(b, g, r)`).
   - **Example**:
     ```python
     merged = cv.merge((b, g, r))
     cv.imshow('Merged', merged)
     cv.waitKey(0)
     ```
     - **Explanation**: Reconstructs the original BGR image.

## Visualizing Individual Channels
1. **Blank Image for Display**:
   - To show a single channel’s contribution in color:
     ```python
     blank = np.zeros(img.shape[:2], dtype='uint8')
     blue = cv.merge((b, blank, blank))
     green = cv.merge((blank, g, blank))
     red = cv.merge((blank, blank, r))
     cv.imshow('Blue Channel', blue)
     cv.imshow('Green Channel', green)
     cv.imshow('Red Channel', red)
     cv.waitKey(0)
     ```
     - **Explanation**: Creates a BGR image with only one channel active, showing its color contribution.

## Example Code
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')

# Split channels
b, g, r = cv.split(img)
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

# Merge channels
merged = cv.merge((b, g, r))
cv.imshow('Merged', merged)

# Visualize channels in color
blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv.merge((b, blank, blank))
green = cv.merge((blank, g, blank))
red = cv.merge((blank, blank, r))
cv.imshow('Blue Channel', blue)
cv.imshow('Green Channel', green)
cv.imshow('Red Channel', red)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Splitting isolates color contributions; merging reconstructs the image.
- Single channels appear as grayscale; use `merge` with blanks to visualize in color.
- Useful for tasks like color-based filtering or enhancement.

## Next Steps
The next section covers advanced blurring techniques.


---

#### Part 11: Blurring

# Blurring in OpenCV

## Overview
This tutorial explores advanced blurring techniques in OpenCV to reduce noise and smooth images, including Gaussian, average, median, and bilateral blurring.

## Why Blur?
- **Noise Reduction**: Removes artifacts from poor lighting or sensor issues.
- **Preprocessing**: Prepares images for edge detection or other analyses.

## Setup
```python
import cv2 as cv

img = cv.imread('photos/cat.jpg')
cv.imshow('Original', img)
cv.waitKey(0)
```

## Blurring Techniques
1. **Average Blur**: `cv2.blur(image, ksize)`
   - **Purpose**: Averages pixel values within a kernel window.
   - **Parameters**:
     - `image`: Input image.
     - `ksize`: Kernel size (width, height), e.g., `(5, 5)`.
   - **Example**:
     ```python
     avg = cv.blur(img, (5, 5))
     cv.imshow('Average Blur', avg)
     cv.waitKey(0)
     ```
     - **Explanation**: Simple but may lose edge details.

2. **Gaussian Blur**: `cv2.GaussianBlur(image, ksize, sigmaX)`
   - **Purpose**: Uses a Gaussian kernel for weighted averaging, preserving edges better.
   - **Parameters**:
     - `image`: Input image.
     - `ksize`: Odd-sized kernel (e.g., `(5, 5)`).
     - `sigmaX`: Gaussian standard deviation (0 for auto).
   - **Example**:
     ```python
     gauss = cv.GaussianBlur(img, (5, 5), 0)
     cv.imshow('Gaussian Blur', gauss)
     cv.waitKey(0)
     ```

3. **Median Blur**: `cv2.medianBlur(image, ksize)`
   - **Purpose**: Replaces each pixel with the median of its neighbors, effective for salt-and-pepper noise.
   - **Parameters**:
     - `image`: Input image.
     - `ksize`: Odd integer (e.g., `5`).
   - **Example**:
     ```python
     median = cv.medianBlur(img, 5)
     cv.imshow('Median Blur', median)
     cv.waitKey(0)
     ```

4. **Bilateral Blur**: `cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)`
   - **Purpose**: Preserves edges while smoothing, ideal for noise reduction without blurring boundaries.
   - **Parameters**:
     - `image`: Input image.
     - `d`: Diameter of pixel neighborhood.
     - `sigmaColor`: Filter sigma in color space.
     - `sigmaSpace`: Filter sigma in coordinate space.
   - **Example**:
     ```python
     bilateral = cv.bilateralFilter(img, 10, 35, 25)
     cv.imshow('Bilateral Blur', bilateral)
     cv.waitKey(0)
     ```

## Example Code
```python
import cv2 as cv

img = cv.imread('photos/cat.jpg')

# Average blur
avg = cv.blur(img, (5, 5))
cv.imshow('Average Blur', avg)

# Gaussian blur
gauss = cv.GaussianBlur(img, (5, 5), 0)
cv.imshow('Gaussian Blur', gauss)

# Median blur
median = cv.medianBlur(img, 5)
cv.imshow('Median Blur', median)

# Bilateral blur
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- **Average**: Simple, fast, but blurs edges.
- **Gaussian**: Better edge preservation, widely used.
- **Median**: Best for salt-and-pepper noise.
- **Bilateral**: Edge-preserving, computationally intensive.
- Larger kernels increase blur intensity.

## Next Steps
The next section covers bitwise operations for image manipulation.


---

#### Part 12: BITWISE Operations

# BITWISE Operations in OpenCV

## Overview
This tutorial explains bitwise operations (AND, OR, XOR, NOT) in OpenCV, used for combining or manipulating binary images (e.g., masks).

## What are Bitwise Operations?
- **Definition**: Operations performed on individual bits of pixel values, typically on binary (black-and-white) images.
- **Use Case**: Combining shapes, creating masks, or isolating regions.

## Setup
```python
import cv2 as cv
import numpy as np

# Create blank image and shapes
blank = np.zeros((400, 400), dtype='uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)
```

## Bitwise Operations
1. **AND**: `cv2.bitwise_and(src1, src2)`
   - **Purpose**: Returns pixels where both images have non-zero values.
   - **Example**:
     ```python
     bit_and = cv.bitwise_and(rectangle, circle)
     cv.imshow('Bitwise AND', bit_and)
     ```
     - **Explanation**: Shows the overlapping region of the rectangle and circle.

2. **OR**: `cv2.bitwise_or(src1, src2)`
   - **Purpose**: Returns pixels where either image has non-zero values.
   - **Example**:
     ```python
     bit_or = cv.bitwise_or(rectangle, circle)
     cv.imshow('Bitwise OR', bit_or)
     ```
     - **Explanation**: Shows the union of both shapes.

3. **XOR**: `cv2.bitwise_xor(src1, src2)`
   - **Purpose**: Returns pixels where exactly one image has non-zero values.
   - **Example**:
     ```python
     bit_xor = cv.bitwise_xor(rectangle, circle)
     cv.imshow('Bitwise XOR', bit_xor)
     ```
     - **Explanation**: Shows non-overlapping regions.

4. **NOT**: `cv2.bitwise_not(src)`
   - **Purpose**: Inverts pixel values (0 becomes 255, 255 becomes 0).
   - **Example**:
     ```python
     bit_not = cv.bitwise_not(rectangle)
     cv.imshow('Bitwise NOT', bit_not)
     ```
     - **Explanation**: Inverts the rectangle (white to black, black to white).

## Example Code
```python
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
```

## Key Points
- Bitwise operations work on binary images (0 or 255).
- AND: Intersection, OR: Union, XOR: Exclusive regions, NOT: Inversion.
- Useful for masking and combining shapes.

## Next Steps
The next section explores masking techniques.


---

#### Part 13: Masking

# Masking in OpenCV

## Overview
This tutorial covers masking in OpenCV, allowing you to focus on specific regions of an image by applying a binary mask.

## What is Masking?
- **Definition**: A mask is a binary image (0 or 255) that specifies which parts of an image to process or display. Pixels where the mask is 255 are kept; 0 pixels are ignored.
- **Use Case**: Isolating regions of interest (e.g., a circular area).

## Setup
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')
```

## Creating a Mask
1. **Example**: Circular mask
   ```python
   mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
   cv.imshow('Mask', mask)
   ```
   - Creates a white circle (255) on a black background (0).

## Applying the Mask
1. **Function**: `cv2.bitwise_and(src1, src2, mask)`
   - **Purpose**: Applies the mask to an image, keeping only masked regions.
   - **Parameters**:
     - `src1`: Input image.
     - `src2`: Same as `src1` (or another image).
     - `mask`: Binary mask (non-zero regions are kept).
   - **Example**:
     ```python
     masked = cv.bitwise_and(img, img, mask=mask)
     cv.imshow('Masked Image', masked)
     cv.waitKey(0)
     ```
     - **Explanation**: Only the circular region (mask=255) is visible; other areas are black.

## Example Code
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')

# Create circular mask
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', mask)

# Apply mask
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Masks are binary (0 or 255).
- Use `bitwise_and` to apply masks, isolating specific regions.
- Masks can be any shape (e.g., circles, rectangles) created with drawing functions.

## Next Steps
The next section covers histogram computation for image analysis.


---

#### Part 14: Histogram Computation

# Histogram Computation in OpenCV

## Overview
This tutorial explains how to compute and visualize histograms in OpenCV, which represent the distribution of pixel intensities in an image.

## What is a Histogram?
- **Definition**: A histogram shows the frequency of pixel intensity values (0 to 255) in an image, useful for understanding brightness or color distribution.
- **Use Case**: Image analysis, contrast adjustment.

## Setup
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('photos/cat.jpg')
```

## Grayscale Histogram
1. **Function**: `cv2.calcHist(images, channels, mask, histSize, ranges)`
   - **Purpose**: Computes the histogram of pixel intensities.
   - **Parameters**:
     - `images`: List of images (e.g., `[gray]`).
     - `channels`: List of channels to compute (e.g., `[0]` for grayscale).
     - `mask`: Optional mask (None for whole image).
     - `histSize`: Number of bins (e.g., `[256]` for 0-255).
     - `ranges`: Range of values (e.g., `[0, 256]`).
   - **Example**:
     ```python
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
     plt.plot(gray_hist)
     plt.title('Grayscale Histogram')
     plt.xlabel('Bins')
     plt.ylabel('Number of Pixels')
     plt.show()
     ```

## Color Histogram
1. **Example**:
   ```python
   colors = ('b', 'g', 'r')
   for i, col in enumerate(colors):
       hist = cv.calcHist([img], [i], None, [256], [0, 256])
       plt.plot(hist, color=col)
       plt.xlim([0, 256])
   plt.title('Color Histogram')
   plt.xlabel('Bins')
   plt.ylabel('Number of Pixels')
   plt.show()
   ```
   - Computes histograms for Blue, Green, Red channels.

## Masked Histogram
1. **Example**:
   ```python
   blank = np.zeros(img.shape[:2], dtype='uint8')
   mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
   gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
   plt.plot(gray_hist)
   plt.title('Masked Grayscale Histogram')
   plt.xlabel('Bins')
   plt.ylabel('Number of Pixels')
   plt.show()
   ```
   - Computes histogram for the masked region only.

## Example Code
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('photos/cat.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Grayscale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(gray_hist)
plt.title('Grayscale Histogram')
plt.show()

# Color histogram
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title('Color Histogram')
plt.show()

# Masked histogram
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
plt.plot(gray_hist)
plt.title('Masked Grayscale Histogram')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Histograms show pixel intensity distributions.
- Use masks to focus on specific regions.
- Matplotlib is used for visualization due to OpenCV’s limited plotting capabilities.

## Next Steps
The next section covers thresholding for binarizing images.


---

#### Part 15: Thresholding/Binarizing Images

# Thresholding in OpenCV

## Overview
This tutorial covers thresholding techniques in OpenCV to binarize images, creating black-and-white representations for tasks like contour detection.

## What is Thresholding?
- **Definition**: Thresholding converts a grayscale image into a binary image by setting pixel values above or below a threshold to specific values (e.g., 0 or 255).
- **Use Case**: Preprocessing for contour detection, object segmentation.

## Setup
```python
import cv2 as cv

img = cv.imread('photos/cat.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

## Simple Thresholding
1. **Function**: `cv2.threshold(src, thresh, maxval, type)`
   - **Purpose**: Applies a fixed threshold to each pixel.
   - **Parameters**:
     - `src`: Input grayscale image.
     - `thresh`: Threshold value (0-255).
     - `maxval`: Value for pixels above/below threshold (e.g., 255).
     - `type`:
       - `cv.THRESH_BINARY`: Above threshold → maxval, below → 0.
       - `cv.THRESH_BINARY_INV`: Inverse of BINARY.
       - `cv.THRESH_TRUNC`: Above threshold → thresh, below unchanged.
       - `cv.THRESH_TOZERO`: Below threshold → 0, above unchanged.
       - `cv.THRESH_TOZERO_INV`: Inverse of TOZERO.
   - **Example**:
     ```python
     ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
     cv.imshow('Binary Threshold', thresh)
     cv.waitKey(0)
     ```
     - **Explanation**: Pixels ≥ 125 become 255 (white), < 125 become 0 (black).

## Adaptive Thresholding
1. **Function**: `cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)`
   - **Purpose**: Computes thresholds dynamically for small regions, ideal for varying lighting.
   - **Parameters**:
     - `src`: Grayscale image.
     - `maxValue`: Value for thresholded pixels.
     - `adaptiveMethod`:
       - `cv.ADAPTIVE_THRESH_MEAN_C`: Mean of neighborhood.
       - `cv.ADAPTIVE_THRESH_GAUSSIAN_C`: Gaussian-weighted mean.
     - `thresholdType`: `cv.THRESH_BINARY` or `cv.THRESH_BINARY_INV`.
     - `blockSize`: Neighborhood size (odd, e.g., 11).
     - `C`: Constant subtracted from mean.
   - **Example**:
     ```python
     adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
     cv.imshow('Adaptive Threshold', adaptive)
     cv.waitKey(0)
     ```

## Example Code
```python
import cv2 as cv

img = cv.imread('photos/cat.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Simple thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Binary Threshold', thresh)

# Adaptive thresholding
adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
cv.imshow('Adaptive Threshold', adaptive)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Simple thresholding uses a global threshold, less effective for uneven lighting.
- Adaptive thresholding adjusts thresholds locally, better for complex images.
- Binary images are ideal for contour detection and segmentation.

## Next Steps
The next section covers edge detection techniques.


---

#### Part 16: Edge Detection

# Edge Detection in OpenCV

## Overview
This tutorial explores edge detection techniques in OpenCV, including Canny, Sobel, and Laplacian, used to identify boundaries in images.

## Setup
```python
import cv2 as cv

img = cv.imread('photos/cat.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

## 1. Canny Edge Detection
1. **Function**: `cv2.Canny(image, threshold1, threshold2)`
   - **Purpose**: Detects edges using a multi-stage process (blurring, gradient computation, edge tracing).
   - **Parameters**:
     - `image`: Grayscale image.
     - `threshold1`, `threshold2`: Hysteresis thresholds.
   - **Example**:
     ```python
     canny = cv.Canny(gray, 125, 175)
     cv.imshow('Canny', canny)
     cv.waitKey(0)
     ```
     - **Explanation**: Cleaner edges, widely used due to robustness.

## 2. Sobel Edge Detection
1. **Function**: `cv2.Sobel(src, ddepth, dx, dy, ksize)`
   - **Purpose**: Computes gradients in x or y directions.
   - **Parameters**:
     - `src`: Grayscale image.
     - `ddepth`: Output depth (e.g., `cv.CV_64F` for floating-point).
     - `dx`, `dy`: Order of derivative (1 for x or y, 0 for other).
     - `ksize`: Kernel size (odd, e.g., 3).
   - **Example**:
     ```python
     sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
     sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
     sobel_combined = cv.bitwise_or(sobelx, sobely)
     cv.imshow('Sobel X', sobelx)
     cv.imshow('Sobel Y', sobely)
     cv.imshow('Sobel Combined', sobel_combined)
     ```

## 3. Laplacian Edge Detection
1. **Function**: `cv2.Laplacian(src, ddepth)`
   - **Purpose**: Computes the Laplacian of the image, highlighting edges.
   - **Parameters**:
     - `src`: Grayscale image.
     - `ddepth`: Output depth (e.g., `cv.CV_64F`).
   - **Example**:
     ```python
     lap = cv.Laplacian(gray, cv.CV_64F)
     lap = np.uint8(np.absolute(lap))
     cv.imshow('Laplacian', lap)
     cv.waitKey(0)
     ```

## Example Code
```python
import cv2 as cv
import numpy as np

img = cv.imread('photos/cat.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Canny
canny = cv.Canny(gray, 125, 175)
cv.imshow('Canny', canny)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
sobel_combined = cv.bitwise_or(sobelx, sobely)
cv.imshow('Sobel Combined', sobel_combined)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- **Canny**: Most robust, uses Sobel internally.
- **Sobel**: Computes directional gradients, useful for advanced applications.
- **Laplacian**: Highlights all edges, less common.
- Preprocessing (e.g., blurring) improves results.

## Next Steps
The next section covers face detection with Haar cascades.


---

#### Part 17: Face Detection with Haar Cascades

# Face Detection with Haar Cascades in OpenCV

## Overview
This tutorial explains how to detect faces in images using OpenCV’s Haar cascade classifiers, a popular method for object detection.

## What is Face Detection?
- **Definition**: Identifies the presence and location of faces in an image, distinct from face recognition (identifying who the face belongs to).
- **Method**: Uses pre-trained classifiers to detect face-like patterns.

## Setup
y
- **Haar Cascades**: Pre-trained XML files provided by OpenCV for detecting faces, eyes, smiles, etc.
- **Source**: Available on OpenCV’s GitHub (e.g., `haarcascade_frontalface_default.xml`).

## Getting the Haar Cascade
1. **Download**:
   - Visit OpenCV’s GitHub: [haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades).
   - Copy the raw XML code for `haarcascade_frontalface_default.xml`.
   - Save as `haar_face.xml` in your project directory.

## Face Detection
1. **Function**: `cv2.CascadeClassifier(path)`
   - **Purpose**: Loads the Haar cascade file.
   - **Example**:
     ```python
     haar_cascade = cv.CascadeClassifier('haar_face.xml')
     ```

2. **Function**: `cascade.detectMultiScale(image, scaleFactor, minNeighbors)`
   - **Purpose**: Detects faces and returns rectangular coordinates.
   - **Parameters**:
     - `image`: Grayscale image.
     - `scaleFactor`: Scales the image down each iteration (e.g., 1.1).
     - `minNeighbors`: Minimum neighboring rectangles to confirm a face (e.g., 3).
   - **Example**:
     ```python
     import cv2 as cv

     img = cv.imread('photos/person.jpg')
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     haar_cascade = cv.CascadeClassifier('haar_face.xml')
     faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
     print(f'Number of faces found: {len(faces_rect)}')
     ```

3. **Drawing Rectangles**:
   ```python
   for (x, y, w, h) in faces_rect:
       cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
   cv.imshow('Detected Faces', img)
   cv.waitKey(0)
   ```
   - Draws green rectangles around detected faces.

## Example Code
```python
import cv2 as cv

img = cv.imread('photos/group_1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print(f'Number of faces found: {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow('Detected Faces', img)
cv.waitKey(0)
cv.destroyAllWindows()
```

## Key Points
- Haar cascades are sensitive to noise (e.g., may detect non-faces like necks).
- Adjust `scaleFactor` and `minNeighbors` to balance sensitivity