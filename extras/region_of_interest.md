The line `faces_roi = gray[y:y+h, x:x+w]` extracts a rectangular region (a "region of interest" or ROI) from a grayscale image `gray` using array slicing. Here, `gray` is typically a 2D NumPy array (from OpenCV or similar), where each element represents a pixel intensity. The coordinates `(x, y)` specify the top-left corner of the rectangle, and `w` and `h` are its width and height, respectively.

### How It Works
- `gray[y:y+h, x:x+w]` slices the `gray` array:
  - `y:y+h` selects rows from index `y` to `y+h` (height of the rectangle).
  - `x:x+w` selects columns from index `x` to `x+w` (width of the rectangle).
- This creates a sub-array (`faces_roi`) containing the pixel values of the specified rectangular region.
- In the context of the loop, `faces_roi` is a cropped portion of the image (e.g., a face), which is appended to `features`, and `label` is appended to `labels` (likely for machine learning).

### Simple Example
Assume `gray` is a 5x5 grayscale image (as a NumPy array), and we want to extract a 2x2 region starting at `(x, y) = (1, 1)` with `w = 2`, `h = 2`.

```python
import numpy as np

# Example 5x5 grayscale image (2D array)
gray = np.array([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24]
])

# List of one face rectangle (x, y, w, h)
faces_rect = [(1, 1, 2, 2)]
features = []
labels = []

# Loop to extract ROI
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]  # Extract 2x2 region
    features.append(faces_roi)
    labels.append("face")

print("Extracted ROI:\n", features[0])
```

### Explanation of Slicing
- `y:y+h` → `1:1+2` → `1:3` → rows 1 to 2 (indices 1 and 2).
- `x:x+w` → `1:1+2` → `1:3` → columns 1 to 2 (indices 1 and 2).
- `faces_roi` is a 2x2 array from `gray`, containing:
  ```
  [[ 6,  7],
   [11, 12]]
  ```

### Output
```
Extracted ROI:
[[ 6  7]
 [11 12]]
```

### Notes
- `gray` must be a 2D NumPy array (common in image processing with OpenCV).
- Ensure `x+w` and `y+h` don't exceed the image dimensions, or you'll get an error.
- This is often used in face detection to crop face regions for further processing (e.g., feature extraction).