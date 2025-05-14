import numpy as np

# Create a 3D NumPy array (e.g., like an RGB image)
array = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], # this means one 3x3 -> 2d array
                     [[10, 11, 12], [13, 14, 15], [16, 17, 18]],# this means one 3x3 -> 2d array
                     [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])# this means one 3x3 -> 2d array

# final thought is we are getting three 2d array of 3x3 size. which means one (3x3x3) array.

# Shape: (height, width, channels)
print(f'Shape: {array.shape}')

# Dimensions 
print(f'Dimensions: {array.ndim}')

# Slicing examples 

# 1. Get a single 2D slice (first row or height index)
print(f'2D slice: \n{array[0, :, :]}')
print()

# 2. Get a single channel (e.g., "R" channel, index 0 of last dimension)
channel = array[:, :, 0]
print("Single channel: \n", channel)
print()

# 3. Get a 2x2 sub-array from the first two rows and columns, all channels
sub_array = array[0:2, 0:2, :]
print("2x2 sub-array: \n", sub_array) 