import cv2 as cv 

img = cv.imread('llama.jpg')

print(f'Shape of the image: {img.shape}')
print(f'Dimension of the image: {img.ndim}')
print(f'{img.shape[0]}')
print(f'{img.shape[1]}')
print(f'{img.shape[2]}')

