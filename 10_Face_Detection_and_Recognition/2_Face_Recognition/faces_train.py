import os 
import cv2 as cv
import numpy as np

people = []
# r makes Python read the string literally, so backslashes in file paths work correctly.
for p in os.listdir(r'D:\WorkSpace\Machine Learning\OpenCV-with-Python\10_Face_Detection_and_Recognition\2_Face_Recognition\train'):
    people.append(p)

DIR = r'D:\WorkSpace\Machine Learning\OpenCV-with-Python\10_Face_Detection_and_Recognition\2_Face_Recognition\train'

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        