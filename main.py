# By: Landon Prince (5/18/2024)

import os
import cv2 as cv
from utils import is_supported
from face_recognize import recognize_faces

print("Welcome to DeepSight")

while True:
    try:
        image_path = input("Enter the image file path: ")
        if not os.path.exists(image_path):
            raise ValueError("Image file not found.")
        if not is_supported(image_path):
            raise ValueError("Image type not supported.")
        break
    except ValueError as e:
        print(e)
        print("Please try again.")

try:
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise Exception("Empty image")
    print("Image successfully loaded")

    recognize_faces(image)
except Exception as e:
    print("Failed to load image:", e)


