# By: Landon Prince (5/18/2024)

import os
import cv2 as cv
from utils import get_corresponding_color


def recognize_faces(img):
    DIR = r'C:\Users\owner\dataset'
    people = [person for person in os.listdir(DIR)]

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    face_recognizer = cv.face.LBPHFaceRecognizer.create()
    face_recognizer.read('face_trained.yml')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        color_code, text_color = get_corresponding_color(confidence)
        reset_color = '\033[0m'

        print(f'label: {people[label]}, confidence: {color_code}{confidence:.2f}{reset_color}')

        text = f'{people[label]}: {confidence:.2f}'
        cv.putText(img, text, (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1.0, text_color, 2)

        cv.rectangle(img, (x, y), (x + w, y + h), text_color, 2)

    cv.imshow('Detected Faces', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
