# By: Landon Prince (5/18/2024)

import os
import cv2 as cv
import numpy as np
import concurrent.futures

DIR = r'C:\Users\owner\dataset'
people = [person for person in os.listdir(DIR)]

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []


# For each image in dataset, extract faces and labels using haar cascade
def process_image(img_path, label):
    img_array = cv.imread(img_path)

    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    local_features = []
    local_labels = []

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]
        local_features.append(faces_roi)
        local_labels.append(label)

    return local_features, local_labels


# Map image labels to indices and asynchronously process each image
def create_train():
    label_dict = {person: index for index, person in enumerate(people)}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for person in people:
            path = os.path.join(DIR, person)
            label = label_dict[person]

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                futures.append(executor.submit(process_image, img_path, label))

        for future in concurrent.futures.as_completed(futures):
            local_features, local_labels = future.result()
            features.extend(local_features)
            labels.extend(local_labels)


create_train()

features = np.array(features, dtype=object)
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')
