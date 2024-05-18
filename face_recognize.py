import os
import cv2 as cv

# path to the dataset directory
DIR = r'C:\Users\owner\dataset'

# list to store the names of people in the dataset
people = [person for person in os.listdir(DIR)]

# load the Haar cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# create an LBPH Face Recognizer and load the trained model
face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('face_trained.yml')

# load an image for face recognition
img = cv.imread(r'C:\Users\owner\dataset\Aaron_Pena\Aaron_Pena_0001.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img', gray)

# Detect faces in the grayscale image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Iterate over each detected face
for (x, y, w, h) in faces_rect:
    # Extract the Region of Interest (ROI) for each detected face
    faces_roi = gray[y:y + h, x:x + w]

    # Perform face recognition on the ROI
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label: {people[label]}, confidence: {confidence}')

    # Display the label of the recognized person on the image
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_PLAIN,
               1.0, (0, 255, 0), 2)
    # Draw a rectangle around the detected face
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the detected faces
cv.imshow('img', img)
cv.waitKey(0)
