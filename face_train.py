import os
import cv2 as cv
import numpy as np
import concurrent.futures

# path to the dataset directory
DIR = r'C:\Users\owner\dataset'

# list to store the names of people in the dataset
people = [person for person in os.listdir(DIR)]

# load the Haar cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# lists to store the extracted features and labels
features = []
labels = []


# function to process an image
def process_image(img_path, label):

    # read the image from the specified path
    img_array = cv.imread(img_path)

    # convert the image to grayscale
    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale image using the Haar cascade
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    local_features = []
    local_labels = []

    # Iterate over each detected face
    for (x, y, w, h) in faces_rect:
        # Extract the region of interest for each detected face
        faces_roi = gray[y:y + h, x:x + w]
        local_features.append(faces_roi)
        local_labels.append(label)

    # Return the features and labels for the processed image
    return local_features, local_labels


# function to create the training data
def create_train():

    # create a dictionary mapping each person's name to a unique label
    label_dict = {person: index for index, person in enumerate(people)}

    # use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        # iterate over each person in the dataset
        for person in people:
            # construct the path to the person's directory
            path = os.path.join(DIR, person)
            # get the label for the current person
            label = label_dict[person]

            # process each image in the person's directory
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                # submit a job to process the image in parallel
                futures.append(executor.submit(process_image, img_path, label))

        # wait for all jobs to complete and collect the results
        for future in concurrent.futures.as_completed(futures):
            # retrieve the features and labels for the processed image
            local_features, local_labels = future.result()
            # extend the global features and labels lists with the local ones
            features.extend(local_features)
            labels.extend(local_labels)


# create the training dataset
create_train()

# convert the features and labels lists to numpy arrays
features = np.array(features, dtype=object)
labels = np.array(labels)

# create an LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer.create()

# train the face recognizer with the features and labels
face_recognizer.train(features, labels)

# save the trained recognizer to a file
face_recognizer.save('face_trained.yml')
