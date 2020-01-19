import cv2
import os
import json
import numpy as np

def detect_face(face_cascade, img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#load OpenCV face detector, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
    #face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
 
#let's detect multiscale images(some images may be closer to camera than others)
#result is a list of faces
    faces = face_cascade.detectMultiScale(gray, minNeighbors=3);#, scaleFactor=1.2
 
#if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
 
#under the assumption that there will be only one face,
#extract the face area
    (x, y, w, h) = faces[0]
 
#return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

showImages=True

#collecting all file paths into json array
ls=os.walk('trainingData')

files={}

for path in ls:
    if path[1]:
        for dir in path[1]:
            files[str(dir)]=[]
        continue
    files[path[0][(path[0].find('\\')+1):]]=path[2]

#create classifier to detect faces
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

#list to hold all subject faces
faces = []
#list to hold labels for all subjects
labels = []
#list of avaiable values for labels
labelKeys=list(files.keys())

#iterating over each directory
for name in files.items():
    for file in name[1]:
        path='trainingData/'+name[0]+'/'+file
        image=cv2.imread(path)

        if showImages:
            cv2.imshow('path', image) 
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

        faceImage,_=detect_face(face_cascade, image)
        if faceImage is None:
            continue

        faces.append(faceImage)
        labels.append(labelKeys.index(name[0]))

        if showImages:
            cv2.imshow('Face on '+path, faceImage)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
 
#EigenFaceRecognizer EigenFaceRecognizer_create
#FisherFaceRecognizer FisherFaceRecognizer_create

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

face_recognizer.save('faceRecogParams.xml')