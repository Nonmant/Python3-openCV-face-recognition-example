import cv2

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

#create classifier to detect faces
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('faceRecogParams.xml')

for i in range(3):
    image=cv2.imread('checkingData/'+str(i+1)+'.jpg')

    if showImages:
            cv2.imshow(str(i+1), image) 
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

    faceImage,_=detect_face(face_cascade, image)
    if faceImage is None:
            continue

    if showImages:
            cv2.imshow('Face on '+str(i+1), faceImage)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

    label=face_recognizer.predict(faceImage)
    print(str(label))
