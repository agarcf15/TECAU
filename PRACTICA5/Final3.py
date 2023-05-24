import cv2
import os
import glob
import numpy as np

face_recognizer1 = cv2.face.EigenFaceRecognizer_create()

face_recognizer1.read('modeloEigenFace.xml')



cap = cv2.VideoCapture(1)
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
flag = False

while True:
     ret,frame = cap.read()
     frame = cv2.flip(frame,1)
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     auxFrame = frame.copy()
     faces = faceClassif.detectMultiScale(gray, 1.3, 5)
     k = cv2.waitKey(1)
     if k == ord('p'):
         break
     for (x,y,w,h) in faces:
         
         rostro = auxFrame[y:y+h,x:x+w]
         rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
         rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)

         result = face_recognizer1.predict(rostro)
         print(result)
         if result[1] > 7000:
             cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
         else:
             cv2.rectangle(frame, (x,y),(x+w,y+h),(128,255,0),2)
         cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
         cv2.imshow('frame',frame)
cap.release()
cv2.destroyAllWindows()
