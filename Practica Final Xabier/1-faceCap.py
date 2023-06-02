import cv2
import os

#Creamos una carpeta para almacenar los rostros si esta no existe
cantImagenes = 250

if not os.path.exists('Rostros'):
     print('Carpeta creada: Rostros encontrados')
     os.makedirs('Rostros')
video = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
count = 0
while True:
     ret,frame = video.read()
     frame = cv2.flip(frame,1)
     
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     auxFrame = frame.copy()
     faces = faceClassif.detectMultiScale(gray, 1.3, 5)
     k = cv2.waitKey(1)
     if k == ord('x'):
        break
     for (x,y,w,h) in faces:
         cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
         rostro = auxFrame[y:y+h,x:x+w]
         rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
         if count < cantImagenes:
             cv2.imwrite('Rostros/rostro_{}.jpg'.format(count),cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY))
             count = count +1
         cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
         cv2.imshow('frame',frame)
video.release()
cv2.destroyAllWindows()