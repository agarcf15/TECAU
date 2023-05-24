import cv2
import os
#Creamos una carpeta para almacenar los rostros si esta no existe
imagenes = input("Con cuantas imagenes quieres entrenar: ")
print("Se haran "+imagenes+" imagenes")
nombre = input("Quién vas a guardar: ")
if not os.path.exists('Rostros_'+nombre):
     print('Carpeta creada: Rostros encontrados')
     os.makedirs('Rostros_'+nombre)
cap = cv2.VideoCapture(1)
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
flag = False
archivo = open('nombres.txt', 'a')
archivo.write(nombre)
archivo.close()
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
         cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
         rostro = auxFrame[y:y+h,x:x+w]
         rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
         if flag==True:
             cv2.imwrite('Rostros_'+nombre+'/rostro_{}.jpg'.format(count),cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY))
             cv2.imshow('rostro',rostro)
             count = count +1
         if k == ord('q'):
             flag = False
         if k == ord('s'):
             flag = True
         if count ==  int(imagenes):
             flag = False
         cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
         cv2.putText(frame,'Presione s, para almacenar los rostros encontrados',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
         cv2.imshow('frame',frame)
cap.release()
cv2.destroyAllWindows()