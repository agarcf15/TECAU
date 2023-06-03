import cv2
import os

def faceCapture():
    #Creamos una carpeta para almacenar los rostros si esta no existe
    imagenes = input("Con cuantas imágenes quieres entrenar: ")
    print("Se harán "+imagenes+" imágenes")
    nombre = input("Quién vas a guardar: ")
    if not os.path.exists('Rostros_'+nombre):
        print('Carpeta creada:')
        os.makedirs('Rostros_'+nombre)
    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0
    flag = False
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(f'{nombre}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, size)
    
    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        result.write(frame)
        
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
            cv2.rectangle(frame,(10,5),(525,25),(255,255,255),-1)
            cv2.putText(frame,f'Presione s, para almacenar los rostros encontrados {count}/{imagenes}',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
            cv2.imshow('frame',frame)
        if count ==  int(imagenes):
            flag = False
            break
    cap.release()
    cv2.destroyAllWindows()