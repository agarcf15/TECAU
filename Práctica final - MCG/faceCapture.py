import cv2
import os

def captureFace():
    capture = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        #Try to capture a face, once captured, return the image and break the loop
        ret, frame = capture.read()
        if ret == False: break
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),
            interpolation=cv2.INTER_CUBIC)            
            return rostro

def capturaRostro():
    #Creamos una carpeta para almacenar los rostros si esta no existe
    if not os.path.exists('Rostros'):
        print('Carpeta creada: Rostros encontrados')
        os.makedirs('Rostros')

    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if os.path.exists('faceTest.jpg'): os.remove('faceTest.jpg')
    count = 0
    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        k = cv2.waitKey(1)
        if k == 27: break
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150), 
            interpolation=cv2.INTER_CUBIC)
            if k == ord('s'):
                cv2.imwrite('faceTest.jpg'.format(count),rostro)
                cv2.imshow('rostro',rostro)
                count = count +1
                return
            cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
            cv2.putText(frame,'Presione s, para almacenar los rostros encontrados',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
            cv2.imshow('frame',frame)
            if os.path.exists('faceTest.jpg'): break
    cap.release()
    cv2.destroyAllWindows()

#Modifica el codigo anterior para crear una función que capture n imágenes de rostros y las almacene en la carpeta Rostros encontrados
def capturaRostros(n, userNumber):
    if not os.path.exists('Rostros'):
        print('Carpeta creada: Rostros encontrados')
        os.makedirs('Rostros')
    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        #For each face detected save it in a file
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150), 
            interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'Rostros/rostro{userNumber}_{count}.jpg',rostro)
            count = count +1
        if count >= n:
            break
        if count % 20 == 0:
            print('Se han capturado {} rostros'.format(count))
        #show the image and the count of faces captured
        cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
        cv2.putText(frame,'Rostros capturados: {}'.format(count),(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
    cap.release()
    cv2.destroyAllWindows()