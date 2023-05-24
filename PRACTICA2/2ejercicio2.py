# -*- coding: utf-8 -*-
import cv2
import numpy as np

azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)
captura = cv2.VideoCapture(0)
while True:
    ret,frame=captura.read()
    if ret == True:
        print("test")

        frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        maskAzul = cv2.inRange(frameHSV, azulBajo,azulAlto)
        print("test")
        #Vamos a obtener los contornos, para ello solo utilizaremos la segundavariables que devuelve la función
        contorno,_ = cv2.findContours(maskAzul, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        #Dibujamos los contornos, el -1 significa que dibuja todos los contornos(255,0,0) es el color en el que vamos a pintar los contornos en RGB, 3 es el grosor de la linea
        for c in contorno:
            area=cv2.contourArea(c)
            print("test")
            if area >3000:
                cv2.drawContours(frame, [c], 0, (255,0,0), 3)
                #Visualizamos los colores
                #cv2.imshow('maskAzul',maskAzul)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
        captura.release()
        cv2.destroyAllWindows()
