# -*- coding: utf-8 -*-
import cv2
import  numpy as np
class Colores():
    """docstring for Video"""
    def __init__(self):
        pass
    #1º Leemos las imagenes de la camara
    def detectarColor(self):
        
    #Estos son los rangos para cada una de las componentes
        redBajo1 = np.array([0,100,20],np.uint8)
        redAlto1 = np.array([8,255,255],np.uint8)
        redBajo2 = np.array([175,100,20],np.uint8)
        redAlto2 = np.array([179,255,255],np.uint8)
        greenBajo = np.array([40,100,20],np.uint8)
        greenAlto = np.array([70,255,255],np.uint8)
        blueBajo = np.array([110,100,20],np.uint8)
        blueAlto = np.array([130,255,255],np.uint8)

        captura = cv2.VideoCapture(1)
        
        while True:
            ret,frame=captura.read()
            if ret == True:
                #Transformamos los colores de RGB a HSV
                frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                #Determinamos los Rangos del color que queremosdetectar desde redbajo1 a redalto1 e igual con los altos, asi detecta todo espacio de rojo
                maskRed1 = cv2.inRange(frameHSV, redBajo1,redAlto1)
                maskRed2 = cv2.inRange(frameHSV, redBajo2,redAlto2)
                maskRed = cv2.add(maskRed1, maskRed2)
                mgreen = cv2.inRange(frameHSV, greenBajo, greenAlto)
                maskGreen = cv2.add(mgreen, mgreen)
                mblue = cv2.inRange(frameHSV, blueBajo, blueAlto)
                maskBlue = cv2.add(mblue, mblue)
                maskColor = cv2.add(maskRed, maskGreen)
                maskColor = cv2.add(maskColor, maskBlue)
                
                #Ahora vamos a mostrar en la imagen el color de la mascara
                #Visualizamos los colores
                cv2.imshow('mask',maskColor)
                cv2.imshow('frame', frame)
                # Se cierra cuando pulsamos la s
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
        captura.release()
        cv2.destroyAllWindows()
colores = Colores()
colores.detectarColor()