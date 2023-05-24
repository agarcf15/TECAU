# -*- coding: utf-8 -*-
import cv2
import numpy as np
#Capturamos video de la web
video = cv2.VideoCapture(1)
#Con este contador vamos a ir tomando imagenes y luego restandolas para detectar movimiento
contador = 0
while True:
    ret, frame = video.read()
    if ret == False: break
    #Cambiamos Frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Establecemos el fondo
    if contador == 20:
        bgGray = gray
    if contador > 20:
        #Resta entre las imagenes que recibimos y el fondo, asi distinguimos que se esta moviendo
        dif = cv2.absdiff(gray, bgGray)
        _, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, cnts,-1, (0,0,255),2)
        #cv2.imshow('dif',dif)
        cv2.imshow('th',th)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 9000:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
    cv2.imshow('Frame', frame)
    contador = contador +1
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
video.release()
cv2.destroyAllWindows()
