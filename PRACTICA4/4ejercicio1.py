# -*- coding: utf-8 -*-
import cv2
import numpy as np
#Cargamos el clasificador
clasificadorF = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clasificadorL = cv2.CascadeClassifier('haarcascade_profileface.xml')

captura = cv2.VideoCapture(1)

#Funciona tanto en color como en escala de grises
#LLamamos al clasificador, pasandole la imagen en escala de grises
#Escale factor establece cuanto se escala la imagen. Esta
#configuración reduce la imagen en un 10%. Si la imagen se escala
#demasiado se pierde información y no reconocerá todas las caras
# Si la imagen se escala poco, se van a usar mas cantidad de imagen
#aumentando el tiempo y dando falsos positivos.
#Se aplica una pirámide de imágenes ya que unos rostros pueden ocupar
#mas o menos en la imagen y es necesario para capturar la información
while True:
    ret,frame=captura.read()
    if ret == True:
        faces = clasificadorF.detectMultiScale(frame, 
                                      scaleFactor=1.2,
                                      minNeighbors=5, # los n vecinos son los cuadros delimitadores de un rostro
                                      minSize=(30,30), #Tamaño mínimo del objeto, los objetos mas pequeños son ignorados
                                      maxSize=(400,400))
    #Si se detecta algún rostro con estos parámetros del clasificador se
    #almacenan a continuación para ponerles contorno
        side = clasificadorL.detectMultiScale(frame, 
                                      scaleFactor=1.2,
                                      minNeighbors=5, # los n vecinos son los cuadros delimitadores de un rostro
                                      minSize=(30,30), #Tamaño mínimo del objeto, los objetos mas pequeños son ignorados
                                      maxSize=(400,400))
    #Si se detecta algún rostro con estos parámetros del clasificador se
    #almacenan a continuación para ponerles contorno
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    for(x,y,w,h) in side:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
            break
captura.release()
cv2.destroyAllWindows()
        

