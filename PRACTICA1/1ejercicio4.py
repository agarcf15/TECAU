# -*- coding: utf-8 -*-

import cv2
#La función imread va a leer un imagen de una ruta
# El parámetro 0 en la función imread carga la imagen en escala de grises. Si queremos cargar la imagen en color, quitamos ese parametro
imagen = cv2.imread('afilapuntas.jpg',0)
#La función imwrite va a almacenar una variable de tipo imagen en una ruta
cv2.imwrite('afilapuntasgrises.jpg', imagen)



#130 es el umbra, solo se mostraran las imágenes entre 130 y 255
_,binarizada = cv2.threshold(imagen,200,255,cv2.THRESH_BINARY)
cv2.imshow('Grises',imagen)
cv2.imshow('Grises2',binarizada)
_,binarizada = cv2.threshold(imagen,190,255,cv2.THRESH_BINARY_INV)
cv2.imshow('Grises3',binarizada)
_,binarizada = cv2.threshold(imagen,250,255,cv2.THRESH_TRUNC)
cv2.imshow('Grises4',binarizada)
_,binarizada = cv2.threshold(imagen,9,10,cv2.THRESH_TOZERO)
cv2.imshow('Grises5',binarizada)
cv2.waitKey(0)
cv2.destroyAllWindows() 