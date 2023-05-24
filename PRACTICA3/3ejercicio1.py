# -*- coding: utf-8 -*-
import cv2

img1=cv2.imread('imagen1.jpg')
img2=cv2.imread('imagen2.jpg')
#cv2.add nos sirve para añadir imagenes y sumarlas, estas imagenes tienen que tener las mismas dimensiones
suma = cv2.add(img1,img2)
#Tambien podemos restar imagenes ya pero el valor de los pixeles tenderan mas a 0 cualquier valor < que 0 es 0
resta = cv2.subtract(img1,img2)
#Tambien se puede restar y quedarse con el valor absoluto |-132| = 132
absoluto= cv2.absdiff(img1,img2)
#Con los siguientes prints podemos visualizar si los pixeles de una parte de las imagenes son iguales
print (img1[0:3,0:3])
print (img2[0:3,0:3])
#Y de su suma. OJO si la suma es > 255 entonces devuelve 255
print (suma[0:3,0:3])

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
corr_coef = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
similarity = round((corr_coef + 1) / 2 * 100, 2)
print (similarity)
# Convertir la imagen de diferencia a escala de grises
gray = cv2.cvtColor(absoluto, cv2.COLOR_BGR2GRAY)
# Aplicar un umbral a la imagen de diferencia para obtener una imagen binaria
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2.imshow('suma', suma)
#cv2.imshow('resta', resta)
#cv2.imshow('absoluto', absoluto)
# Mostrar la imagen de diferencia binaria
cv2.imshow('Diferencia', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()




