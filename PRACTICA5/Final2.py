import cv2
import os
import glob
import numpy as np

with open('nombres.txt', 'r') as archivo:
    lineas = archivo.readlines()
nombres =[]
labels = []
facesData = []
label = 0

for i in lineas:
    nombres.append(i.rstrip('\n'))
    for archivo in glob.glob('Rostros_'+i.rstrip('\n')+'/*.jpg'):
        labels.append(label)
        facesData.append(cv2.imread(archivo, label))
    label += 1
print(labels)
print(nombres)
print(facesData)
#modelos
face_recognizer1 = cv2.face.EigenFaceRecognizer_create()
#face_recognizer2 = cv2.face.FisherFaceRecognizer_create()
#face_recognizer3 = cv2.face.LBPHFaceRecognizer_create()
# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer1.train(facesData, np.array(labels))
# Almacenando el modelo obtenido
face_recognizer1.write('modeloEigenFace.xml')
#face_recognizer2.write('modeloFisherFace.xml')
#face_recognizer3.write('modeloLBPHFace.xml')
print("Modelo almacenado...")
