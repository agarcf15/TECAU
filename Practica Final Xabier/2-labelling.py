import cv2
import os
import numpy as np

labels = []
facesData = []
label = 0

for fileName in os.listdir('Rostros/'):
    labels.append(label)
    path = 'Rostros/' + fileName
    facesData.append(cv2.imread(path, label))
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")