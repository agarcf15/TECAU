import cv2
import os
#Function create labels for the images
def faceLabels():
    labels = []
    facesData = []
    label = 0
    #Go over all the folders inside the folder faces
    #The label will be the nubmer that is just before the "_"
    for fileName in os.listdir('Rostros/'):
        #Path of the image
        path = 'Rostros/' + fileName
        #print('Labels: ',label,'- Path: ',path)
        #Add the label and the image to the list
        #Label is the text after "rostro" and before "_"
        label = int(fileName[fileName.find('rostro')+6:fileName.find('_')])
        labels.append(label)
        #read image in gray scale
        image = cv2.imread(path)
        facesData.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return labels, facesData

