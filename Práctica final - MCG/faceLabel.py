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
        label = fileName[fileName.find('rostro')+6:fileName.find('_')]
        if label == '':
            label = 2
        label = int(label)
        #check if label is empty, if so, add 0

        labels.append(label)
        #read image in gray scale
        image = cv2.imread(path)
        facesData.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        #add the same label and the image flipped horizontally
        labels.append(label)
        facesData.append(cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2GRAY))
    return labels, facesData

