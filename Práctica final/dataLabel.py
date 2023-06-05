"""
Alejandro García Fernández
Dennis Wong Vasquez
Manuel Clerigué García
Xabier Barguilla Arribas
"""
import cv2
import os
import numpy as np
#Function create labels for the images


labels = [0]
facesData = []
if os.path.exists('Desconocido.jpg'):
    facesData.append(cv2.imread('Desconocido.jpg',0))
else:
    #Add a blank 150x150 image if the file doesn't exist
    facesData.append(np.zeros((150,150),dtype=np.uint8))
usernames = ['Desconocido']

def faceLabels():
    #Get all folder in the current directory and discard the ones that don't start with "Rostros_"
    folders = [folder for folder in os.listdir() if folder.find('Rostros_') == 0]
    
    for folder in folders:
        #Get the username from the folder name
        usernames.append(folder[folder.find('_')+1:])
        #Add a new label for the user
        label = len(usernames)-1

        #Foreach image in the folder
        for imagePath in os.listdir(folder):
            #Read image in gray scale
            image = cv2.imread(folder+'/'+imagePath,0)
        
            #Add the label and the image to the data
            labels.append(label)
            facesData.append(image)
        
            #add the same label and the image flipped horizontally
            labels.append(label)
            facesData.append(cv2.flip(image,1))

