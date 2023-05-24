import cv2
import os
import faceCapture as fc
import faceLabel as fl
import numpy as np

class ModelType:
    EigenFaces = 1
    FisherFaces = 2
    LBPH = 3

class ModelTreshold:
    EigenFaces = 5700
    FisherFaces = 15
    LBPH = 94

preds = ['Unknown','User1','User2', 'User3', 'User4']

#Loads the relevant model and teh relevant model file name
def loadModelAttrs(modelType=ModelType.EigenFaces):
    if modelType == ModelType.EigenFaces:
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        modelName = 'modeloEigen.xml'
    elif modelType == ModelType.FisherFaces:
        face_recognizer = cv2.face.FisherFaceRecognizer_create()
        modelName = 'modeloFisher.xml'
    elif modelType == ModelType.LBPH:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        modelName = 'modeloLBPH.xml'
    return face_recognizer, modelName

#Loads the data using the images from /Rostros and trains the model
def trainModel(modelType=ModelType.EigenFaces):
    labels, facesData = fl.faceLabels()
    face_recognizer, modelName = loadModelAttrs(modelType)
    print("Training...")
    face_recognizer.train(facesData,np.array(labels))
    face_recognizer.write(modelName)
    print("Model trained and saved")

#Given an image or a path to an image, predicts the image and shows the result
def predictImage(modelType=ModelType.EigenFaces, image='faceTest.jpg', originalImage='faceTest.jpg', modelTreshold=ModelTreshold.EigenFaces):
    #Load the model data
    face_recognizer, modelName = loadModelAttrs(modelType)

    #Load the trained model
    face_recognizer.read(modelName)
    #Now we can predict the image
    #Load the image
    image = cv2.imread(image) if isinstance(image, str) else image
    originalImage = cv2.imread(originalImage) if isinstance(originalImage, str) else originalImage
    #if image is not gray scale, convert it
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Predict the image
    prediction = face_recognizer.predict(image)
    #draw text with the label and confidence
    color = (0,255,0)
    if prediction[1] > modelTreshold:
        prediction = (0, prediction[1])
        color = (0,0,255)
    cv2.putText(originalImage, '{}'.format(prediction), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    #wait for a key to exit
    cv2.imshow('image',originalImage)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.destroyAllWindows()
    
    
#Capture images with the webcam and predict them showing the result in the image
def predictWebcam(modelType=ModelType.EigenFaces, modelTreshold=ModelTreshold.EigenFaces):
    #Load the model data
    face_recognizer, modelName = loadModelAttrs(modelType)

    #Load the trained model
    face_recognizer.read(modelName)
    #Capture the image
    capture = cv2.VideoCapture(0)
    #Create the classifier
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    print("Press 's' to close")
    while True:
        #Capture the image
        ret, frame = capture.read()
        if ret == False: break
        #Convert image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Detect faces in the image
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        #Draw a rectangle around the face
        for (x,y,w,h) in faces:
            #Predict the image with a 150x150 size
            faceImage = gray[y:y+h,x:x+w]
            faceImage = cv2.resize(faceImage,(150,150), interpolation=cv2.INTER_CUBIC)
            prediction = face_recognizer.predict(faceImage)
            #draw text with the label and confidence
            #save preiction [1] as a string with 2 decimals
            predi = "{:.2f}".format(prediction[1])
            color = (0,255,0)
            #Check the treshold to see if the prediction is good enough
            if prediction[1] > modelTreshold:
                #if not, set the label to unknown and the color to red
                prediction = (0, prediction[1])
                color = (0,0,255)
            cv2.putText(frame, f'{preds[prediction[0]]} - {predi}', (x,y-25), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)
        #Show the image
        cv2.imshow('Press "s" to close',frame)
        #On any key press exit
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    capture.release()
    cv2.destroyAllWindows()

##Usage##
#fc.capturaRostros(nunmber, userNumber) -> creates a number of images in the Rostros folder to be used for training
#Run this function for each user you want to train, with a different userNumber in order from 0 to n

#trainModel(modelType) -> trains the model with the images in the Rostros folder

#predictWebcam(modelType, modelTreshold) -> predicts the image from the webcam using the model and the treshold

#predictImage(modelType, image, originalImage, modelTreshold) -> predicts the given image using the model and the treshold

##To stop the webcam prediction press 's'

#Example:
fc.capturaRostros(250, userNumber=1)
trainModel(ModelType.EigenFaces)
predictWebcam(ModelType.EigenFaces, ModelTreshold.EigenFaces)
    