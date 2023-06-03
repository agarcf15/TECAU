import cv2
import dataCapture as dc
import dataLabel as dl
import numpy as np

class ModelType:
    EigenFaces = 1
    FisherFaces = 2
    LBPH = 3

class ModelThreshold:
    EigenFaces = [3700]
    FisherFaces = [10]
    LBPH = [50]


#Loads the relevant model and the relevant model file name
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

#Loads the data using the images from each rostros folder and trains the model
def trainModel(modelType=ModelType.EigenFaces):
    dl.faceLabels()
    face_recognizer, modelName = loadModelAttrs(modelType)
    print('Training...')
    face_recognizer.train(dl.facesData,np.array(dl.labels))
    face_recognizer.write(modelName)
    print('Model trained and saved')


def trainThreshold(modelType=ModelType.EigenFaces):
    #Itertae over the unique labels skipping the first one (Desconocido)
    for i in list(set(dl.labels))[1:]:
        #Add a new threshold to each threshold list for this user
        ModelThreshold.EigenFaces.append(0)
        ModelThreshold.FisherFaces.append(0)
        ModelThreshold.LBPH.append(0)
        trainThresholdFromSavedVideo(modelType, f'{dl.usernames[i]}.avi', i)
    print(f'Threshold establecido para todos los usuarios')

#Uses the videos saved during the data capture fase to train the model threshold for each user
def trainThresholdFromSavedVideo(modelType=ModelType.EigenFaces, video='video.mp4', user=1):
    #Load the trained model
    face_recognizer, modelName = loadModelAttrs(modelType)
    face_recognizer.read(modelName)
    
    #Load the cascade classifier to detect faces
    CascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    capture = cv2.VideoCapture(video)
    confidence = []
    
    #Predict until 60 faces are detected to get a good average confidence
    while len(confidence) < 60:
        ret, frame = capture.read()
        if ret == False: break
        frame = cv2.flip(frame,1)
        
        #Convert image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CascadeClassifier.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            
            #Predict the image with a 150x150 size
            faceImage = gray[y:y+h,x:x+w]
            faceImage = cv2.resize(faceImage,(150,150), interpolation=cv2.INTER_CUBIC)
            
            #Predict and save the confidence
            prediction = face_recognizer.predict(faceImage)
            confidence.append(prediction[1])
            
        if len(confidence) % 20 == 0:
            print(f'Calculando threshold...')
    
    #Replace the threshold for the user with the average of the confidences
    if modelType == ModelType.EigenFaces:
        ModelThreshold.EigenFaces[user] = sum(confidence)/len(confidence)*1.08 if len(confidence) > 0 else 3700
    elif modelType == ModelType.FisherFaces:
        ModelThreshold.FisherFaces[user] = sum(confidence)/len(confidence)*1.08 if len(confidence) > 0 else 10
    elif modelType == ModelType.LBPH:
        ModelThreshold.LBPH[user] = sum(confidence)/len(confidence)*1.08 if len(confidence) > 0 else 50
    
#Capture images with the webcam and predict them showing the result in the image
def predictWebcam(modelType=ModelType.EigenFaces, modelThreshold=ModelThreshold.EigenFaces):
    #Load the model 
    face_recognizer, modelName = loadModelAttrs(modelType)
    face_recognizer.read(modelName)
    
    capture = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    print('Press "s" to close')
    while True:
        #Capture the image
        ret, frame = capture.read()
        if ret == False: break
        frame = cv2.flip(frame,1)
        
        #Convert image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Detect faces in the image
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            
            #Recognize and predict faces in the image with a 150x150 size
            faceImage = gray[y:y+h,x:x+w]
            faceImage = cv2.resize(faceImage,(150,150), interpolation=cv2.INTER_CUBIC)
            prediction = face_recognizer.predict(faceImage)

            #Create label and color green
            predi = '{:.2f}'.format(prediction[1])
            color = (0,255,0)
            
            #Check the Threshold to see if the prediction is good enough
            if prediction[1] > modelThreshold[prediction[0]]:
                #If the confidence is not within the threshold, user is unknown and the text is red
                prediction = (0, prediction[1])
                color = (0,0,255)
        
            #Draw the rectangle and label
            cv2.putText(frame, f'{dl.usernames[prediction[0]]} - {predi}', (x,y-25), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)
        
        #Show the image
        cv2.imshow('Press "s" to close',frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    capture.release()
    cv2.destroyAllWindows()
    
def addUsers():
    prompt = input('¿Deseas añadir un usuario al sistema? (y/n): ')
    while prompt == 'y':
        dc.faceCapture()
        prompt = input('¿Deseas añadir otro usuario al sistema? (y/n): ')
  
    

#Usage:
#addUsers(): Captures images from the webcam and saves them in the rostros folder
#Foreach user you will be asked to enter a name and a number, then a video will be shown and you will have to press 's' to start capturing images

#trainModel(): Loads the images from the rostros folder with their respective labels and trains the model

#trainThreshold(): Uses the videos saved during the data capture fase to train the model threshold for each user

#predictWebcam(): Captures images from the webcam and predicts them showing the result in the image
#If the prediction is not within the threshold, the user will be unknown and the text will be red

modelType = ModelType.LBPH

addUsers()
trainModel(modelType)
trainThreshold(modelType)
predictWebcam(modelType, ModelThreshold.LBPH)