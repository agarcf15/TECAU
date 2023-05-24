# -*- coding: utf-8 -*-
import cv2
class Video():
    """docstring for Video"""
    def __init__(self):
        pass
    def capturarVideo(self):
        #El argumento puede ser 0 o 1 a la hora de usar la camara del portátil, si hay varias conectadas
        captura = cv2.VideoCapture(1)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # VideoWriter nos permite guardar el video que obtengamos con la cámara del equipo
        # Los parámetros de la función son Nombre, Codec , fps, tamaño
        salida = cv2.VideoWriter('videoGuardado.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),20.0,(1920,1080) )
        while(captura.isOpened()):
            #ret es un boolean que indica si va bien y se esta grabando imagenes, imagen es la captura
            ret,imagen=captura.read()
            if ret == True:
                #Mostramos el video
                cv2.imshow('video', imagen)
                #Guardamos la imagen
                salida.write(imagen)
                #0xFF especificacion de 64bit
                # Se cierra cuando pulsamos la s
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
        captura.release()
        salida.release()
        cv2.destroyAllWindows()
    def leerVideo(self):
        print("leyendo")
        captura = cv2.VideoCapture('videoGuardado.mp4')
        while(captura.isOpened()):
            #ret es un boolean que indica si va bien y se esta grabando imagenes, imagen es la captura
            ret,imagen=captura.read()
            if ret == True:
                cv2.imshow('video', imagen)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
                else:
                    break
video = Video()
video.capturarVideo()