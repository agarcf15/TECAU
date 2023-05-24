#Importamos OpenCV, esta importación será igual siempre igual!
import cv2
#La función imread va a leer un imagen de una ruta
# El parámetro 0 en la función imread carga la imagen en escala de grises. Si queremos cargar la imagen en color, quitamos ese parametro
imagen = cv2.imread('imagen.png',0)
#La función imwrite va a almacenar una variable de tipo imagen en una ruta
cv2.imwrite('grises.jpg', imagen)
#imshow se utilizara para visualizar las imagenes
cv2.imshow('Visualizando imagen', imagen)
#WaitKey se utiliza para cerrar las visualizaciones, podemos establecer un tiempo, que se cierre al pulsar una tecla concreta ocualquier tecla.
cv2.waitKey(1000) 
# Destruye todas las ventanas creadas
cv2.destroyAllWindows()