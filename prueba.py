import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time


rostroCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

#eye_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_eye.xml')

#leyeCascade = cv2.CascadeClassifier('hojoI.xml')
#reyeCascade = cv2.CascadeClassifier('ojoD.xml')
#mouthCascade = cv2.CascadeClassifier('Mouth.xml')
#noseCascade = cv2.CascadeClassifier('Nariz.xml')

#leyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml')

BDFotos = []
BDFotos.append(cv2.imread('messi1.jpg'))
BDFotos.append(cv2.imread('messi2.jpg'))
BDFotos.append(cv2.imread('messi3.jpg'))

imagen = cv2.imread('Juannn.jpg') 
filtro = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

rostros = rostroCascade.detectMultiScale(
	filtro,
	scaleFactor = 1.2,
	minNeighbors = 5,
	minSize= (30,30),
	flags= cv2.CASCADE_SCALE_IMAGE
)


for (x, y, w, h) in rostros:
	cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
	crop_img = imagen[y:y+h, x:x+w]

#cv2.imshow("Rostros encontrados", imagen) 
cv2.imshow("cropped", crop_img) #Imagen del rostro recortado
cv2.waitKey(0)
cv2.destroyAllWindows()