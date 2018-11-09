import cv2
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

fotosRecopiladas = {}

BDFotos = []
fotoMessi = cv2.imread('messi3_rostro.jpg', cv2.IMREAD_GRAYSCALE)
fotoMessi = cv2.resize(fotoMessi, (300, 300))
fotoMessi = np.array(fotoMessi)
fotoRonaldo = cv2.imread('ronaldo1_rostro.jpg', cv2.IMREAD_GRAYSCALE)
fotoRonaldo = cv2.resize(fotoRonaldo, (300, 300))
fotoRonaldo = np.array(fotoRonaldo)
fotoHazard = cv2.imread('hazard1_rostro.jpg', cv2.IMREAD_GRAYSCALE)
fotoHazard = cv2.resize(fotoHazard, (300, 300))
fotoHazard = np.array(fotoHazard)
BDFotos = [fotoMessi, fotoRonaldo, fotoHazard]
target = [0, 1, 2]

arrayAuxiliar = []
arrayFotos = []
for foto in BDFotos:
	for elementos in foto:
		for elemento in elementos:
			arrayAuxiliar.append(elemento)
	arrayFotos.append(arrayAuxiliar)
	arrayAuxiliar = []

target_names = ['Messi', 'Ronaldo', 'Hazard']
#target_names.append('Messi')
#target_names.append('Ronaldo')
#target_names.append('Hazard')

fotosRecopiladas['data'] = arrayFotos
fotosRecopiladas['target'] = target
fotosRecopiladas['target_names'] = target_names

X_train, X_test, Y_train, Y_test = train_test_split(fotosRecopiladas['data'], fotosRecopiladas['target'])

redNeuronal = MLPClassifier(max_iter=100, hidden_layer_sizes=(300, 300), alpha=0.003)

redNeuronal.fit(fotosRecopiladas['data'], fotosRecopiladas['target'])

fotoAEvaluar = cv2.imread('hazard2_rostro.jpg', cv2.IMREAD_GRAYSCALE)
fotoAEvaluar = cv2.resize(fotoAEvaluar, (300, 300))

arrayFotoAEvaluar = []
for elementos in fotoAEvaluar:
	for elemento in elementos:
		arrayFotoAEvaluar.append(elemento)

print(redNeuronal.predict([arrayFotoAEvaluar]))