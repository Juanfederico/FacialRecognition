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
BDFotosConcat = np.concatenate([fotoMessi, fotoRonaldo, fotoHazard])

target = []
for i in range(3):
	for j in range(300):
		target.append(i)


target_names = ['Messi', 'Ronaldo', 'Hazard']
#target_names.append('Messi')
#target_names.append('Ronaldo')
#target_names.append('Hazard')

fotosRecopiladas['data'] = BDFotosConcat
fotosRecopiladas['target'] = target
fotosRecopiladas['target_names'] = target_names

#print(fotoMessi)
X_train, X_test, Y_train, Y_test = train_test_split(fotosRecopiladas['data'], fotosRecopiladas['target'])

redNeuronal = MLPClassifier(max_iter=3000, hidden_layer_sizes=(300, 300), alpha=0.003)

redNeuronal.fit(fotosRecopiladas['data'], fotosRecopiladas['target'])

fotoAEvaluar = cv2.imread('messi2_rostro.jpg', cv2.IMREAD_GRAYSCALE)
fotoAEvaluar = cv2.resize(fotoAEvaluar, (300, 300))
fotoAEvaluar = np.array(fotoAEvaluar)

#print(redNeuronal.predict(fotoAEvaluar))

arrayPrediccion = redNeuronal.predict(fotoAEvaluar)
prediccionMessi = 0
prediccionRonaldo = 0
prediccionHazard = 0
for numero in arrayPrediccion:
	print(numero)
	if numero==0: prediccionMessi += 1
	if numero==1: prediccionRonaldo += 1
	if numero==2: prediccionHazard += 1
print(arrayPrediccion)
print(prediccionMessi)
print(prediccionRonaldo)
print(prediccionHazard)

if prediccionMessi>prediccionRonaldo and prediccionMessi>prediccionHazard:
	print("El rostro es de Messi")
elif prediccionRonaldo>prediccionMessi and prediccionRonaldo>prediccionHazard:
	print("El rostro es de Ronaldo")
elif prediccionHazard>prediccionMessi and prediccionHazard>prediccionRonaldo:
	print("El rostro es de Hazard")

cv2.imshow("fotoAEvaluar", fotoAEvaluar) #Imagen del rostro recortado
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(redNeuronal.score(fotoAEvaluar, 0))
#print(redNeuronal.score(fotoAEvaluar, 1))
#print(redNeuronal.score(fotoAEvaluar, 2))

#print(red.score(X_test, Y_test))