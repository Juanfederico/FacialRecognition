import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os

print("Funcionan las librerias")

img = cv2.imread('messi2.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('image', img)

for i, line in enumerate(img):
	for j, pixel in enumerate(line):
		if pixel>220: contadorClaro += 1
		if pixel<40: contadorOscuro += 1

if contadorClaro>contadorOscuro: print("La imagen es mas clara")
else: print("La imagen es mas oscura")

#cv2.rectangle(img, (20, 40), (20, 40), (255,0,0), 2)
cv2.imshow('image', img)
#cv2.waitKey(0)

cap = cv2.VideoCapture(0)
contador = 0
while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()
	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Display the resulting frame
	image = cv2.imshow('frame',frame)
	contador += 1
	if contador>100:
		if np.mean(frame) < 30:
			#print("La pantalla esta oscura")
			os.system('cls')
			break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()