import numpy as np
import cv2, time

# Define cascade root
cascade_root = '..\\0_data_lan\\import\\haarcascades\\'

# Define cascades:
face_cascade = cv2.CascadeClassifier(cascade_root+'haarcascade_frontalface_default.xml')
lefteye_cascade = cv2.CascadeClassifier(cascade_root+'haarcascade_mcs_lefteye.xml')
righteye_cascade = cv2.CascadeClassifier(cascade_root+'haarcascade_mcs_righteye.xml')
eyebox_cascade_b = cv2.CascadeClassifier(cascade_root+"haarcascade_mcs_eyepair_big.xml")

cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for x,y,w,h in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (120,120,0), 2)
		roi_gray = gray[x:x+w, y:y+h]
		roi_color = img[x:x+w, y:y+h]

		eyebox_b = eyebox_cascade_b.detectMultiScale(roi_gray)
		for ex,ey,ew,eh in eyebox_b:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (120,0,120), 2)

		lefteye = lefteye_cascade.detectMultiScale(roi_gray)
		for ex,ey,ew,eh in lefteye:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)

		rightteye = righteye_cascade.detectMultiScale(roi_gray)
		for ex,ey,ew,eh in rightteye:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)

	cv2.imshow('img', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()