import numpy as np
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
import os
from os import path as p

detector = MTCNN()
#required_size (w,h) - desired size after cropping
#path = true if image(param) is path to the image_file (string) and false if its a numpy array in RGB format. By default path = true.
def extract_face(image, required_size=(160, 160), path=True):
		if path :
			if p.exists(image):
				image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
			else :
				print ("File Not Found", image)
				return	                                   
		
		face_locations = detector.detect_faces(image)          # detect faces in the image
		face_main = []
		area = 0
		for face in zip(face_locations):
			(_,_,w,h) = face[0]['box']
			if (w*h> area):
				area = w*h 
				face_main = face
		
		landmarks = face_main[0]['keypoints']
		left_eye = landmarks ['left_eye']
		right_eye = landmarks['right_eye']
		
		dY = (right_eye[1]-left_eye[1])
		dX = (right_eye[0]-left_eye[0])
		angle = np.degrees(np.arctan2(dY, dX))
		
		desiredLeftEye=(0.35, 0.35)                   #in percentage
		desiredFaceWidth = required_size[0]
		desiredFaceHeight = required_size[1]
		desiredRightEyeX = 1.0 - desiredLeftEye[0]

		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - desiredLeftEye[0])
		desiredDist *= desiredFaceWidth
		scale = desiredDist / dist
		
		eyesCenter = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
		tX = desiredFaceWidth * 0.5
		tY = desiredFaceHeight * desiredLeftEye[1]              #desired position of centre of eye = (tx,ty) 
		M[0, 2] += (tX - eyesCenter[0])                         
		M[1, 2] += (tY - eyesCenter[1])
		
		(w, h) = (desiredFaceWidth, desiredFaceHeight)      
		output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
		face_array = asarray(output)
		
		return face_array

if __name__ == '__main__':
	img_path = os.path.join('uploads','sample_img1.jpg')
	print(img_path)
	img = extract_face(img_path)
	if img is not None:
		plt.imshow(img)
		plt.show()