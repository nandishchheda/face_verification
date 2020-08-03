import cv2
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from align import extract_face
from keras.models import load_model
import matplotlib.pyplot as plt
from verification import get_embeddings
import os	
model_path = os.path.join('model','facenet_keras.h5')
if __name__ == '__main__':
	print("Loading the Facenet model ...")
	model = load_model(model_path)
	img1 = cv2.cvtColor(cv2.imread(os.path.join('sample','obama.jpg')), cv2.COLOR_BGR2RGB)
	print("Loading the image ...")
	plt.imshow(img1)
	plt.show()
	print("Detecting faces ...")
	detector = MTCNN()
	face_locations = detector.detect_faces(img1)
	box = face_locations[0]['box']
	plt.imshow(img1)
	ax = plt.gca()
	rect = plt.Rectangle((box[0],box[1]),box[2],box[3],fill=False, color='red')
	ax.add_patch(rect)
	plt.show()
	print("Cropping and Aligning ...")
	cropped_image = extract_face(img1,path=False)
	plt.imshow(cropped_image)
	plt.show()
	print("Generating embeddings ...")
	print(model.predict(expand_dims(cropped_image,axis=0)))

