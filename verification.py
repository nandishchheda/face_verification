import tensorflow as tf
from keras.models import load_model
import numpy as np
from align import extract_face
import cv2
import os.path as path 	

model_path = path.join('model','facenet_keras.h5')
model = load_model(model_path)

def get_model(model_path):
	return load_model(model_path)

def cosine_dist(A,B):	#A,B are 1d arrays. Note : cosine_dist = 1 - cosine_angle
	return 1 - np.dot(A,B)/(np.sqrt(np.dot(A,A))*np.sqrt(np.dot(B,B)))

def prepare(image1, image2, path=True):
	img1 = extract_face(image1, path=path).astype('float32')
	img2 = extract_face(image2, path=path).astype('float32')
	img1 = np.expand_dims((img1 - 127.5)*0.0078125 , axis=0)
	img2 = np.expand_dims((img2 - 127.5)*0.0078125 , axis=0)
	return np.concatenate((img1,img2),axis=0)

def get_embeddings(image1):	#image1.shape [160,160,3]
	embeddings = model.predict(np.expand_dims((image1-127.5)*0.0078125 , axis=0))
	return embeddings

def predict_boolean(image_pair):
	embeddings = model.predict(image_pair)
	return True if cosine_dist(embeddings[0],embeddings[1])<=0.581 else False	

if __name__ == '__main__':
	model = get_model(model_path)
	image_pair = prepare(path.join('sample','sample_img1.jpg'), path.join('sample','sample_img2.jpg'))

	print("Same" if predict_boolean(image_pair) else "Different")
	print("Testing get_embeddings")
	image = extract_face(path.join('sample','sample_img3.jpg')).astype('float32')
	print(get_embeddings(image))