import requests
import json
import time

def test_face_match():
    url = 'http://127.0.0.1:8080/face_match'
    # open file in binary mode
    files = {'file1': open('sample\sample_img3.jpg', 'rb'),
             'file2': open('sample\sample_img2.jpg', 'rb')}     
    resp = requests.post(url, files=files)
    print( 'face_match response:\n', json.dumps(resp.json()) )

def test_face_embeddings():
	url = 'http://127.0.0.1:8080/face_embeddings'
	files = {'file': open('sample\sample_img3.jpg', 'rb')}
	resp = requests.post(url, files=files)
	print('face_embeddings response:\n', json.dumps(resp.json()))
if __name__ == '__main__':
	# s = time.time()
	# test_face_match()
	# print("Time Required by the face_match api", time.time() - s)
	s = time.time()
	test_face_embeddings()
	print("Time required by the face_embeddings api", time.time() - s)