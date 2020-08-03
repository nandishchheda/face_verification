from flask import Flask, request
import json
from verification import *
from align import extract_face
import os
import tensorflow as tf

UPLOAD_FOLDER = '.'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
graph = tf.get_default_graph()

@app.route('/')
def home():
	return "Hello World!"
@app.route('/face_match', methods=['POST'])
def face_match():
    if request.method == 'POST':
        # check if the post request has the file part
        if ('file1' in request.files) and ('file2' in request.files):        
            file1 = request.files.get('file1')
            file2 = request.files.get('file2')
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], file1.filename))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], file2.filename))                         
            f1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            f2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename) 
            with graph.as_default():
            	X = prepare(f1_path,f2_path)
            with graph.as_default():
            	ret = predict_boolean(X)	
            os.remove(f1_path)
            os.remove(f2_path)   
            resp_data = {"match": ret} 
            return json.dumps(resp_data)

@app.route('/face_embeddings', methods=['POST'])
def face_embeddings():
    if request.method == 'POST':
        # check if the post request has the file part
        if ('file' in request.files):        
            file1 = request.files.get('file')
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], file1.filename))                         
            f1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)

            with graph.as_default():
            	X = extract_face(f1_path)
            with graph.as_default():
            	ret = get_embeddings(X)	
            os.remove(f1_path)
            resp_data = {"embeddings": ret.tolist()} # convert numpyarray to python list for json.dumps
            return json.dumps(resp_data)                     
    
# When debug = True, code is reloaded on the fly while saved
if __name__ == '__main__':
	app.run(host='0.0.0.0', port='8080',debug=True)
	