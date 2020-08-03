# Face Verification using FaceNet and MTCNN

A Face Verification model using the keras implementation of Google FaceNet and MTCNN for alignment. Suitable for real-time face verification task. 

The weights for the pre-trained FaceNet model can be found [here]([https://github.com/nyoki-mtl/keras-facenet]).

To use the REST API endpoints for the model run the api.py script to launch the rest service on a local server followed by the test_client.py script to make a client request to the API endpoints. 

    python test_client.py

## Requirements
numpy
Flask == 1.1.2
matplotlib >= 1.4.3
tensorflow == 1.7
keras == 2.1.5
mtcnn == 0.1.0
requests
opencv-python>=4.1

## Results
The model was tested on [LFW Dataset]([http://vis-www.cs.umass.edu/lfw/]). 

    python lfw.py
**Accuracy:** 0.9818333333333333 
**Time Required for each pair:** 0.28122927697499595 +/- 0.08248103882603905

## References

 - https://github.com/nyoki-mtl/keras-facenet (Google FaceNet weights)
 - [https://medium.com/@Intellica.AI/a-guide-for-building-your-own-face-detection-recognition-system-910560fe3eb7](https://medium.com/@Intellica.AI/a-guide-for-building-your-own-face-detection-recognition-system-910560fe3eb7)
 - [https://towardsdatascience.com/build-face-recognition-as-a-rest-api-4c893a16446e](https://towardsdatascience.com/build-face-recognition-as-a-rest-api-4c893a16446e)
 - [https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/)
