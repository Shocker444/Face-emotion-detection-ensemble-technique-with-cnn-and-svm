import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from model import extractor_model

import numpy as np
import cv2

import joblib


rescale = layers.Rescaling(scale=1./127.5, offset=-1)
classes = ['suprise', 'fear', 'disgust', 'happy','sad', 'angry', 'neutral']
def export_model(image, extractor, classifier):
    image = rescale(image)
    layer_output = K.function(inputs = extractor.layers[0].input, outputs = extractor.layers[-5].output)

    extracted = layer_output(image[tf.newaxis])

    pred = classifier.predict(extracted)

    return pred

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
extractor = extractor_model()
extractor.load_weights('facemodel2.h5')
classifier = joblib.load('final_classifier4.sav')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]

        face = tf.image.resize(face, (224, 224))
        prediction = export_model(face, extractor, classifier)

        cv2.putText(frame, classes[prediction[0]], (x, y), cv2.FONT_HERSHEY_COMPLEX, .75, (255, 0, 0), thickness=2)
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(225, 0, 0), thickness=2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()