import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from model import extractor_model
import cv2
import joblib

mp_face_detection = mp.solutions.face_detection
rescale = layers.Rescaling(scale=1./127.5, offset=-1)
classes = ['suprise', 'fear', 'disgust', 'happy','sad', 'angry', 'neutral']


def export_model(image, extractor, classifier):
    image = rescale(image)
    layer_output = K.function(inputs = extractor.layers[0].input, outputs = extractor.layers[-5].output)

    extracted = layer_output(image[tf.newaxis])

    pred = classifier.predict(extracted)

    return pred


extractor = extractor_model()
extractor.load_weights('facemodel2.h5')
classifier = joblib.load('final_classifier4.sav')

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=1) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image = frame.copy()
        height, width, channel = image.shape

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        if results.detections:
            for detection in results.detections:

                bb = detection.location_data.relative_bounding_box

                x, y, w, h = int(bb.xmin*width), int(bb.ymin*height), int(bb.width*width), int(bb.height*height)
                face = image[y:y + h, x:x + w]
                face = tf.image.resize(face, (224, 224))
                prediction = export_model(face, extractor, classifier)

                cv2.putText(frame, classes[prediction[0]], (x, y), cv2.FONT_HERSHEY_COMPLEX, .75, (255, 100, 150), thickness=2)
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(225, 255, 255), thickness=2)


        cv2.imshow('MediaPipe Face Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()