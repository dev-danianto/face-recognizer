import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion recognition model
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Fixed typo

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Correct XML file

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]

        # Preprocess ROI for emotion model
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension
        roi = np.expand_dims(roi, axis=-1)  # Add channel dimension (now shape: 1,64,64,1)

        # Predict emotion
        emotion_prediction = emotion_model.predict(roi)
        emotion_index = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_index]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Fixed syntax
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()