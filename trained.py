import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained CNN model
model_path = r'/Users/samyakgupta/Documents/pattern recognition and anomaly detection /projects/emotion detection/Emotion_Detection_CNN-main/model.h5'
classifier = load_model(model_path)

# Load the Haar cascade classifier for face detection
face_cascade_path = r'/Users/samyakgupta/Documents/pattern recognition and anomaly detection /projects/emotion detection/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(face_cascade_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start capturing video from default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face found
    for (x, y, w, h) in faces:
        # Extract ROI (Region of Interest) containing the face
        roi_gray = gray[y:y + h, x:x + w]

        # Resize ROI to match model input size
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess the ROI for model prediction
        roi = roi_gray_resized.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Perform emotion prediction
        prediction = classifier.predict(roi)[0]
        label_index = np.argmax(prediction)
        label = emotion_labels[label_index]

        # Draw bounding box around the face and display predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
