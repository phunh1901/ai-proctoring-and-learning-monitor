import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from ultralytics import YOLO


class EmotionDetector:
    def __init__(self, weight_path=None, yolo_path=None):
        # ---- Build FER CNN Model ----
        tf.keras.backend.clear_session()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if weight_path is None:
            weight_path = os.path.join(current_dir, "model_weights.h5")
        if yolo_path is None:
            yolo_path = os.path.join(current_dir, "best.pt")
        
        self.model = Sequential()

        # 1st conv block
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(BatchNormalization())

        # 2nd conv block
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))

        # 3rd + 4th conv
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))

        # 5th + 6th conv
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))

        # FC
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            metrics=["accuracy"]
        )

        # Load weights
        self.model.load_weights(weight_path)

        # YOLO Detector
        self.face_detector = YOLO(yolo_path)

        # Emotion labels
        self.labels = [
            "angry",
            "disgusted",
            "fearful",
            "happy",
            "neutral",
            "sad",
            "surprise"
        ]

    # =========================
    # 🔥 Main function to call
    # =========================
    def get_emotion(self, frame):
        """
        Input:  BGR frame
        Output: (annotated_frame, top_emotion or None)
        """
        results = self.face_detector(frame, conf=0.5, verbose=False)
        detected_emotion = None

        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # ---- CNN Preprocess ----
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                gray = gray / 255.0
                gray = np.reshape(gray, (1, 48, 48, 1))

                preds = self.model.predict(gray, verbose=0)
                emotion = self.labels[np.argmax(preds)]
                detected_emotion = emotion

                # Draw UI
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2
                )

        return frame, detected_emotion
