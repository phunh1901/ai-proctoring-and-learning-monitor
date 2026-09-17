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
        
        # OPTIMIZATION: Warm-up the model
        dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
        self.model.predict(dummy_input, verbose=0)

        # YOLO Detector with optimization
        self.face_detector = YOLO(yolo_path)
        
        # OPTIMIZATION: Warm-up YOLO
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.face_detector(dummy_frame, conf=0.5, verbose=False)

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
        
        # OPTIMIZATION: Frame skip counter
        self.frame_count = 0
        self.skip_frames = 2  # Process every 3rd frame
        self.last_emotion = None
        self.last_boxes = []

    
    def get_emotion(self, frame):
        self.frame_count += 1
        
        # OPTIMIZATION: Process every Nth frame
        if self.frame_count % self.skip_frames != 0:
            # Use last detection results
            return self._draw_last_results(frame), self.last_emotion
        
        # Detect faces with YOLO
        results = self.face_detector(frame, conf=0.5, verbose=False)
        detected_emotion = None
        current_boxes = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            
            # OPTIMIZATION: Process only first face if multiple detected
            if len(boxes) > 0:
                box = boxes[0]  # Take only the first face
                x1, y1, x2, y2 = map(int, box)
                current_boxes.append((x1, y1, x2, y2))

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                #  CNN Preprocess 
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                gray = gray.astype(np.float32) / 255.0  # ⚡ Faster conversion
                gray = np.reshape(gray, (1, 48, 48, 1))

                # OPTIMIZATION: Use batch prediction
                preds = self.model.predict(gray, verbose=0)
                emotion = self.labels[np.argmax(preds)]
                detected_emotion = emotion

                # Draw UI
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # OPTIMIZATION: Draw smaller text
                font_scale = 0.7
                thickness = 2
                cv2.putText(
                    frame,
                    emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    thickness
                )
                
                # Only process first face
                break

        # Save last results
        self.last_emotion = detected_emotion if detected_emotion else self.last_emotion
        self.last_boxes = current_boxes

        return frame, self.last_emotion
    
    
    def _draw_last_results(self, frame):
        for (x1, y1, x2, y2) in self.last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.last_emotion:
                cv2.putText(
                    frame,
                    self.last_emotion,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
        return frame
    
    
    def set_skip_frames(self, skip_frames):
        self.skip_frames = max(0, skip_frames)