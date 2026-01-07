import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from datetime import datetime

class HeadPoseDetector:
    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        log_file="head_pose_log.csv",
        violation_seconds=7
    ):
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Logging
        self.log_file = log_file
        self.last_status = None
        self._init_csv()

        # Số lần vi phạm
        self.violation_seconds = violation_seconds
        self.looking_away_start = None
        self.violation_count = 0

    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Status", "Angle_X", "Angle_Y", "Angle_Z", "Violation_Count"])

    def log_to_csv(self, status, x, y, z):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, status, round(x,2), round(y,2), round(z,2), self.violation_count])

    def get_pose(self, image):
        img_h, img_w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        
        pose_data = {
            "text": "No Face",
            "angles": (0,0,0),
            "violation_count": self.violation_count
        }

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_3d, face_2d = [], []
                nose_2d = None

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_w/2],
                    [0, focal_length, img_h/2],
                    [0, 0, 1]
                ])

                success, rot_vec, _ = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, np.zeros((4,1))
                )

                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, *_ = cv2.RQDecomp3x3(rmat)

                x_ang = angles[0] * 360
                y_ang = angles[1] * 360
                z_ang = angles[2] * 360

                # Phát hiện hướng
                if y_ang < -10: text = "Looking Left"
                elif y_ang > 10: text = "Looking Right"
                elif x_ang < -10: text = "Looking Down"
                elif x_ang > 10: text = "Looking Up"
                else: text = "Forward"

                pose_data = {
                    "text": text,
                    "angles": (x_ang, y_ang, z_ang),
                    "violation_count": self.violation_count
                }

                # Đếm số lần vi phạm
                if text != "Forward" and text != "No Face":
                    if self.looking_away_start is None:
                        self.looking_away_start = time.time()
                    else:
                        elapsed = time.time() - self.looking_away_start
                        if elapsed >= self.violation_seconds:
                            self.violation_count += 1
                            self.log_to_csv(text, x_ang, y_ang, z_ang)
                            self.looking_away_start = None
                else:
                    self.looking_away_start = None

                # Vẽ đường mũi
                if nose_2d:
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y_ang * 10), int(nose_2d[1] - x_ang * 10))
                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    self.drawing_spec,
                    self.drawing_spec
                )

        return image, pose_data
