import cv2
import numpy as np
import pickle
import os
import shutil
from ai_modules.face_verification.model_loader import load_resnet, load_mtcnn
from keras.applications.resnet50 import preprocess_input

class FaceRecog:
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.DATA_FILE = os.path.join(self.DATA_DIR, "data.pkl")
        self.STUDENTS_DIR = os.path.join(self.DATA_DIR, "Students")

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.STUDENTS_DIR, exist_ok=True)

        # Load models
        self.resnet = load_resnet()
        self.detector = load_mtcnn()
        
        # Cache database để không phải đọc file mỗi lần
        self._db_cache = None
        self._db_cache_time = 0

    def detect_and_extract_face(self, image):
        # Phát hiện và cắt khuôn mặt
        results = self.detector.detect_faces(image)
        if len(results) == 0:
            return None

        # Lấy khuôn mặt có confidence cao nhất
        face = max(results, key=lambda x: x['confidence'])
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        face_img = image[y:y+h, x:x+w]
        return face_img

    def get_embedding(self, face_image):
        # Tạo embedding vector từ khuôn mặt
        if face_image is None:
            return None

        # Resize và preprocess
        face_image = cv2.resize(face_image, (224, 224))
        face_image = face_image.astype('float32')
        face_image = np.expand_dims(face_image, axis=0)
        face_image = preprocess_input(face_image)
    
        # Extract embedding (tắt verbose để giảm lag)
        embedding = self.resnet.predict(face_image, verbose=0)[0]
        return embedding

    def calculate_distance(self, emb1, emb2):
        # Tính khoảng cách Euclidean giữa 2 embeddings
        return np.linalg.norm(emb1 - emb2)

    def register(self, student_id, name, image_files):
        """Đăng ký sinh viên mới"""
        person_folder = os.path.join(self.STUDENTS_DIR, student_id)
        os.makedirs(person_folder, exist_ok=True)

        embeddings = []
    
        # Xử lý từng ảnh
        for idx, img_file in enumerate(image_files):
            if img_file is None:
                continue
            
            # Đọc ảnh
            arr = np.frombuffer(img_file.getvalue(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
            if img is None:
                continue
        
            # Lưu ảnh gốc
            img_path = os.path.join(person_folder, f"{idx+1}.jpg")
            cv2.imwrite(img_path, img)
        
            # Detect face và tạo embedding
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = self.detect_and_extract_face(img_rgb)

            if face is None:
                continue

            emb = self.get_embedding(face)
            if emb is not None:
                embeddings.append(emb)

        # Kiểm tra có face hợp lệ không
        if len(embeddings) == 0:
            shutil.rmtree(person_folder)
            return None

        # Trung bình các embeddings
        final_embedding = np.mean(embeddings, axis=0)

        # Load hoặc tạo database
        if os.path.exists(self.DATA_FILE):
            try:
                with open(self.DATA_FILE, "rb") as f:
                    data = pickle.load(f)
            except:
                data = {}
        else:
            data = {}

        # Lưu thông tin sinh viên
        data[student_id] = {
            "name": name,
            "embedding": final_embedding.tolist(),
            "folder": student_id
        }

        # Ghi vào file
        with open(self.DATA_FILE, "wb") as f:
            pickle.dump(data, f)

        # Clear cache
        self._db_cache = None
        
        return final_embedding

    def _load_database(self):
        # Load database với caching
        current_time = os.path.getmtime(self.DATA_FILE) if os.path.exists(self.DATA_FILE) else 0
        
        # Nếu cache còn mới, dùng cache
        if self._db_cache is not None and self._db_cache_time >= current_time:
            return self._db_cache
        
        # Load database mới
        if not os.path.exists(self.DATA_FILE):
            return None
            
        with open(self.DATA_FILE, "rb") as f:
            data = pickle.load(f)
        
        # Update cache
        self._db_cache = data
        self._db_cache_time = current_time
        
        return data

    def verify_from_frame(self, frame, threshold=50):
        # Load database
        data = self._load_database()
        if data is None:
            return None

        # Convert sang RGB và detect face
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = self.detect_and_extract_face(img_rgb)
        if face is None:
            return None

        # Tạo embedding
        emb_input = self.get_embedding(face)
        if emb_input is None:
            return None

        # Tìm người gần nhất
        best_id = None
        best_name = None
        best_distance = float('inf')

        for student_id, info in data.items():
            emb_db = np.array(info["embedding"])
            dist = self.calculate_distance(emb_input, emb_db)

            if dist < best_distance:
                best_distance = dist
                best_id = student_id
                best_name = info["name"]

        is_match = best_distance < threshold

        return {
            "student_id": best_id,
            "name": best_name,
            "distance": best_distance,
            "match": is_match
        }