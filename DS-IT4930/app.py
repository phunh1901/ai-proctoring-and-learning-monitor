import streamlit as st
import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime

# SETUP 
current_dir = os.path.dirname(os.path.abspath(__file__))
face_data_dir = os.path.join(current_dir, "ai_modules", "face_verification", "data")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ai_modules.face_verification.Face_Recognition import FaceRecog
from ai_modules.head_pose_estimation.HPE import HeadPoseDetector

st.set_page_config(page_title="Student System", layout="wide")
st.title("Hệ thống Giám sát Thi trực tuyến")

#SESSION
ss = st.session_state
ss.setdefault("exam_verified", False)
ss.setdefault("exam_started", False)
ss.setdefault("student_info", None)
ss.setdefault("violations", [])

face_recog = FaceRecog(data_dir=face_data_dir)

@st.cache_resource
def load_pose():
    return HeadPoseDetector(log_file="exam_pose_log.csv")

# MENU 
mode = st.sidebar.radio("Menu", ["Xác thực", "Thi", "Admin"])

# Xác thực
if mode == "Xác thực":
    st.header("Xác thực danh tính")

    if not ss.exam_verified:
        img = st.camera_input("Chụp ảnh")
        if img:
            frame = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("Đang đối soát..."):
                result = face_recog.verify_from_frame(frame, threshold=50)

            if result and result["match"]:
                ss.exam_verified = True
                ss.student_info = result
                st.success(f"Đúng sinh viên: {result['name']}")
            else:
                st.error("Không khớp sinh viên!")
    else:
        st.success(f"Đã xác thực: {ss.student_info['name']}")
        if st.button("Vào phòng thi"):
            ss.exam_started = True
            st.rerun()
# Exam
elif mode == "Thi":
    if not ss.exam_verified:
        st.warning("Vui lòng xác thực trước!")
    elif not ss.exam_started:
        st.info("Nhấn Vào phòng thi ở mục Xác thực")
    else:
        col1, col2 = st.columns([3,1])
        frame_view = col1.empty()
        warn_text = col1.empty()

        col2.subheader("Giám sát")
        vio_box = col2.empty()
        pose_box = col2.empty()

        if col2.button("Nộp bài"):
            ss.exam_started = False
            st.rerun()

        pose = load_pose()
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        while ss.exam_started:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            frame, data = pose.get_pose(frame)

            status = data["text"]
            angles = data["angles"]
            violation_count = data["violation_count"] 

            # UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_view.image(frame_rgb, channels="RGB")

            pose_box.info(
                f"**Hướng:** {status}\n"
                f"**X:** {angles[0]:.1f} | **Y:** {angles[1]:.1f}"
            )

            vio_box.error(f"Số lần vi phạm: {violation_count}")

            if violation_count >= 5:
                warn_text.warning("Cảnh báo: Bạn đang quay đầu quá nhiều!")

        cam.release()


#  ADMIN 
elif mode == "Admin":
    st.header("Đăng ký sinh viên")

    mssv = st.text_input("MSSV")
    name = st.text_input("Họ tên")

    imgs = [
        st.file_uploader("Ảnh 1", type=["jpg","png"]),
        st.file_uploader("Ảnh 2", type=["jpg","png"]),
        st.file_uploader("Ảnh 3", type=["jpg","png"])
    ]

    if st.button("Lưu"):
        if mssv and name and all(imgs):
            with st.spinner("Đang xử lý..."):
                ok = face_recog.register(mssv, name, imgs)
            st.success("Thành công!" if ok else "Thất bại!")
        else:
            st.error("Điền đủ thông tin!")
