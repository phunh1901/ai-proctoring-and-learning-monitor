import streamlit as st
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# ================= SETUP =================
current_dir = os.path.dirname(os.path.abspath(__file__))
face_data_dir = os.path.join(current_dir, "ai_modules", "face_verification", "data")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import models
from ai_modules.face_verification.Face_Recognition import FaceRecog
from ai_modules.head_pose_estimation.HPE import HeadPoseDetector
from ai_modules.face_emotion.face_emotion import EmotionDetector

st.set_page_config(page_title="Student System", layout="wide")

# ================= SESSION =================
ss = st.session_state
ss.setdefault("exam_verified", False)
ss.setdefault("mode_selected", None)
ss.setdefault("exam_started", False)
ss.setdefault("student_info", None)
ss.setdefault("show_register", False)
ss.setdefault("page", "auth")  # auth, mode_select, learning, exam

# ================= LOAD MODELS =================
@st.cache_resource
def load_face_recog():
    return FaceRecog(data_dir=face_data_dir)

@st.cache_resource
def load_emotion_detector():
    return EmotionDetector()

@st.cache_resource
def load_pose_detector():
    return HeadPoseDetector(log_file="exam_pose_log.csv")

face_recog = load_face_recog()
emotion_detector = load_emotion_detector()

# ================= HEADER =================
st.title("Hệ thống Giám sát – Học & Thi trực tuyến")

# ================= AUTHENTICATION PAGE =================
if ss.page == "auth":
    
    # Toggle between Auth and Register
    tab1, tab2 = st.tabs(["Xác thực danh tính", "Đăng ký sinh viên mới"])
    
    # ===== TAB 1: AUTHENTICATION =====
    with tab1:
        st.header("Xác thực danh tính")
        st.info("Chụp ảnh khuôn mặt của bạn để xác thực")
        
        img = st.camera_input(" Chụp ảnh", key="auth_camera")
        
        if img:
            # Decode image
            frame = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            
            # Display captured image
            show_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(show_rgb, caption="Ảnh xác thực", width=300)
            
            with st.spinner("Đang đối soát khuôn mặt..."):
                result = face_recog.verify_from_frame(frame, threshold=50)
                
                if result and result["match"]:
                    distance = result.get("distance", 0)
                    similarity = max(0, 100 - distance)
                    
                    ss.exam_verified = True
                    ss.student_info = result
                    ss.page = "mode_select"
                    
                    st.success(f" Xác thực thành công!")
                    st.info(f"**Sinh viên:** {result['name']}")
                    st.info(f"**Độ tương đồng:** {similarity:.2f}%")
                    
                    if st.button(" Tiếp tục", type="primary"):
                        st.rerun()
                else:
                    st.error(" Không tìm thấy sinh viên phù hợp!")
                    st.warning("Vui lòng thử lại hoặc đăng ký nếu bạn là sinh viên mới.")
    
    # ===== TAB 2: REGISTRATION =====
    with tab2:
        st.header("Đăng ký sinh viên mới")
        
        with st.form("registration_form", clear_on_submit=True):
            st.subheader("Thông tin sinh viên")
            
            col1, col2 = st.columns(2)
            with col1:
                mssv = st.text_input("MSSV *", placeholder="Ví dụ: 20210001")
            with col2:
                name = st.text_input("Họ và tên *", placeholder="Ví dụ: Nguyễn Văn A")
            
            st.divider()
            st.subheader("Ảnh chân dung (3 ảnh)")
            st.caption("Lưu ý: Chụp ở nhiều góc độ khác nhau để tăng độ chính xác")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                img1 = st.file_uploader("Ảnh 1 - Nhìn thẳng", type=["jpg","png","jpeg"], key="reg_img1")
            with col2:
                img2 = st.file_uploader("Ảnh 2 - Nghiêng trái", type=["jpg","png","jpeg"], key="reg_img2")
            with col3:
                img3 = st.file_uploader("Ảnh 3 - Nghiêng phải", type=["jpg","png","jpeg"], key="reg_img3")
            
            # Display preview images
            if img1 or img2 or img3:
                st.subheader("Xem trước ảnh")
                preview_cols = st.columns(3)
                for idx, (img, col) in enumerate(zip([img1, img2, img3], preview_cols)):
                    if img:
                        col.image(img, caption=f"Ảnh {idx+1}", use_container_width=True)
            
            submit = st.form_submit_button("Lưu đăng ký", type="primary", use_container_width=True)
            
            if submit:
                # Validation
                if not mssv or not name:
                    st.error(" Vui lòng điền đầy đủ MSSV và Họ tên!")
                elif img1 is None or img2 is None or img3 is None:
                    st.error("Vui lòng tải lên đủ 3 ảnh!")
                else:
                    # Process registration
                    with st.spinner(" Đang xử lý đăng ký..."):
                        try:
                            imgs = [img1, img2, img3]
                            success = face_recog.register(mssv, name, imgs)
                            
                            # Check if success is a boolean or other type
                            if success is True or (hasattr(success, '__len__') and len(success) > 0):
                                st.success("Đăng ký thành công!")
                                st.balloons()
                                st.info("Bạn có thể quay lại tab **Xác thực danh tính** để đăng nhập.")
                            else:
                                st.error("Đăng ký thất bại! Vui lòng thử lại.")
                                st.warning("Lỗi có thể do: ảnh không rõ mặt, góc chụp không phù hợp, hoặc MSSV đã tồn tại.")
                        except Exception as e:
                            st.error(f"Lỗi hệ thống: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

# ================= MODE SELECTION PAGE =================
elif ss.page == "mode_select":
    st.success(f"Xin chào, **{ss.student_info['name']}**")
    
    st.header("Chọn chế độ hệ thống")
    st.write("Vui lòng chọn chế độ bạn muốn sử dụng:")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.divider()
        
        if st.button("Learning Mode - Học tập", use_container_width=True, type="primary"):
            ss.page = "learning"
            st.rerun()
        
        st.caption("Chế độ giám sát cảm xúc trong quá trình học")
        
        st.divider()
        
        if st.button("Exam Mode - Thi cử", use_container_width=True, type="primary"):
            ss.page = "exam"
            st.rerun()
        
        st.caption("Chế độ giám sát tư thế đầu khi làm bài thi")
        
        st.divider()
        
        if st.button("Đăng xuất", use_container_width=True):
            # Reset session
            ss.exam_verified = False
            ss.student_info = None
            ss.page = "auth"
            st.rerun()

# ================= LEARNING MODE =================
elif ss.page == "learning":
    st.header("earning Mode - Giám sát học tập")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Điều khiển")
        if st.button("Quay lại", use_container_width=True):
            ss.page = "mode_select"
            st.rerun()
        
        st.divider()
        st.subheader("Trạng thái")
        emotion_box = st.empty()
        st.caption("Giám sát cảm xúc trong học tập")
    
    with col1:
        st.subheader("Camera giám sát")
        
        # Initialize camera control in session state
        if 'learning_active' not in ss:
            ss.learning_active = False
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("Bắt đầu giám sát", disabled=ss.learning_active, use_container_width=True):
                ss.learning_active = True
                st.rerun()
        
        with col_stop:
            if st.button("Dừng giám sát", disabled=not ss.learning_active, use_container_width=True):
                ss.learning_active = False
                st.rerun()
        
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        # Camera streaming
        if ss.learning_active:
            cam = cv2.VideoCapture(0)
            
            # Check if camera opened successfully
            if not cam.isOpened():
                st.error("Không thể mở camera! Vui lòng kiểm tra:")
                st.warning("- Camera có được kết nối không?\n- Ứng dụng khác có đang sử dụng camera không?\n- Trình duyệt có quyền truy cập camera không?")
                ss.learning_active = False
            else:
                status_text.success("Camera đang hoạt động")
                
                try:
                    while ss.learning_active:
                        ret, frame = cam.read()
                        if not ret:
                            st.error("Không thể đọc frame từ camera!")
                            break
                        
                        frame = cv2.flip(frame, 1)
                        
                        # Detect emotion
                        try:
                            frame, emotion = emotion_detector.get_emotion(frame)
                            emotion_box.info(f"Cảm xúc: **{emotion}**")
                        except Exception as e:
                            emotion_box.warning(f"Không phát hiện được cảm xúc")
                            emotion = "Unknown"
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                finally:
                    cam.release()
                    cv2.destroyAllWindows()
        else:
            frame_placeholder.info("Nhấn 'Bắt đầu giám sát' để khởi động camera")

# ================= EXAM MODE =================
elif ss.page == "exam":
    st.header("Exam Mode - Giám sát thi cử")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Điều khiển")
        if st.button("Nộp bài", type="primary", use_container_width=True):
            if 'exam_active' in ss:
                ss.exam_active = False
            ss.page = "mode_select"
            st.rerun()
        
        st.divider()
        st.subheader("Thống kê")
        violation_box = st.empty()
        pose_info_box = st.empty()
        warning_box = st.empty()
    
    with col1:
        st.subheader("Camera giám sát")
        
        # Initialize camera control in session state
        if 'exam_active' not in ss:
            ss.exam_active = False
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("Bắt đầu thi", disabled=ss.exam_active, use_container_width=True, type="primary"):
                ss.exam_active = True
                st.rerun()
        
        with col_stop:
            if st.button("Tạm dừng", disabled=not ss.exam_active, use_container_width=True):
                ss.exam_active = False
                st.rerun()
        
        frame_placeholder = st.empty()
        status_text = st.empty()
        
        # Camera streaming
        if ss.exam_active:
            # Load pose detector
            pose_detector = load_pose_detector()
            
            # Start camera
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Check if camera opened successfully
            if not cam.isOpened():
                st.error("Không thể mở camera! Vui lòng kiểm tra:")
                st.warning("- Camera có được kết nối không?\n- Ứng dụng khác có đang sử dụng camera không?\n- Trình duyệt có quyền truy cập camera không?")
                ss.exam_active = False
            else:
                status_text.success("Camera đang hoạt động - Bài thi đang được giám sát")
                
                try:
                    while ss.exam_active:
                        ret, frame = cam.read()
                        if not ret:
                            st.error("Không thể đọc frame từ camera!")
                            break
                        
                        frame = cv2.flip(frame, 1)
                        
                        # Detect head pose
                        try:
                            frame, data = pose_detector.get_pose(frame)
                            
                            status = data.get("text", "Unknown")
                            angles = data.get("angles", [0, 0, 0])
                            violation_count = data.get("violation_count", 0)
                            
                            # Display statistics
                            violation_box.metric("Số lần vi phạm", violation_count)
                            
                            pose_info_box.info(
                                f"**Hướng nhìn:** {status}\n\n"
                                f"**Góc X (Pitch):** {angles[0]:.1f}°\n\n"
                                f"**Góc Y (Yaw):** {angles[1]:.1f}°\n\n"
                                f"**Góc Z (Roll):** {angles[2]:.1f}°"
                            )
                            
                            # Warning for excessive violations
                            if violation_count >= 5:
                                warning_box.error(
                                    f"**CẢNH BÁO:** Bạn đã vi phạm {violation_count} lần!\n\n"
                                    "Vui lòng giữ mắt nhìn thẳng vào màn hình."
                                )
                            elif violation_count >= 3:
                                warning_box.warning(
                                    f"Cảnh báo: {violation_count} lần vi phạm. "
                                    "Hãy chú ý giữ tư thế!"
                                )
                            else:
                                warning_box.success("Tư thế tốt")
                        
                        except Exception as e:
                            warning_box.warning(f"Không phát hiện được tư thế")
                        
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                finally:
                    cam.release()
                    cv2.destroyAllWindows()
        else:
            frame_placeholder.info("Nhấn 'Bắt đầu thi' để khởi động camera và bắt đầu giám sát")