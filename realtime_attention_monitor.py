# realtime_attention_monitor.py
"""
Real-time Attention & Presence Monitor (Streamlit)
→ Updated 2025-06-09: replaced st.sleep with time.sleep,
  and use_column_width → use_container_width.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
# face detection model for landmark confidence
mp_face_detection = mp.solutions.face_detection
from collections import deque
from threading import Thread

# ────────────────────────────────────────────────────────────────────────────────
# 1. Mediapipe initialisation
# ────────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

FACE_3D_MODEL_POINTS = np.array([
    (0.0,   0.0,   0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

FACE_LANDMARK_IDXS = [1, 152, 33, 263, 61, 291]

# ────────────────────────────────────────────────────────────────────────────────
# 2. Helper functions
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_models():
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    return face_mesh, face_detection, segmentor

def get_head_pose(image_points, focal_len, img_center):
    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, image_points,
        np.array([[focal_len, 0, img_center[0]],
                  [0, focal_len, img_center[1]],
                  [0, 0, 1]], dtype="double"),
        None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0, 0, 0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rot_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [float(a) for a in euler_angles]
    return yaw, pitch, roll

def is_eye_contact(landmarks, frame_w, frame_h, thresh=0.30):
    left_iris = landmarks[468]; right_iris = landmarks[473]
    left_eye_l = landmarks[33]; left_eye_r = landmarks[133]
    right_eye_l = landmarks[362]; right_eye_r = landmarks[263]

    left_eye_w = abs(left_eye_r.x - left_eye_l.x)
    right_eye_w = abs(right_eye_r.x - right_eye_l.x)

    left_off = abs((left_iris.x - (left_eye_l.x + left_eye_r.x) / 2)) / left_eye_w
    right_off = abs((right_iris.x - (right_eye_l.x + right_eye_r.x) / 2)) / right_eye_w

    return (left_off + right_off) / 2 < thresh

def classify_orientation(yaw, pitch, roll, tol=15):
    if yaw > tol:    return "RIGHT"
    if yaw < -tol:   return "LEFT"
    if pitch > tol:  return "DOWN"
    if pitch < -tol: return "UP"
    return "CENTER"

# ────────────────────────────────────────────────────────────────────────────────
# New helper: Eye Aspect Ratio for blink detection
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
BLINK_THRESH = 0.20

def eye_aspect_ratio(landmarks, idxs):
    # compute EAR
    p = landmarks
    a = np.linalg.norm(np.array([p[idxs[1]].x, p[idxs[1]].y]) - np.array([p[idxs[5]].x, p[idxs[5]].y]))
    b = np.linalg.norm(np.array([p[idxs[2]].x, p[idxs[2]].y]) - np.array([p[idxs[4]].x, p[idxs[4]].y]))
    c = np.linalg.norm(np.array([p[idxs[0]].x, p[idxs[0]].y]) - np.array([p[idxs[3]].x, p[idxs[3]].y]))
    return (a + b) / (2.0 * c)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Motion detector
# ────────────────────────────────────────────────────────────────────────────────
class MotionDetector:
    def __init__(self, buf_len=5, thresh=25):
        self.queue = deque(maxlen=buf_len)
        self.thresh = thresh

    def update(self, gray_frame):
        self.queue.append(gray_frame)
        if len(self.queue) < self.queue.maxlen:
            return "STATIC", 0.0
        diff = cv2.absdiff(self.queue[0], gray_frame)
        non_zero = cv2.countNonZero(diff)
        intensity = non_zero / (gray_frame.shape[0]*gray_frame.shape[1]) * 100
        status = "MOVING" if non_zero > self.thresh * 1000 else "STATIC"
        return status, intensity

# ────────────────────────────────────────────────────────────────────────────────
# 4. Threaded video capture
# ────────────────────────────────────────────────────────────────────────────────
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.frame = None
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.flip(frame, 1)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# ────────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config("Attention Monitor", layout="wide")
st.title("🧿 Real-Time Attention & Presence Monitor")

col_vid, col_metrics = st.columns([3,1], gap="large")
with col_vid:
    placeholder_video = st.empty()
with col_metrics:
    st.header("🔍 Live Metrics")
    metric_table = st.empty()

st.info("Please grant camera access ⬆️ (your browser will prompt).")

face_mesh, face_detection, segmentor = _init_models()
motion_detector = MotionDetector()
stream = VideoStream(0)

from collections import deque
# before loop: 
blink_times = deque()
prev_blink = False
fps_times = deque(maxlen=30)

try:
    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces for confidence
        det_results = face_detection.process(rgb)
        landmark_confidence = 0.0
        if det_results.detections:
            landmark_confidence = max([d.score[0] for d in det_results.detections]) * 100

        # track timestamp for FPS
        now = time.time()
        fps_times.append(now)
        fps = len(fps_times) / (fps_times[-1] - fps_times[0]) if len(fps_times)>1 else 0.0

        face_detected = False; eye_contact = False; head_orientation = "Unknown"; extra_person = False
        blink_status = False; blink_rate = 0
        face_distance_label = "Unknown"
        motion_level = "LOW"
        lip_status = False

        results = face_mesh.process(rgb)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        body_status, motion_intensity = motion_detector.update(gray)
        if motion_intensity < 1: motion_level="LOW"
        elif motion_intensity < 5: motion_level="MEDIUM"
        else: motion_level="HIGH"

        yaw = pitch = roll = 0.0

        if results.multi_face_landmarks:
            face_detected = True
            n_faces = len(results.multi_face_landmarks)
            extra_person = n_faces > 1

            for idx, landmarks in enumerate(results.multi_face_landmarks):
                lm = landmarks.landmark
                image_points = np.array([
                    (lm[i].x * w, lm[i].y * h)
                    for i in FACE_LANDMARK_IDXS
                ], dtype=np.float64)
                yaw, pitch, roll = get_head_pose(
                    image_points, focal_len=w, img_center=(w/2, h/2)
                )
                if idx == 0:
                    head_orientation = classify_orientation(yaw, pitch, roll)
                    eye_contact = is_eye_contact(lm, w, h)
                    # blink detection
                    ear = (eye_aspect_ratio(lm, LEFT_EYE_IDX) + eye_aspect_ratio(lm, RIGHT_EYE_IDX)) / 2.0
                    if ear < BLINK_THRESH and not prev_blink:
                        blink_times.append(now)
                        prev_blink = True
                    elif ear >= BLINK_THRESH:
                        prev_blink = False
                    blink_rate = len([t for t in blink_times if now - t < 60])
                    blink_status = ear < BLINK_THRESH

                    # face distance (% of frame area)
                    xs = [p.x for p in lm]; ys = [p.y for p in lm]
                    min_x, max_x = min(xs)*w, max(xs)*w
                    min_y, max_y = min(ys)*h, max(ys)*h
                    area_frac = ((max_x-min_x)*(max_y-min_y)) / (w*h)
                    if area_frac < 0.10: face_distance_label = "FAR"
                    elif area_frac < 0.20: face_distance_label = "IDEAL"
                    else: face_distance_label = "NEAR"

                    # lip movement (MAR)
                    upper = np.array([lm[13].x*w, lm[13].y*h]); lower = np.array([lm[14].x*w, lm[14].y*h])
                    left = np.array([lm[78].x*w, lm[78].y*h]); right = np.array([lm[308].x*w, lm[308].y*h])
                    mar = np.linalg.norm(upper-lower) / np.linalg.norm(left-right)
                    lip_status = mar > 0.03

                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1, circle_radius=1)
                )

        border_color = (0, 255, 0) if eye_contact else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 2)
        cv2.putText(
            frame,
            f"Faces:{'0' if not face_detected else ('1' if not extra_person else '>1')}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)

        with col_vid:
            placeholder_video.image(frame, channels="BGR", use_container_width=True)
        matrix_data = {
            "Metric": [
                "Face Detected","Eye Contact","Head Orientation","Body Movement","Additional Person",
                "Blink Status","Blink Rate","Face Distance","Yaw (°)","Pitch (°)","Roll (°)",
                "Motion Level","Landmark Confidence (%)","Frame Rate","Lip Movement"
            ],
            "Value": [
                "YES" if face_detected else "NO",
                "YES" if eye_contact else "NO",
                head_orientation,
                body_status,
                "YES" if extra_person else "NO",
                "YES" if blink_status else "NO",
                str(blink_rate),
                face_distance_label,
                f"{yaw:.1f}",f"{pitch:.1f}",f"{roll:.1f}",
                motion_level,
                f"{landmark_confidence:.1f}",
                f"{fps:.1f}",
                "YES" if lip_status else "NO"
            ]
        }
        with col_metrics:
            metric_table.table(matrix_data)

        # ←–– replaced st.sleep with time.sleep
        time.sleep(0.03)

except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    st.info("🛑 Stream stopped.")