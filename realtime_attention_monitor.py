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
from collections import deque

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from streamlit_autorefresh import st_autorefresh

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
    segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    return face_mesh, segmentor

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

# Plain ICE config dict (STUN only)
# Force relay over public TURN (openrelay)
RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                # TCP 443 guarantees egress from most PaaS
                "turn:openrelay.metered.ca:443?transport=tcp"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ],
    # <--- force browser & aiortc to skip host/srflx candidates
    "iceTransportPolicy": "relay"
}

class AttentionProcessor(VideoProcessorBase):
    """
    VideoProcessor that applies Mediapipe face/eye/head/body metrics per frame.
    Stores latest metrics in self.metrics.
    """
    def __init__(self):
        super().__init__()
        self.metrics = {}

        # Initialize models and motion detector here for use in processor
        global face_mesh
        global motion_detector
        global face_detection
        # Initialize face_mesh and motion_detector if not already
        if 'face_mesh' not in globals():
            face_mesh, _ = _init_models()
        if 'motion_detector' not in globals():
            motion_detector = MotionDetector()
        # Face detection model for confidence
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.prev_blink = False
        self.blink_times = deque(maxlen=100)
        self.prev_time = time.time()
        self.fps = 0.0
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Run face detection for confidence
        det = self.face_detection.process(rgb)
        landmark_confidence = 0.0
        if det.detections:
            landmark_confidence = max([d.score[0] for d in det.detections]) * 100

        # Motion detection
        status = "STATIC"
        intensity = 0
        if hasattr(motion_detector, 'update'):
            res_motion = motion_detector.update(gray)
            if isinstance(res_motion, tuple):
                status, intensity = res_motion
            else:
                status = res_motion
                intensity = 0

        if intensity < 1: motion_level="LOW"
        elif intensity < 5: motion_level="MEDIUM"
        else: motion_level="HIGH"

        # FaceMesh inference
        res = face_mesh.process(rgb)
        # Default metrics
        face_detected = False; eye_contact=False; head_orientation="Unknown"; extra_person=False
        blink_status=False; blink_rate=0; face_distance="Unknown"; lip_status=False
        yaw=pitch=roll=0.0

        now = time.time()
        delta = now - self.prev_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.prev_time = now

        # track blink times in global deque
        if res.multi_face_landmarks:
            face_detected=True
            n_faces=len(res.multi_face_landmarks)
            extra_person = n_faces>1
            for idx, lm in enumerate(res.multi_face_landmarks):
                pts = lm.landmark
                # head pose
                img_pts = np.array([(pts[i].x*w, pts[i].y*h) for i in FACE_LANDMARK_IDXS], dtype=np.float64)
                yaw,pitch,roll = get_head_pose(img_pts, focal_len=w, img_center=(w/2,h/2))
                if idx==0:
                    eye_contact = is_eye_contact(pts, w, h)
                    # blink
                    # Define eye aspect ratio function and indices for blink detection
                    def eye_aspect_ratio(landmarks, eye_indices):
                        # Compute euclidean distances between vertical eye landmarks
                        A = math.dist((landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y),
                                      (landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y))
                        B = math.dist((landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y),
                                      (landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y))
                        # Compute euclidean distance between horizontal eye landmarks
                        C = math.dist((landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y),
                                      (landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y))
                        ear = (A + B) / (2.0 * C)
                        return ear
                    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
                    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
                    BLINK_THRESH = 0.2

                    ear=(eye_aspect_ratio(pts,LEFT_EYE_IDX)+eye_aspect_ratio(pts,RIGHT_EYE_IDX))/2.0
                    if ear<BLINK_THRESH and not self.prev_blink:
                        self.blink_times.append(now)
                        self.prev_blink=True
                    elif ear>=BLINK_THRESH:
                        self.prev_blink=False
                    blink_rate = len([t for t in self.blink_times if now-t<60])
                    blink_status = ear<BLINK_THRESH
                    # face distance
                    xs = [p.x for p in pts]; ys = [p.y for p in pts]
                    min_x, max_x = min(xs) * w, max(xs) * w
                    min_y, max_y = min(ys) * h, max(ys) * h
                    area_frac = ((max_x - min_x) * (max_y - min_y)) / (w * h)
                    if area_frac<0.10: face_distance="FAR"
                    elif area_frac<0.20: face_distance="IDEAL"
                    else: face_distance="NEAR"
                    # lip movement
                    upper=np.array([pts[13].x*w,pts[13].y*h]); lower=np.array([pts[14].x*w,pts[14].y*h])
                    left_pt=np.array([pts[78].x*w,pts[78].y*h]); right_pt=np.array([pts[308].x*w,pts[308].y*h])
                    mar=np.linalg.norm(upper-lower)/np.linalg.norm(left_pt-right_pt)
                    lip_status=mar>0.03
        # Save metrics
        self.metrics = {
            "Face Detected": "YES" if face_detected else "NO",
            "Eye Contact": "YES" if eye_contact else "NO",
            "Head Orientation": head_orientation,
            "Body Movement": status,
            "Additional Person": "YES" if extra_person else "NO",
            "Blink Status": "YES" if blink_status else "NO",
            "Blink Rate": str(blink_rate),
            "Face Distance": face_distance,
            "Yaw (°)": f"{yaw:.1f}",
            "Pitch (°)": f"{pitch:.1f}",
            "Roll (°)": f"{roll:.1f}",
            "Motion Level": motion_level,
            "Landmark Confidence (%)": f"{landmark_confidence:.1f}",
            "Frame Rate": f"{self.fps:.1f}",
            "Lip Movement": "YES" if lip_status else "NO"
        }

        # Overlay a simple border for gaze
        color = (0,255,0) if eye_contact else (0,0,255)
        cv2.rectangle(img, (0,0),(w,h), color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Motion detector
# ────────────────────────────────────────────────────────────────────────────────
class MotionDetector:
    def __init__(self, buf_len=5, thresh_percent=1):
        self.queue = deque(maxlen=buf_len)
        self.thresh_percent = thresh_percent

    def update(self, gray_frame):
        self.queue.append(gray_frame)
        if len(self.queue) < self.queue.maxlen:
            return "STATIC", 0.0
        diff = cv2.absdiff(self.queue[0], gray_frame)
        non_zero = cv2.countNonZero(diff)
        total_px = gray_frame.shape[0] * gray_frame.shape[1]
        intensity = (non_zero / total_px) * 100  # percent changed
        status = "MOVING" if intensity > self.thresh_percent else "STATIC"
        return status, intensity

# ────────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config("Attention Monitor", layout="wide")
st.title("Real-Time Attention & Presence Monitor")

st.info("Please grant camera access ⬆️ (your browser will prompt).")

face_mesh, segmentor = _init_models()
motion_detector = MotionDetector()

# Auto-refresh every second for metrics updates
st_autorefresh(interval=1000, key="refresh")

col_vid, col_metrics = st.columns([3,1], gap="large")

with col_vid:
    webrtc_ctx = webrtc_streamer(
        key="monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=AttentionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_metrics:
    st.header("🔍 Live Metrics")
    placeholder = st.empty()
    if webrtc_ctx and webrtc_ctx.video_processor and webrtc_ctx.video_processor.metrics:
        placeholder.table({
            "Metric": list(webrtc_ctx.video_processor.metrics.keys()),
            "Value": list(webrtc_ctx.video_processor.metrics.values())
        })