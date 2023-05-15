import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import mediapipe as mp
import tflite_runtime.interpreter as tflite
import av
import queue
import pandas as pd
from nametypes import Stats, Timings, Detection
from pathlib import Path
from utils import get_ice_servers, download_file
from face_recognition import detect_faces, align_faces, inference, draw_detections, recognize_faces, process_gallery

# Set logging level to error (To avoid getting spammed by queue warnings etc.)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

ROOT = Path(__file__).parent

MODEL_URL = "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/mobileNet.tflite"
MODEL_LOCAL_PATH = ROOT /"./models/mobileNet.tflite"

DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
MAX_FACES = 2

# Set page layout for streamlit to wide
st.set_page_config(layout="wide")

download_file(MODEL_URL, MODEL_LOCAL_PATH, file_hash="6c19b789f661caa8da735566490bfd8895beffb2a1ec97a56b126f0539991aa6")

# Session-specific caching of the face recognition model
cache_key = "face_id_model"
if cache_key in st.session_state:
    face_recognition_model = st.session_state[cache_key]
else:
    face_recognition_model = tflite.Interpreter(model_path=MODEL_LOCAL_PATH.as_posix())
    st.session_state[cache_key] = face_recognition_model

# Session-specific caching of the face detection model
cache_key = "face_detection_model"
if cache_key in st.session_state:
    face_detection_model = st.session_state[cache_key]
else:
    face_detection_model = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
            max_num_faces=MAX_FACES,
        )
    st.session_state[cache_key] = face_detection_model

stats_queue: "queue.Queue[Stats]" = queue.Queue()
timings_queue: "queue.Queue[Timings]" = queue.Queue()
detections_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame_start = time.time()

    # Convert frame to numpy array
    frame = frame.to_ndarray(format="rgb24")

    # Get frame resolution
    resolution = frame.shape

    start = time.time()
    detections = detect_faces(frame, face_detection_model)
    time_detection = (time.time() - start) * 1000

    start = time.time()
    detections = align_faces(frame, detections)
    time_normalization = (time.time() - start) * 1000

    start = time.time()
    detections = inference(detections, face_recognition_model)
    time_inference = (time.time() - start) * 1000

    start = time.time()
    detections = recognize_faces(detections, gallery)
    time_recognition = (time.time() - start) * 1000

    start = time.time()
    frame = draw_detections(frame, detections)
    time_drawing = (time.time() - start) * 1000

    # Convert frame back to av.VideoFrame
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

    # Put detections, stats and timings into queues (to be accessible by other thread)
    detections_queue.put(detections)
    timings_queue.put(Timings(detection=time_detection,
                              normalization=time_normalization,
                              inference=time_inference, 
                              recognition=time_recognition,
                              drawing=time_drawing))
    stats_queue.put(Stats(fps=1 / (time.time() - frame_start),
                          resolution=resolution,
                          num_faces=len(detections)))
    
    return frame


# Streamlit app
st.title("FaceID App Demonstration")

width = 640

st.sidebar.markdown("**Gallery**")
gallery = st.sidebar.file_uploader("Upload images to gallery", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if gallery:
    gallery = process_gallery(gallery, face_detection_model, face_recognition_model)
    st.sidebar.markdown("**Gallery Images**")
    for identity in gallery:
        st.sidebar.image(identity.image, caption=identity.name, width=112) #  TODO formatting

ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers("twilio")},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {
                "width": {
                    "min": width,
                    "ideal": width,
                    "max": width,
                }}, "audio": False},
    async_processing=True,
)

st.markdown("**Stats**")
stats = st.empty()

st.markdown("**Timings [ms]**")
timings = st.empty()

st.markdown("**Detections**")
detections = st.empty()

# Display Live Stats
if ctx.state.playing:
    while True:
        stats.dataframe(pd.DataFrame([stats_queue.get()]).style.format({"fps": "{:.2f}"}))
        timings.dataframe(pd.DataFrame([timings_queue.get()]).style.format("{:.2f}"))
        detections.dataframe(pd.DataFrame(detections_queue.get()).drop(columns=["face"]))  # TODO formatting
        time.sleep(1)
