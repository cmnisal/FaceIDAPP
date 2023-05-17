import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import mediapipe as mp
import tflite_runtime.interpreter as tflite
import av
import numpy as np
import queue
from streamlit_toggle import st_toggle_switch
import pandas as pd
from tools.nametypes import Stats, Detection
from pathlib import Path
from tools.utils import get_ice_servers, download_file, display_match, rgb, format_dflist
from tools.face_recognition import (
    detect_faces,
    align_faces,
    inference,
    draw_detections,
    recognize_faces,
    process_gallery,
)

# Set logging level to error (To avoid getting spammed by queue warnings etc.)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

ROOT = Path(__file__).parent

MODEL_URL = (
    "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/mobileNet.tflite"
)
MODEL_LOCAL_PATH = ROOT / "./models/mobileNet.tflite"

DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
MAX_FACES = 2

# Set page layout for streamlit to wide
st.set_page_config(
    layout="wide", page_title="FaceID App Demo", page_icon=":sunglasses:"
)
with st.sidebar:
    st.markdown("# Preferences")
    face_rec_on = st_toggle_switch(
        "Face Recognition",
        key="activate_face_rec",
        default_value=True,
        active_color=rgb(255, 75, 75),
        track_color=rgb(50, 50, 50),
    )

    st.markdown("## Webcam & Stream")
    resolution = st.selectbox(
        "Webcam Resolution",
        [(1920, 1080), (1280, 720), (640, 360)],
        index=2,
    )
    st.markdown("Note: To change the resolution, you have to restart the stream.")

    ice_server = st.selectbox("ICE Server", ["twilio", "metered"], index=0)
    st.markdown(
        "Note: metered is a free server with limited bandwidth, and can take a while to connect. Twilio is a paid service and is payed by me, so please don't abuse it."
    )

    st.markdown("## Face Detection")
    max_faces = st.number_input("Maximum Number of Faces", value=2, min_value=1)
    detection_confidence = st.slider(
        "Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5
    )
    tracking_confidence = st.slider(
        "Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9
    )
    st.markdown("## Face Recognition")
    similarity_threshold = st.slider(
        "Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67
    )
    st.markdown(
        "This sets a maximum distance for the cosine similarity between the embeddings of the detected face and the gallery images. If the distance is below the threshold, the face is recognized as the gallery image with the lowest distance. If the distance is above the threshold, the face is not recognized."
    )

download_file(
    MODEL_URL,
    MODEL_LOCAL_PATH,
    file_hash="6c19b789f661caa8da735566490bfd8895beffb2a1ec97a56b126f0539991aa6",
)

# Session-specific caching of the face recognition model
cache_key = "face_id_model"
if cache_key in st.session_state:
    face_recognition_model = st.session_state[cache_key]
else:
    face_recognition_model = tflite.Interpreter(model_path=MODEL_LOCAL_PATH.as_posix())
    st.session_state[cache_key] = face_recognition_model

# Session-specific caching of the face recognition model
cache_key = "face_id_model_gal"
if cache_key in st.session_state:
    face_recognition_model_gal = st.session_state[cache_key]
else:
    face_recognition_model_gal = tflite.Interpreter(
        model_path=MODEL_LOCAL_PATH.as_posix()
    )
    st.session_state[cache_key] = face_recognition_model_gal

# Session-specific caching of the face detection model
cache_key = "face_detection_model"
if cache_key in st.session_state:
    face_detection_model = st.session_state[cache_key]
else:
    face_detection_model = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        max_num_faces=max_faces,
    )
    st.session_state[cache_key] = face_detection_model

stats_queue: "queue.Queue[Stats]" = queue.Queue()
detections_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Initialize detections
    detections = []

    # Initialize stats
    stats = Stats()

    # Start timer for FPS calculation
    frame_start = time.time()

    # Convert frame to numpy array
    frame = frame.to_ndarray(format="rgb24")

    # Get frame resolution and add to stats
    resolution = frame.shape
    stats = stats._replace(resolution=resolution)

    if face_rec_on:
        # Run face detection
        start = time.time()
        detections = detect_faces(frame, face_detection_model)
        stats = stats._replace(num_faces=len(detections) if detections else 0)
        stats = stats._replace(detection=(time.time() - start) * 1000)

        # Run face alignment
        start = time.time()
        detections = align_faces(frame, detections)
        stats = stats._replace(alignment=(time.time() - start) * 1000)

        # Run inference
        start = time.time()
        detections = inference(detections, face_recognition_model)
        stats = stats._replace(inference=(time.time() - start) * 1000)

        # Run face recognition
        start = time.time()
        detections = recognize_faces(detections, gallery, similarity_threshold)
        stats = stats._replace(recognition=(time.time() - start) * 1000)

        # Draw detections
        start = time.time()
        frame = draw_detections(frame, detections)
        stats = stats._replace(drawing=(time.time() - start) * 1000)

    # Convert frame back to av.VideoFrame
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

    # Calculate FPS and add to stats
    stats = stats._replace(fps=1 / (time.time() - frame_start))

    # Send data to other thread
    detections_queue.put(detections)
    stats_queue.put(stats)

    return frame


# Streamlit app
st.title("FaceID App Demonstration")

st.sidebar.markdown("**Gallery**")
gallery = st.sidebar.file_uploader(
    "Upload images to gallery", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)
if gallery:
    gallery = process_gallery(gallery, face_detection_model, face_recognition_model_gal)
    st.sidebar.markdown("**Gallery Images**")
    st.sidebar.image(
        [identity.image for identity in gallery],
        caption=[identity.name for identity in gallery],
        width=112,
    )

st.markdown("**Stats**")
stats = st.empty()

ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers(name=ice_server)},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {
                "min": resolution[0],
                "ideal": resolution[0],
                "max": resolution[0],
            },
            "height": {
                "min": resolution[1],
                "ideal": resolution[1],
                "max": resolution[1],
            },
        },
        "audio": False,
    },
    async_processing=True,
)

st.markdown("**Identified Faces**")
identified_faces = st.empty()

st.markdown("**Detections**")
detections = st.empty()

# Display Live Stats
if ctx.state.playing:
    while True:
        # Get stats
        stats_dataframe = pd.DataFrame([stats_queue.get()])

        # Write stats to streamlit
        stats.dataframe(stats_dataframe.style.format(thousands=" ", precision=2))

        # Get detections
        detections_data = detections_queue.get()
        detections_dataframe = (
            pd.DataFrame(detections_data)
            .drop(columns=["face", "face_match"], errors="ignore")
            .applymap(lambda x: (format_dflist(x)))
        )

        # Write detections to streamlit
        detections.dataframe(detections_dataframe)

        # Write identified faces to streamlit
        identified_faces.image(
            [display_match(d) for d in detections_data if d.name is not None],
            caption=[
                d.name + f"({d.distance:2f})"
                for d in detections_data
                if d.name is not None
            ],
            width=112,
        )
