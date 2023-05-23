import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import av
import numpy as np
import os
import cv2
import queue
from streamlit_toggle import st_toggle_switch
import pandas as pd
from tools.nametypes import Stats, Detection, Identity, Match
from tools.utils import get_ice_servers, rgb, format_dflist
from tools.face_detection import FaceDetection
from tools.face_recognition import FaceRecognition
from tools.annotation import Annotation
from tools.gallery import init_gallery
from st_aggrid import AgGrid


# Set logging level to error (To avoid getting spammed by queue warnings etc.)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


# Set page layout for streamlit to wide
st.set_page_config(
    layout="wide", 
    page_title="FaceID App Demo", page_icon=":sunglasses:"
)
with st.sidebar:
    st.markdown("# Settings")
    face_rec_on = st_toggle_switch(
        "Live Face Recognition",
        key="activate_face_rec",
        default_value=True,
        active_color=rgb(255, 75, 75),
        track_color=rgb(50, 50, 50),
        label_after=True,
    )

    with st.expander("Advanced Settings", expanded=False):
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
        st.markdown("---")
        st.markdown("## Face Detection")
        detection_min_face_size = st.slider(
            "Min Face Size", min_value=5, max_value=120, value=40
        )
        detection_scale_factor = st.slider(
            "Scale Factor", min_value=0.1, max_value=1.0, value=0.7
        )
        detection_confidence = st.slider(
            "Min Detection Confidence", min_value=0.5, max_value=1.0, value=0.9
        )
        st.markdown("---")
        st.markdown("## Face Recognition")
        similarity_threshold = st.slider(
            "Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67
        )
        st.markdown(
            "This sets a maximum distance for the cosine similarity between the embeddings of the detected face and the gallery images. If the distance is below the threshold, the face is recognized as the gallery image with the lowest distance. If the distance is above the threshold, the face is not recognized."
        )

    st.markdown("# Face Gallery")
    files = st.sidebar.file_uploader(
        "Upload images to gallery",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    with st.expander("Uploaded Images", expanded=True):
        st.image(files, width=112, caption=files)


gallery = init_gallery(
    files, min_detections_conf=detection_confidence, min_similarity=similarity_threshold
)

face_detector = FaceDetection(
    min_detections_conf=detection_confidence,
    min_face_size=detection_min_face_size,
    scale_factor=detection_scale_factor,
)
face_recognizer = FaceRecognition(min_similarity=similarity_threshold)
annotator = Annotation()

stats_queue: "queue.Queue[Stats]" = queue.Queue()
detections_queue: "queue.Queue[List[Detection]]" = queue.Queue()
identities_queue: "queue.Queue[List[Identity]]" = queue.Queue()
matches_queue: "queue.Queue[List[Match]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Initialize detections
    detections, identities, matches = [], [], []

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
        frame, detections = face_detector(frame)
        stats = stats._replace(num_faces=len(detections) if detections else 0)
        stats = stats._replace(detection=(time.time() - start) * 1000)

        # Run face recognition
        start = time.time()
        identities = face_recognizer(frame, detections)
        stats = stats._replace(recognition=(time.time() - start) * 1000)

        # Do matching
        start = time.time()
        matches = face_recognizer.find_matches(identities, gallery)
        stats = stats._replace(matching=(time.time() - start) * 1000)

        # Draw annotations
        start = time.time()
        frame = annotator(frame, detections, identities, matches)
        stats = stats._replace(annotation=(time.time() - start) * 1000)

    # Convert frame back to av.VideoFrame
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

    # Calculate FPS and add to stats
    stats = stats._replace(fps=1 / (time.time() - frame_start))

    # Send data to other thread
    detections_queue.put_nowait(detections)
    identities_queue.put_nowait(identities)
    matches_queue.put_nowait(matches)
    stats_queue.put_nowait(stats)

    return frame


# Streamlit app
st.title("Live Webcam Face Recognition")

st.markdown("**Stats**")
disp_stats = st.empty()

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

# Display Detections and Identities
st.markdown("**Detections**")
disp_detections = st.empty()

# Display Gallery and Detection Identities
col_identities_gal, col_identities_det = st.columns(2)
col_identities_gal.markdown("**Gallery Identities**")
disp_identities_gal = col_identities_gal.empty()
col_identities_det.markdown("**Detection Identities**")
disp_identities_det = col_identities_det.empty()

# Diplay Matched Faces Metrics
col_matches, col_match_metrics = st.columns(2)
col_matches.markdown("**Matches**")
disp_matches = col_matches.empty()
col_match_metrics.markdown("**Match Metrics**")
disp_match_metrics = col_match_metrics.empty()

# Display gallery identities
if gallery:
    disp_identities_gal.image(
        image=[identity.face_aligned for identity in gallery],
        caption=[identity.name for identity in gallery],
    )
else:
    disp_identities_gal.empty()

# Display Live Stats
if ctx.state.playing:
    while True:
        # Get stats, format and displayw
        stats_data = stats_queue.get()
        stats_dataframe = pd.DataFrame([stats_data]).applymap(
            lambda x: (format_dflist(x))
        )
        disp_stats.dataframe(stats_dataframe, use_container_width=True)

        # Get detections, format and display
        detections_data = detections_queue.get()
        detections_dataframe = pd.DataFrame(detections_data).applymap(
            lambda x: (format_dflist(x))
        )
        if detections_data:
            disp_detections.dataframe(detections_dataframe, use_container_width=True)
        else:
            disp_detections.empty()

        # Display detection identities
        identities_data = identities_queue.get()
        if identities_data:
            disp_identities_det.image(
                image=[identity.face_aligned for identity in identities_data]
            )
        else:
            disp_identities_det.empty()

        # Display matches and match metrics
        matches_data = matches_queue.get()
        if matches_data:
            disp_matches.image(
                image=[match.faces_aligned for match in matches_data],
                caption=[match.name for match in matches_data],
            )
            match_metrics_dataframe = (
                pd.DataFrame(matches_data)
                .drop(columns=["faces", "faces_aligned"])
                .applymap(lambda x: (format_dflist(x)))
            )
            disp_match_metrics.dataframe(match_metrics_dataframe, use_container_width=True)
        else:
            disp_matches.empty()
            disp_match_metrics.empty()
