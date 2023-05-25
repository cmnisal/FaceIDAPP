import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import av
import queue
from streamlit_toggle import st_toggle_switch
import pandas as pd
from tools.nametypes import Stats, Detection, Identity, Match
from tools.utils import get_ice_servers, rgb, format_dflist
from tools.face_detection import FaceDetection
from tools.face_recognition import FaceRecognition
from tools.annotation import Annotation
from tools.gallery import init_gallery
from tools.pca import pca


# Set logging level to error (To avoid getting spammed by queue warnings etc.)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


# Set page layout for streamlit to wide
st.set_page_config(layout="wide", page_title="FaceID App Demo", page_icon=":sunglasses:")
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

        ice_server = st.selectbox("ICE Server", ["twilio", "metered"], index=1)
        st.markdown(
            "Note: metered is a free server with limited bandwidth, and can take a while to connect. Twilio is a paid service and is payed by me, so please don't abuse it."
        )
        st.markdown("---")
        st.markdown("## Face Detection")
        detection_min_face_size = st.slider("Min Face Size", min_value=5, max_value=120, value=40)
        detection_scale_factor = st.slider("Scale Factor", min_value=0.1, max_value=1.0, value=0.7)
        detection_confidence = st.slider("Min Detection Confidence", min_value=0.5, max_value=1.0, value=0.9)
        st.markdown("---")
        st.markdown("## Face Recognition")
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67)
        st.markdown(
            "This sets a maximum distance for the cosine similarity between the embeddings of the detected face and the gallery images. If the distance is below the threshold, the face is recognized as the gallery image with the lowest distance. If the distance is above the threshold, the face is not recognized."
        )
        model_name = st.selectbox("Model", ["MobileNetV2", "ResNet50", "ArcFaceOctupletLoss", "FaceTransformerOctupletLoss"], index=1)
        st.markdown(
            "Note: The mobileNet model is smaller and faster, but less accurate. The resNet model is bigger and slower, but more accurate."
        )

    st.markdown("# Face Gallery")
    files = st.sidebar.file_uploader(
        "Upload images to gallery",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    with st.expander("Uploaded Images", expanded=True):
        if files:
            st.image(files, width=112, caption=files)
        else:
            st.info("No images uploaded yet.")


gallery = init_gallery(
    files,
    min_detections_conf=detection_confidence,
    min_similarity=similarity_threshold,
    model_name=model_name,
)

face_detector = FaceDetection(
    min_detections_conf=detection_confidence,
    min_face_size=detection_min_face_size,
    scale_factor=detection_scale_factor,
)
face_recognizer = FaceRecognition(model_name=model_name, min_similarity=similarity_threshold)
annotator = Annotation()

transfer_queue: "queue.Queue[Stats, List[Detection], List[Identity], List[Match]]" = queue.Queue()


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
        frame = annotator(frame, detections, identities, matches, gallery)
        stats = stats._replace(annotation=(time.time() - start) * 1000)

    # Convert frame back to av.VideoFrame
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

    # Calculate FPS and add to stats
    stats = stats._replace(fps=1 / (time.time() - frame_start))

    # Send data to other thread
    transfer_queue.put_nowait([stats, detections, identities, matches])

    return frame


# Streamlit app
st.title("Live Webcam Face Recognition")

st.markdown("**Stream Stats**")
disp_stats = st.info("No streaming statistics yet, please start the stream.")

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

tab_recognition, tab_metrics, tab_pca = st.tabs(["Recognized Identities", "Recognition Metrics", "Live PCAs"])


with tab_recognition:
    # Display Gallery and Recognized Identities
    col1, col2 = st.columns(2)
    col1.markdown("**Gallery Identities**")
    disp_identities_gal = col1.info("No gallery images uploaded yet ...")
    col2.markdown("**Recognized Identities**")
    disp_identities_rec = col2.info("No recognized identities yet ...")

with tab_metrics:
    # Display Detections and Identities
    st.markdown("**Detection Metrics**")
    disp_detection_metrics = st.info("No detected faces yet ...")

    # Display Recognition Metrics
    st.markdown("**Recognition Metrics**")
    disp_recognition_metrics = st.info("No recognized identities yet ...")

with tab_pca:
    # Display 2D and 3D PCA
    col1, col2 = st.columns(2)
    col1.markdown("**PCA 2D**")
    disp_pca3d = col1.info("Only available if more than 1 recognized face ...")
    col2.markdown("**PCA 3D**")
    disp_pca2d = col2.info("Only available if more than 1 recognized face ...")
    freeze_pcas = st.button("Freeze PCAs for Interaction", key="reset_pca")

    # Show PCAs
    if freeze_pcas and gallery:
        col1, col2 = st.columns(2)
        if len(st.session_state.matches) > 1:
            col1.plotly_chart(
                pca(
                    st.session_state.matches,
                    st.session_state.identities,
                    gallery,
                    dim=3,
                ),
                use_container_width=True,
            )
            col2.plotly_chart(
                pca(
                    st.session_state.matches,
                    st.session_state.identities,
                    gallery,
                    dim=2,
                ),
                use_container_width=True,
            )


# Show Gallery Identities
if gallery:
    disp_identities_gal.image(
        image=[identity.face_aligned for identity in gallery],
        caption=[match.name for match in gallery],
    )
else:
    disp_identities_gal.info("No gallery images uploaded yet ...")


# Display Live Stats
if ctx.state.playing:
    while True:
        # Retrieve data from other thread
        stats, detections, identities, matches = transfer_queue.get()

        # Save for PCA Snapshot
        st.session_state.identities = identities
        st.session_state.matches = matches

        # Show Stats
        disp_stats.dataframe(
            pd.DataFrame([stats]).applymap(lambda x: (format_dflist(x))),
            use_container_width=True,
        )

        # Show Detections Metrics
        if detections:
            disp_detection_metrics.dataframe(
                pd.DataFrame(detections).applymap(lambda x: (format_dflist(x))),
                use_container_width=True,
            )
        else:
            disp_detection_metrics.info("No detected faces yet ...")

        # Show Match Metrics
        if matches:
            disp_recognition_metrics.dataframe(
                pd.DataFrame(matches).applymap(lambda x: (format_dflist(x))),
                use_container_width=True,
            )
        else:
            disp_recognition_metrics.info("No recognized identities yet ...")

        if len(matches) > 1:
            disp_pca3d.plotly_chart(pca(matches, identities, gallery, dim=3), use_container_width=True)
            disp_pca2d.plotly_chart(pca(matches, identities, gallery, dim=2), use_container_width=True)
        else:
            disp_pca3d.info("Only available if more than 1 recognized face ...")
            disp_pca2d.info("Only available if more than 1 recognized face ...")

        # Show Recognized Identities
        if matches:
            disp_identities_rec.image(
                image=[identities[match.identity_idx].face_aligned for match in matches],
                caption=[gallery[match.gallery_idx].name for match in matches],
            )
        else:
            disp_identities_rec.info("No recognized identities yet ...")
else:
    st.info("Starting stream can take a while - if it's not working, switch to twilio servers for streaming in 'Advanced Settings' ...")

# BUG Recognized Identity Image is not updating on cloud version? (works on local!!!)
