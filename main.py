import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import time
import numpy as np
import cv2
from tools.annotation import Annotator
from tools.normalization import FaceNormalizer
from tools.recognition import FaceIdentity, recognize
from tools.utils import show_images, show_faces
from tools.detection import FaceDetector


# TODOs:
# - FaceIdentity is decklared twice!!! Fix that!
# - Add toggle to switch input on and off
# - Maybe make banner captions of boundingbox smaller and dynamic
# - Fix Bug, when getting to second row for faces! 
# - Check what happens, if image has no face for gallery


MEDIA_STREAM_CONSTRAINTS = {
        "video": {
            "width": {"min": 1920, "ideal": 1920, "max": 1920},
        },
        "audio": False,
    }

class SideBar:
    def __init__(self):
        st.sidebar.markdown("## Preferences")

        st.sidebar.markdown("---")
        st.sidebar.markdown("## Face Detection")
        self.max_faces = st.sidebar.number_input(
            "Maximum Number of Faces", value=2, min_value=1
        )
        self.detection_confidence = st.sidebar.slider(
            "Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5
        )
        self.tracking_confidence = st.sidebar.slider(
            "Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9
        )
        self.margin = st.sidebar.slider("Bounding box margin", 0, 100, 25, 1)
        self.scale_detect = st.sidebar.slider(
            "Scale", min_value=0.0, max_value=1.0, value=0.5, key="b"
        )
        st.sidebar.markdown("---")

        st.sidebar.markdown("## Face Recognition")
        self.similarity_threshold = st.sidebar.slider(
            "Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67
        )
        self.scale_identify = st.sidebar.slider(
            "Scale", min_value=0.0, max_value=1.0, value=1.0, key="a"
        )
        self.model_name = st.sidebar.selectbox(
            "Model",
            ["MobileNet", "ResNet"],
            index=0,
        )
        st.sidebar.markdown("---")

        st.sidebar.markdown("## Gallery")
        self.uploaded_files = st.sidebar.file_uploader(
            "Choose multiple images to upload", accept_multiple_files=True
        )

        self.gallery_images = []
        self.gallery_identities = []
        self.gallery_names = []

        st.sidebar.markdown("**Gallery Faces**")
        if self.uploaded_files is not None:
            self.gallery_names = []
            for file in self.uploaded_files:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.cvtColor(
                    cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
                self.gallery_names.append(file.name)
                gallery_face_detector = FaceDetector(
                    image.shape,
                    max_faces=1,
                    detection_confidence=0.3,
                    tracking_confidence=0.3,
                )
                gallery_face_normalizer = FaceNormalizer(target_size=(112, 112))

                detections = gallery_face_detector.detect_faces(image, 1, None)
                faces = gallery_face_normalizer.face_cropper(image, detections)
                self.gallery_images.append(faces[0])
            gallery_face_identity = FaceIdentity(model=self.model_name)
            self.gallery_identities = gallery_face_identity.extract(
                self.gallery_images, scale=1.0
            )

            show_images(self.gallery_images, self.gallery_names, 3)

            self.gallery_names = [
                name.split(".jpg")[0].split(".png")[0].split(".jpeg")[0]
                for name in self.gallery_names
            ]

        st.sidebar.markdown("---")


class KPI:
    def __init__(self):
        self.kpi_texts = []
        st.markdown("---")
        kpi_names_row1 = [
            "**FrameRate**",
            "**Detected Faces**",
            "**Image Dims**",
            "**Image Resized Dims**",
        ]

        for kpi, name in zip(st.columns(4), kpi_names_row1):
            with kpi:
                st.markdown(name)
                self.kpi_texts.append(st.markdown("-"))
        st.markdown("---")

        kpi_names_row2 = [
            "**Detection [ms]**",
            "**Normalization [ms]**",
            "**Recognition [ms]**",
            "**Annotations [ms]**",
        ]

        for kpi, name in zip(st.columns(4), kpi_names_row2):
            with kpi:
                st.markdown(name)
                self.kpi_texts.append(st.markdown("-"))
        st.markdown("---")

    def update_kpi(self, kpi_values):
        for kpi_text, kpi_value in zip(self.kpi_texts, kpi_values):
            kpi_text.write(
                f"<h1 style='text-align: center; color: red;'>{kpi_value:.2f}</h1>"
                if isinstance(kpi_value, float)
                else f"<h1 style='text-align: center; color: red;'>{kpi_value}</h1>",  # if type(kpi_value) == float else kpi_value
                unsafe_allow_html=True,
            )


# Streamlit app
st.title("FaceID App Demonstration")

# Instantiate WebRTC (and show start button)
ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    video_receiver_size=4,
    async_processing=True,
)

# Shape of WebRTC frame
if ctx.video_receiver:
    shape = ctx.video_receiver.get_frame().to_ndarray(format="rgb24").shape[:2]
else:
    shape = ("-", "-")

# Sidebar
sidebar = SideBar()

# FaceDetector
face_detector = FaceDetector(
    shape,
    max_faces=sidebar.max_faces,
    detection_confidence=sidebar.detection_confidence,
    tracking_confidence=sidebar.tracking_confidence,
)

# Face Normalizer
face_normalizer = FaceNormalizer(target_size=(112, 112))

# FaceRecognition
face_identity = FaceIdentity(model=sidebar.model_name)

# Annotator
annotator = Annotator(shape=shape)

# Live Stream Display
st.markdown("**Showcase**")
image_loc = st.empty()
st.markdown("---")

# KPI Section
kpi = KPI()

# Display Detected Faces
st.markdown("**Detected Faces**")
face_window = st.empty()
st.markdown("---")

# Display Input for Recognition
st.markdown("**Input**")
process_window = st.empty()
st.markdown("---")


if ctx.video_receiver:
    prevTime = 0
    while True:
        start = time.time()
        try:
            frame = ctx.video_receiver.get_frame(timeout=1)
            img = frame.to_ndarray(format="rgb24")
        except:
            continue

        # FACE DETECTION ---------------------------------------------------------
        # Stop the time for the following operations
        start_time = time.time()
        detections = face_detector.detect_faces(
            img, sidebar.scale_detect, display=process_window
        )
        end_time = time.time()
        time_face_detection = (end_time - start_time) * 1000

        # FACE NORMALIZATION ------------------------------------------------------
        start_time = time.time()
        faces = face_normalizer.face_cropper(img, detections)
        end_time = time.time()
        time_face_normalization = (end_time - start_time) * 1000

        # FACE RECOGNITION --------------------------------------------------------
        start_time = time.time()
        identities = face_identity.extract(faces, sidebar.scale_identify)
        recognitions = recognize(
            identities,
            sidebar.gallery_identities,
            sidebar.gallery_names,
            sidebar.gallery_images,
            sidebar.similarity_threshold,
        )
        end_time = time.time()
        time_recognition = (end_time - start_time) * 1000

        # ANNOTATIONS ------------------------------------------------------------
        start_time = time.time()
        img.flags.writeable = True  # make them faster
        img = annotator.draw_mesh(img, detections[0])
        img = annotator.draw_landmarks(img, detections[0])
        img = annotator.draw_bounding_box(img, detections[0], sidebar.margin)
        img = annotator.draw_text(img, detections[0], recognitions[0], sidebar.margin)
        end_time = time.time()
        time_annotations = (end_time - start_time) * 1000

        # DISPLAY LIVE FRAME ------------------------------------------------------
        image_loc.image(img, channels="RGB", caption="Output", use_column_width=True)

        # DISPLAY FACES -----------------------------------------------------------
        show_faces(
            faces,
            *recognitions,
            3,
            scale=sidebar.scale_identify,
            channels="RGB",
            display=face_window,
        )

        # CALCULATE FPS ----------------------------------------------------------
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # UPDATE KPI -------------------------------------------------------------
        kpi.update_kpi(
            [
                fps,
                len(faces),
                shape,
                tuple(int(e * sidebar.scale_detect) for e in shape),
                time_face_detection,
                time_face_normalization,
                time_recognition,
                time_annotations,
            ]
        )
