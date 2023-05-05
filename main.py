import streamlit as st
import streamlit_toggle as tog
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import time
import numpy as np
import cv2
from tools.annotation import Annotator
from tools.normalization import FaceNormalizer
from tools.recognition import FaceIdentity, recognize
from tools.utils import show_images, show_faces, rgb
from tools.detection import FaceDetector

MAX_WEBCAM_WIDTH = 1920
# TODOs:
# - FaceIdentity is decklared twice!!! Fix that!
# - Add toggle to switch input on and off
# - Maybe make banner captions of boundingbox smaller and dynamic
# - Fix Bug, when getting to second row for faces!
# - Check what happens, if image has no face for gallery

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class SideBar:
    def __init__(self):
        with st.sidebar:
            st.markdown("# Preferences")
            st.markdown("---")

            st.markdown("## Webcam")
            self.resolution = st.selectbox(
                "Webcam Resolution",
                [(1920, 1080), (1280, 720), (640, 360)],
                index=1,
            )

            st.markdown("---")
            st.markdown("## Face Detection")
            self.max_faces = st.number_input(
                "Maximum Number of Faces", value=2, min_value=1
            )
            self.detection_confidence = st.slider(
                "Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5
            )
            self.tracking_confidence = st.slider(
                "Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9
            )
            self.margin = st.slider("Bounding box margin", 0, 100, 25, 1)
            self.scale_detect = st.slider(
                "Scale", min_value=0.0, max_value=1.0, value=0.5, key="b"
            )

            self.on_bounding_box = tog.st_toggle_switch(
                "Show Bounding Box", key="show_bounding_box", default_value=True, active_color=rgb(255, 75, 75), track_color=rgb(50, 50, 50) 
            )
            self.on_five_landmarks = tog.st_toggle_switch(
                "Show Five Landmarks", key="show_five_landmarks", default_value=True, active_color=rgb(255, 75, 75),
                track_color=rgb(50, 50, 50) 
            )
            self.on_mesh = tog.st_toggle_switch(
                "Show Mesh", key="show_mesh", default_value=True, active_color=rgb(255, 75, 75),
                track_color=rgb(50, 50, 50) 
            )

            st.markdown("---")

            st.markdown("## Face Recognition")
            self.similarity_threshold = st.slider(
                "Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67
            )
            self.scale_identify = st.slider(
                "Scale", min_value=0.0, max_value=1.0, value=1.0, key="a"
            )
            self.model_name = st.selectbox(
                "Model",
                ["MobileNet", "ResNet"],
                index=0,
            )
            st.markdown("---")

            st.markdown("## Gallery")
            self.uploaded_files = st.file_uploader(
                "Choose multiple images to upload", accept_multiple_files=True
            )

            self.gallery_images = []
            self.gallery_identities = []
            self.gallery_names = []

            st.markdown("**Gallery Faces**")
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

            st.markdown("---")


class KPI:
    def __init__(self):
        self.kpi_texts = []
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

    def update_kpi(self, kpi_values):
        for kpi_text, kpi_value in zip(self.kpi_texts, kpi_values):
            kpi_text.write(
                f"<h1 style='text-align: center; color: red;'>{kpi_value:.2f}</h1>"
                if isinstance(kpi_value, float)
                else f"<h1 style='text-align: center; color: red;'>{kpi_value}</h1>",  # if type(kpi_value) == float else kpi_value
                unsafe_allow_html=True,
            )


# Streamlit app
st.set_page_config(layout="wide")

st.title("FaceID App Demonstration")

# Sidebar
sb = SideBar()

# Instantiate WebRTC (and show start button)
ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {
                "min": MAX_WEBCAM_WIDTH,
                "ideal": MAX_WEBCAM_WIDTH,
                "max": MAX_WEBCAM_WIDTH,
            },
        },
        "audio": False,
    },
    video_receiver_size=4,
    async_processing=True,
)

# Shape of WebRTC frame
shape = sb.resolution

# FaceDetector
face_detector = FaceDetector(
    shape,
    max_faces=sb.max_faces,
    detection_confidence=sb.detection_confidence,
    tracking_confidence=sb.tracking_confidence,
)

# Face Normalizer
face_normalizer = FaceNormalizer(target_size=(112, 112))

# FaceRecognition
face_identity = FaceIdentity(model=sb.model_name)

# Annotator
annotator = Annotator()

# Live Stream Display
image_loc = st.empty()
st.markdown("---")

# KPI Section
st.markdown("**Stats**")
kpi = KPI()
st.markdown("---")

# Display Detected Faces
st.markdown("**Detected Faces**")
face_window = st.empty()
st.markdown("---")

# Display Input for Recognition
process_window = st.empty()
st.markdown("---")


if ctx.video_receiver:
    prevTime = 0
    while True:
        start = time.time()
        try:
            frame = ctx.video_receiver.get_frame(timeout=1)
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, shape)
        except:
            continue

        # FACE DETECTION ---------------------------------------------------------
        # Stop the time for the following operations
        start_time = time.time()
        detections = face_detector.detect_faces(
            img, sb.scale_detect, display=process_window
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
        identities = face_identity.extract(faces, sb.scale_identify)
        recognitions = recognize(
            identities,
            sb.gallery_identities,
            sb.gallery_names,
            sb.gallery_images,
            sb.similarity_threshold,
        )
        end_time = time.time()
        time_recognition = (end_time - start_time) * 1000

        # ANNOTATIONS ------------------------------------------------------------
        start_time = time.time()
        img.flags.writeable = True  # make them faster
        if sb.on_mesh:
            img = annotator.draw_mesh(img, detections[0])
        if sb.on_five_landmarks:
            img = annotator.draw_landmarks(img, detections[0])
        if sb.on_bounding_box:
            img = annotator.draw_bounding_box(
                img, detections[0], recognitions[0], sb.margin
            )
        img = annotator.draw_text(img, detections[0], recognitions[0], sb.margin)
        end_time = time.time()
        time_annotations = (end_time - start_time) * 1000

        # DISPLAY LIVE FRAME ------------------------------------------------------
        image_loc.image(
            img, channels="RGB", caption="Live-Stream", use_column_width=True
        )

        # DISPLAY FACES -----------------------------------------------------------
        show_faces(
            faces,
            *recognitions,
            3,
            scale=sb.scale_identify,
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
                tuple(int(e * sb.scale_detect) for e in shape),
                time_face_detection,
                time_face_normalization,
                time_recognition,
                time_annotations,
            ]
        )
