import streamlit as st
import streamlit_toggle as tog
import time
import numpy as np
import cv2
from tools.annotation import draw_mesh, draw_landmarks, draw_bounding_box, draw_text
from tools.alignment import align_faces
from tools.identification import load_identification_model, inference, identify
from tools.utils import show_images, show_faces, rgb
from tools.detection import load_detection_model, detect_faces
from tools.webcam import init_webcam


# Set page layout for streamlit to wide
st.set_page_config(layout="wide")

# Gallery Processing
@st.cache_data
def gallery_processing(gallery_files):
    """Process the gallery images (Complete Face Recognition Pipeline)

    Args:
        gallery_files (_type_): Files uploaded by the user

    Returns:
        _type_: Gallery Images, Gallery Embeddings, Gallery Names
    """
    gallery_images, gallery_embs, gallery_names = [], [], []
    if gallery_files is not None:
        for file in gallery_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gallery_names.append(file.name.split(".jpg")[0].split(".png")[0].split(".jpeg")[0])
            detections = detect_faces(img, detection_model)
            aligned_faces = align_faces(img, np.asarray([detections[0]]))
            gallery_images.append(aligned_faces[0])
            gallery_embs.append(inference(aligned_faces, identification_model)[0])
    return gallery_images, gallery_embs, gallery_names


class SideBar:
    """A class to handle the sidebar
    """
    def __init__(self):
        with st.sidebar:
            st.markdown("# Preferences")
            self.on_face_recognition = tog.st_toggle_switch(
                "Face Recognition", key="activate_face_rec", default_value=True, active_color=rgb(255, 75, 75), track_color=rgb(50, 50, 50) 
            )

            st.markdown("---")

            st.markdown("## Webcam")
            self.resolution = st.selectbox("Webcam Resolution", [(1920, 1080), (1280, 720), (640, 360)], index=2)
            st.markdown("To change webcam resolution: Please refresh page and select resolution before starting webcam stream.")

            st.markdown("---")
            st.markdown("## Face Detection")
            self.max_faces = st.number_input("Maximum Number of Faces", value=2, min_value=1)
            self.detection_confidence = st.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
            self.tracking_confidence = st.slider("Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9)
            switch1, switch2 = st.columns(2)
            with switch1:
                self.on_bounding_box = tog.st_toggle_switch(
                    "Show Bounding Box", key="show_bounding_box", default_value=True, active_color=rgb(255, 75, 75), track_color=rgb(50, 50, 50) 
                )
            with switch2:
                self.on_five_landmarks = tog.st_toggle_switch(
                    "Show Five Landmarks", key="show_five_landmarks", default_value=True, active_color=rgb(255, 75, 75),
                    track_color=rgb(50, 50, 50) 
                )
            switch3, switch4 = st.columns(2)
            with switch3:
                self.on_mesh = tog.st_toggle_switch(
                    "Show Mesh", key="show_mesh", default_value=True, active_color=rgb(255, 75, 75),
                    track_color=rgb(50, 50, 50) 
                )
            with switch4:
                self.on_text = tog.st_toggle_switch(
                    "Show Text", key="show_text", default_value=True, active_color=rgb(255, 75, 75),
                    track_color=rgb(50, 50, 50) 
                )
            st.markdown("---")

            st.markdown("## Face Recognition")
            self.similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67)

            self.on_show_faces = tog.st_toggle_switch(
                "Show Recognized Faces", key="show_recognized_faces", default_value=True, active_color=rgb(255, 75, 75), track_color=rgb(50, 50, 50) 
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

            


class KPI:
    """Class for displaying KPIs in a row
    Args:
        keys (list): List of KPI names
    """
    def __init__(self, keys):
        self.kpi_texts = []
        row = st.columns(len(keys))
        for kpi, key in zip(row, keys):
            with kpi:
                item_row = st.columns(2)
                item_row[0].markdown(f"**{key}**:")
                self.kpi_texts.append(item_row[1].markdown("-"))

    def update_kpi(self, kpi_values):
        for kpi_text, kpi_value in zip(self.kpi_texts, kpi_values):
            kpi_text.write(
                f"<h5 style='text-align: center; color: red;'>{kpi_value:.2f}</h5>"
                if isinstance(kpi_value, float)
                else f"<h5 style='text-align: center; color: red;'>{kpi_value}</h5>",
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------------------------
# Streamlit App
st.title("FaceID App Demonstration")

# Sidebar
sb = SideBar()

# Initialize the Face Detection and Identification Models
detection_model = load_detection_model(max_faces=2, detection_confidence=0.5, tracking_confidence=0.9)
identification_model = load_identification_model(name="MobileNet")

# Gallery Processing
with st.sidebar():
    gallery_images, gallery_embs, gallery_names= gallery_processing(sb.uploaded_files)
    st.markdown("**Gallery Faces**")
    show_images(gallery_images, gallery_names, 3)
    st.markdown("---")

# Get Access to Webcam
webcam = init_webcam(width=sb.resolution[0])

# KPI Section
st.markdown("**Stats**")
kpi = KPI([
    "**FrameRate**",
    "**Detected Faces**",
    "**Image Dims**",
    "**Detection [ms]**",
    "**Normalization [ms]**",
    "**Inference [ms]**",
    "**Recognition [ms]**",
    "**Annotations [ms]**",
    "**Show Faces [ms]**",
])
st.markdown("---")

# Live Stream Display
stream_display = st.empty()
st.markdown("---")

# Display Detected Faces
st.markdown("**Detected Faces**")
face_window = st.empty()
st.markdown("---")


if webcam:
    prevTime = 0
    while True:
        # Init detections
        detections = []

        # Init times to "-" to show something if face recognition is turned off
        time_detection = "-"
        time_alignment = "-"
        time_inference = "-"
        time_identification = "-"
        time_annotations = "-"
        time_show_faces = "-"

        try:
            # Get Frame from Webcam
            frame = webcam.get_frame(timeout=1)

            # Convert to OpenCV Image
            frame = frame.to_ndarray(format="rgb24")
        except:
            continue
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # FACE RECOGNITION PIPELINE
        if sb.on_face_recognition:
            # FACE DETECTION ---------------------------------------------------------
            start_time = time.time()
            detections = detect_faces(frame, detection_model)
            time_detection = (time.time() - start_time) * 1000

            # FACE ALIGNMENT ---------------------------------------------------------
            start_time = time.time()
            aligned_faces = align_faces(frame, detections)
            time_alignment = (time.time() - start_time) * 1000

            # INFERENCE --------------------------------------------------------------
            start_time = time.time()
            if len(sb.gallery_embs) > 0: 
                faces_embs = inference(aligned_faces, identification_model)
            else:
                faces_embs = []
            time_inference = (time.time() - start_time) * 1000

            # FACE IDENTIFCATION -----------------------------------------------------
            start_time = time.time()
            if len(faces_embs) > 0 and len(sb.gallery_embs) > 0:
                ident_names, ident_dists, ident_imgs = identify(faces_embs, gallery_embs, gallery_names, gallery_images, thresh=sb.similarity_threshold)
            else:
                ident_names, ident_dists, ident_imgs = [], [], []
            time_identification = (time.time() - start_time) * 1000

            # ANNOTATIONS ------------------------------------------------------------
            start_time = time.time()
            frame = cv2.resize(frame, (1920, 1080)) # to make annotation in HD
            frame.flags.writeable = True  # (hack to make annotations faster)
            if sb.on_mesh:
                frame = draw_mesh(frame, detections)
            if sb.on_five_landmarks:
                frame = draw_landmarks(frame, detections)
            if sb.on_bounding_box:
                frame = draw_bounding_box(frame, detections, ident_names)
            if sb.on_text:
                frame = draw_text(frame, detections, ident_names)
            time_annotations = (time.time() - start_time) * 1000

            # DISPLAY DETECTED FACES -------------------------------------------------
            start_time = time.time()
            if sb.on_show_faces:
                show_faces(
                aligned_faces,
                ident_names,
                ident_dists,
                ident_imgs, 
                num_cols=3,
                channels="RGB",
                display=face_window,
            )
            time_show_faces = (time.time() - start_time) * 1000
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



        # DISPLAY THE LIVE STREAM --------------------------------------------------
        stream_display.image(frame, channels="RGB", caption="Live-Stream", use_column_width=True)

        # CALCULATE FPS -----------------------------------------------------------
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # UPDATE KPIS -------------------------------------------------------------
        kpi.update_kpi(
            [
                fps,
                len(detections),
                sb.resolution,
                time_detection,
                time_alignment,
                time_inference,
                time_identification,
                time_annotations,
                time_show_faces,
            ]
        )
