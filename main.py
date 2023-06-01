import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import queue
import onnxruntime as rt
import pandas as pd
import threading
from tools.utils import format_dflist
import mediapipe as mp
import os
from twilio.rest import Client
import cv2
from skimage.transform import SimilarityTransform
from types import SimpleNamespace
from sklearn.metrics.pairwise import cosine_distances
from streamlit_profiler import Profiler


class Stats(SimpleNamespace):
    fps: float = None
    num_faces: int = 0
    detection: float = None
    recognition: float = None
    matching: float = None
    annotation: float = None


class Detection(SimpleNamespace):
    bbox: List[List[float]] = None
    landmarks: List[List[float]] = None


class Identity(SimpleNamespace):
    detection: Detection = Detection()
    name: str = None
    embedding: np.ndarray = None
    face: np.ndarray = None


class Match(SimpleNamespace):
    subject_id: Identity = Identity()
    gallery_id: Identity = Identity()
    distance: float = None
    name: str = None


# Get twilio ice server configuration using twilio credentials from environment variables (set in streamlit secrets)
# Ref: https://www.twilio.com/docs/stun-turn/api
# ICE_SERVERS = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]).tokens.create().ice_servers

# Set page layout for streamlit to wide
st.set_page_config(layout="wide", page_title="Live Face Recognition", page_icon=":sunglasses:")

# Sidebar with settings and face gallery
with st.sidebar:
    st.markdown("# Settings")
    st.markdown("---")
    st.markdown("## Face Detection")
    max_num_faces = st.number_input("Maximum Number of Faces", value=2, min_value=1)
    min_tracking_confidence = st.slider("Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9)
    min_detection_confidence = st.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.9)

    st.markdown("---")
    st.markdown("## Face Recognition")
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=2.0, value=0.67)
    st.markdown(
        "Maximum distance between the subject and the closest gallery face embeddings for considering them as a match."
    )

    st.markdown("---")
    st.markdown("# Face Gallery")
    files = st.sidebar.file_uploader(
        "Upload images to gallery",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

# Streamlit app
st.title("Live Webcam Face Recognition")

st.markdown("**Stream Stats**")
disp_stats = st.info("No streaming statistics yet, please start the stream.")

st.markdown("**Live Stream**")
disp_stream = st.container()
stream_window = st.empty()

col1, col2 = st.columns(2)
col1.markdown("**Gallery Faces**")
disp_gallery = col1.info("No gallery images uploaded yet ...")

col2.markdown("**Matches**")
disp_matches = col2.info("No matches found yet ...")


# Init face detector and face recognizer
face_recognizer = rt.InferenceSession("model.onnx", providers=rt.get_available_providers())
face_detector = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
    max_num_faces=max_num_faces,
)


def detect_faces(frame):
    # Process the frame with the face detector
    result = face_detector.process(frame)

    # Initialize an empty list to store the detected faces
    detections = []

    # Check if any faces were detected
    if result.multi_face_landmarks:
        # Iterate over each detected face
        for count, detection in enumerate(result.multi_face_landmarks):

            # Select 5 Landmarks
            five_landmarks = np.asarray(detection.landmark)[[470, 475, 1, 57, 287]]


            # Extract the x and y coordinates of the landmarks of interest
            landmarks = [[landmark.x * frame.shape[1], landmark.y * frame.shape[0]] for landmark in five_landmarks]

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [landmark.x * frame.shape[1] for landmark in detection.landmark]
            all_y_coords = [landmark.y * frame.shape[0] for landmark in detection.landmark]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

            # Create a Detection object for the face
            detection = Detection(
                idx=count,
                bbox=bbox,
                landmarks=landmarks,
                confidence=None,
            )

            # Add the detection to the list
            detections.append(detection)

    # Return the list of detections
    return detections


def recognize_faces(frame, detections):

    if not detections:
        return []

    # Align faces
    faces_aligned = []
    for detection in detections:
        # Crop image to face bounding box
        face = frame[
            int(detection.bbox[0][1]) : int(detection.bbox[1][1]),
            int(detection.bbox[0][0]) : int(detection.bbox[1][0]),
        ]

        # Transform landmark coordinates to face bounding box coordinates
        landmarks_source = np.array(detection.landmarks)
        landmarks_source[:, 0] -= detection.bbox[0][0]
        landmarks_source[:, 1] -= detection.bbox[0][1]

        # Target landmark coordinates (as used in training)
        landmarks_target = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(landmarks_source, landmarks_target)
        tmatrix = tform.params[0:2, :]
        face_aligned = cv2.warpAffine(face, tmatrix, (112, 112), borderValue=0.0)
        faces_aligned.append(face_aligned)

    # Inference face embeddings with onnxruntime
    embeddings = face_recognizer.run(None, {"input_image": np.asarray(faces_aligned).astype(np.float32)})[0]
    
    # Create Identity objects
    identities = []
    for idx, detection in enumerate(detections):
        identities.append(
            Identity(
                idx=idx,
                detection=detection,
                embedding=embeddings[idx],
                face=faces_aligned[idx],
                name=f"face-{idx}"
            )
        )

    return identities


def match_faces(subjects, gallery):
    if len(gallery) == 0 or len(subjects) == 0:
        return []

    # Get Embeddings
    embs_gal = np.asarray([identity.embedding for identity in gallery])
    embs_det = np.asarray([identity.embedding for identity in subjects])

    # Calculate Cosine Distances
    cos_distances = cosine_distances(embs_det, embs_gal)

    # Find Matches
    matches = []
    for ident_idx, identity in enumerate(subjects):
        dist_to_identity = cos_distances[ident_idx]
        idx_min = np.argmin(dist_to_identity)
        if dist_to_identity[idx_min] < similarity_threshold:
            matches.append(
                Match(
                    subject_id=identity,
                    gallery_id=gallery[idx_min],
                    distance=dist_to_identity[idx_min],
                )
            )

    # Sort Matches by identity_idx
    matches = sorted(matches, key=lambda match: match.gallery_id.name)

    return matches


def draw_annotations(frame, detections, matches):
    shape = np.asarray(frame.shape[:2][::-1])

    # Upscale frame to 1080p for better visualization of drawn annotations
    frame = cv2.resize(frame, (1920, 1080))
    upscale_factor = np.asarray([1920 / shape[0], 1080 / shape[1]])
    shape = np.asarray(frame.shape[:2][::-1])
    
    # Make frame writeable (for better performance)
    frame.flags.writeable = True

    # Draw Detections
    for detection in detections:
        # Draw Landmarks
        for landmark in detection.landmarks:
            cv2.circle(
                frame,
                (landmark * upscale_factor).astype(int),
                2,
                (255, 255, 255),
                -1,
            )

        # Draw Bounding Box
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (255, 0, 0),
            2,
        )

        # Draw Index
        cv2.putText(
            frame,
            str(detection.idx),
            (
                ((detection.bbox[1][0] + 2) * upscale_factor[0]).astype(int),
                ((detection.bbox[1][1] + 2) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    # Draw Matches
    for match in matches:
        detection = match.subject_id.detection
        name = match.gallery_id.name

        # Draw Bounding Box in green
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (0, 255, 0),
            2,
        )
        
        # Draw Banner
        cv2.rectangle(
            frame,
            (
                (detection.bbox[0][0] * upscale_factor[0]).astype(int),
                (detection.bbox[0][1] * upscale_factor[1] - (shape[1] // 25)).astype(int),
            ),
            (
                (detection.bbox[1][0] * upscale_factor[0]).astype(int),
                (detection.bbox[0][1] * upscale_factor[1]).astype(int),
            ),
            (255, 255, 255),
            -1,
        )

        # Draw Name
        cv2.putText(
            frame,
            name,
            (
                ((detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]).astype(int),
                ((detection.bbox[0][1] - shape[1] // 100) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    return frame


# Init gallery
gallery = []
for file in files:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    # Face Detection
    detections = detect_faces(img)

    if detections:
        # Face Recognition
        subjects = recognize_faces(img, detections[:1])

        # Add to gallery
        gallery.append(
            Identity(
                name=os.path.splitext(file.name)[0],
                embedding=subjects[0].embedding,
                face=subjects[0].face,
            )
        )

transfer_queue: "queue.Queue[Stats, List[Match]]" = queue.Queue()


class Camera:
    def __init__(self, video_receiver):
        self.currentFrame = None
        self.capture = video_receiver
        self.stop = False
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True

    def update_frame(self):
        while True:
            self.currentFrame = self.capture.get_frame()
            if self.stop:
                break

    # Get current frame
    def get_frame(self):
        return self.currentFrame


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert frame to numpy array
    frame = frame.to_ndarray(format="rgb24")

    # Run face detection
    start = time.time()
    detections = detect_faces(frame)
    num_faces = len(detections) if detections else 0
    time_detection = (time.time() - start) * 1000

    # Run face recognition
    start = time.time()
    subjects = recognize_faces(frame, detections)
    time_recognition = (time.time() - start) * 1000

    # Run face matching
    start = time.time()
    matches = match_faces(subjects, gallery)
    time_matching = (time.time() - start) * 1000

    # Draw annotations
    start = time.time()
    frame = draw_annotations(frame, detections, matches)
    time_annotation = (time.time() - start) * 1000

    # Convert frame back to av.VideoFrame
    frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

    stats = Stats(
                num_faces=num_faces,
                detection=time_detection,
                recognition=time_recognition,
                matching=time_matching,
                annotation=time_annotation,
            )

    return frame, stats, matches

with disp_stream:
    print("start stream")
    ctx = webrtc_streamer(
        key="LiveFaceRecognition",
        mode=WebRtcMode.SENDONLY,
        #rtc_configuration={"iceServers": ICE_SERVERS},
        #video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": 
            {
                "width": 1920,
                #"height": 1080,
                # "frameRate": 30,
            }
                                  , "audio": False},
    )


# Show Gallery Identities
if gallery:
    disp_gallery.image(
        image=[identity.face for identity in gallery],
        caption=[match.name for match in gallery],
    )

cam = Camera(ctx.video_receiver)



if ctx.state.playing:
    print("start thread")
    cam.thread.start()
    print("thread successfully started")
    start = time.time()
    while True:
        frame = cam.get_frame()
        if frame is not None:

            frame, stats, matches = video_frame_callback(frame)
            frame = frame.to_ndarray(format="rgb24")
            
            stream_window.image(frame, channels="RGB")
            
            # Show Stats
            stats.fps = 1 / (time.time() - start)
            start = time.time()

            disp_stats.dataframe(
                pd.DataFrame([stats.__dict__]).applymap(lambda x: (format_dflist(x))),
                use_container_width=True,
            )

            # Show Matches
            if matches:
                disp_matches.image(
                    image=[match.subject_id.face for match in matches],
                    caption=[match.gallery_id.name for match in matches],
                )
            else:
                disp_matches.info("No matches found yet ...")
