import mediapipe as mp
import streamlit as st


class Detection:
    multi_face_bboxes = []
    multi_face_landmarks = []


#@st.cache_resource
def load_detection_model(max_faces=2, detection_confidence=0.5, tracking_confidence=0.5):
    model = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces,
        )
    return model


def detect_faces(frame, model):
    
    # Process the frame with MediaPipe Face Mesh
    results = model.process(frame)

    # Get the Bounding Boxes from the detected faces
    detections = []
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            x_coords = [
                landmark.x * frame.shape[1] for landmark in landmarks.landmark
            ]
            y_coords = [
                landmark.y * frame.shape[0] for landmark in landmarks.landmark
            ]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            detection = Detection()
            detection.multi_face_bboxes=[x_min, y_min, x_max, y_max]
            detection.multi_face_landmarks=landmarks
            detections.append(detection)
    return detections
