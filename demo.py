import cv2
import mediapipe as mp
import streamlit as st
import dlib
import face_recognition
import numpy as np
from PIL import Image

# Initialize Mediapipe Face Mesh and Face Recognition
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Initialize face landmarks
face_landmark_points = [1, 234, 454, 10, 152]

# Load gallery images
gallery_images = {}
for image_path in st.file_uploader("Upload gallery images", type=["jpg", "jpeg", "png"], accept_multiple_files=True):
    image = face_recognition.load_image_file(image_path)
    image_face_encoding = face_recognition.face_encodings(image)[0]
    gallery_images[image_path.name] = {
        'image': image,
        'face_encoding': image_face_encoding
    }
print("TEST")

def normalize_face(frame, landmarks, size=160):
    h, w, _ = frame.shape
    points = np.array([(landmarks[p].x * w, landmarks[p].y * h) for p in face_landmark_points])
    face = dlib.full_object_detections()
    rect = dlib.rectangle(0, 0, w, h)
    face.append(dlib.full_object_detection(rect, points))
    cropped = dlib.get_face_chip(frame, face[0], size)
    return cropped, points

st.title("Webcam Face Recognition")
FRAME_WINDOW = st.image([])
while True:
    # Capture frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Face Mesh
    results = face_mesh.process(frame_rgb)

    # Draw face landmarks and bounding box
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Normalize and crop face
            cropped_face = frame_rgb #normalize_face(frame_rgb, face_landmarks.landmark)

            # Extract face encoding from cropped face
            if len(face_recognition.face_encodings(cropped_face)) > 0:
                face_encoding = face_recognition.face_encodings(cropped_face)[0]
            else:
                continue

            # Compare cropped face to gallery images
            recognized_label = "Unknown"
            min_distance = 1.0
            for label, data in gallery_images.items():
                distance = np.linalg.norm(data['face_encoding'] - face_encoding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_label = label
            
            # Draw face mesh
            #mp_drawing.draw_landmarks(frame_rgb, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,)

            # Display the label next to the face mesh
            x = int(face_landmarks.landmark[0].x * frame.shape[1])
            y = int(face_landmarks.landmark[0].y * frame.shape[0])
            cv2.putText(frame_rgb, recognized_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Update the frame
    FRAME_WINDOW.image(frame_rgb)

    # Release the webcam
    cap.release()
