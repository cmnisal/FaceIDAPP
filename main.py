import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
import cv2


# Streamlit app
st.title("FaceID App Demonstration")

# Instantiate WebRTC (and show start button)
ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun2.l.google.com:19305"]}]},
    video_receiver_size=1,
    async_processing=True,
)

facedetector = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
)

def annotate(frame, results):
    orig_h, orig_w = frame.shape[0:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(468):
                pt1 = face_landmarks.landmark[i]
                x1, y1 = int(pt1.x * orig_w), int(pt1.y * orig_h)
                cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)

    return frame

# Live Stream Display
image_loc = st.empty()

if ctx.video_receiver:
    while True:
        try:
            frame = ctx.video_receiver.get_frame(timeout=1)
            img = frame.to_ndarray(format="rgb24")
        except:
            continue

        # Process Frame
        result = facedetector.process(img)
        img = annotate(img, result)

        # Display Live Frame
        image_loc.image(img)
