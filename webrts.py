import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import mediapipe as mp


class OpenCVVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.FACEMESH_TESSELATION = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def recv(self, frame):
        # Convert frame to Numpy
        frame = frame.to_ndarray(format="rgb24")

        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(frame)

        # Draw the face mesh on the original frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.draw_spec,
                    connection_drawing_spec=self.draw_spec,
                )

        # Convert frame back to WebRTC frame
        frame = frame[:, :, ::-1].tobytes()

        return frame

    def __del__(self):
        self.face_mesh.close()


# Streamlit app
def main():
    st.title("Webcam Stream App")

    max_faces = st.sidebar.number_input("Maximum Number of Faces", value=1, min_value=1)
    st.sidebar.markdown("---")
    st.text("Face Mesh")
    detection_confidence = st.sidebar.slider(
        "Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.9
    )
    tracking_confidence = st.sidebar.slider(
        "Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9
    )

    webrtc_streamer(
    key="sample",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=OpenCVVideoProcessor,
    async_processing=True,
    )
if __name__ == "__main__":
    main()
