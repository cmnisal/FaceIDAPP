import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
import mediapipe as mp


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class OpenCVVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        return frame



# Streamlit app
def main():
    st.title("Webcam Stream App")
    
    webrtc_streamer(
    key="myExample",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=OpenCVVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    )
if __name__ == "__main__":
    main()
