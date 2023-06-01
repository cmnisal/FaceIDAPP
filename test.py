import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import time
import threading


# Streamlit app
st.title("FaceID App Demonstration")

class Camera:
    def __init__(self, video_receiver):
        self.currentFrame = None
        self.capture = video_receiver
        self.thread = threading.Thread(target=self.update_frame)
        #self.thread.start()

    def update_frame(self):
        while True:
            self.currentFrame = self.capture.get_frame()

    # Get current frame
    def get_frame(self):
        return self.currentFrame

# Instantiate WebRTC (and show start button)
ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": True, "audio": False},
    on_video_ended=lambda: st.experimental_rerun(),
)

# Live Stream Display
image_loc = st.empty()

cam = Camera(ctx.video_receiver)

if ctx.video_receiver:
    cam.thread.start()
    print("Video Receiver Found")
    while True:
        try:
            frame = cam.get_frame()
            img = frame.to_ndarray(format="rgb24")
        except:
            continue
        
        time.sleep(0.5)

        # Display Live Frame
        tmp = time.time()
        image_loc.image(img)
        print(f" Image Printing: {(time.time() - tmp) * 1000}", end="\r")
