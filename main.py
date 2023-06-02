import streamlit as st
import time
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import onnxruntime as rt
import threading
import mediapipe as mp
import os
from twilio.rest import Client
import cv2
from skimage.transform import SimilarityTransform
from types import SimpleNamespace
from sklearn.metrics.pairwise import cosine_distances
from aiortc import VideoStreamTrack
import asyncio


class Grabber(object):
    def __init__(self, video_receiver) -> None:
        self.currentFrame = None
        self.capture = video_receiver
        self.grabber_thread = threading.Thread(target=self.update_frame)
        self.grabber_thread.daemon = True

    def update_frame(self) -> None:
        while True:
            self.currentFrame = self.capture.get_frame()

    def get_frame(self) -> av.VideoFrame:
        return self.currentFrame
    
# Similarity threshold for face matching
SIMILARITY_THRESHOLD = 1.2

# Get twilio ice server configuration using twilio credentials from environment variables (set in streamlit secrets)
# Ref: https://www.twilio.com/docs/stun-turn/api
ICE_SERVERS = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]).tokens.create().ice_servers

# Set page layout for streamlit to wide
st.set_page_config(layout="wide", page_title="Live Face Recognition", page_icon=":sunglasses:")

# Streamlit app
st.title("Live Webcam Face Recognition")

st.markdown("**Live Stream**")
ctx_container = st.container()
stream_container = st.empty()

# Start streaming component

ctx = webrtc_streamer(
    key="LiveFaceRecognition",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": ICE_SERVERS},
    media_stream_constraints={"video": {"width": 1920}, "audio": False},
)

# Initialize frame grabber
grabber = Grabber(ctx.video_receiver)


class ImageRenderingTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(10)

    async def add_image(self, img: np.ndarray):
        await self.queue.put(img)

    async def recv(self):
        img = await self.queue.get()
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

video_track = ImageRenderingTrack()

ctx2 = webrtc_streamer(
    key="LiveFaceRecognition2",
    mode=WebRtcMode.RECVONLY,
    source_video_track=ctx.video_track,
    desired_playing_state=ctx.state.playing,
    rtc_configuration={"iceServers": ICE_SERVERS},
    media_stream_constraints={"video": {"width": 1920}, "audio": False},
)

grabber.thread.start()




if ctx.state.playing:
    # Start frame grabber in background thread
    

    # Start main loop
    while True:
        frame = grabber.get_frame()
        if frame is not None:
            # Convert frame to numpy array
            frame = frame.to_ndarray(format="rgb24")

            # Show Stream
            video_track.add_image(frame)
