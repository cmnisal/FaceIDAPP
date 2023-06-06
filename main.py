from aiortc import VideoStreamTrack
import time
import cv2
import streamlit as st
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import asyncio
import threading

st.title("Live Stream")

ctx = webrtc_streamer(
    key="LiveFaceRecognitionSender",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": {"width": 1920}, "audio": False},
)

class Grabber(object):
    def __init__(self, video_receiver) -> None:
        self.currentFrame = None
        self.video_receiver = video_receiver
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self._stop = False

    def start(self) -> None:
        while self.video_receiver == None:
            time.sleep(0.1)
        self.thread.start()

    def stop(self) -> None:
        self._stop = True
        self.thread.join()

    def update_frame(self) -> None:
        while not self._stop:
            self.currentFrame = self.video_receiver.get_frame()

    def read(self) -> VideoFrame:
        frame = self.currentFrame.to_ndarray(format="bgr24")
        pts, time_base = self.currentFrame.pts, self.currentFrame.time_base
        cv2.putText(frame, "Hello World", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        frame = VideoFrame.from_ndarray(frame, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
    
class SimpleVideoTrack(VideoStreamTrack):
    def __init__(self, grabber):
        super().__init__()
        self.grabber = grabber

    async def recv(self):
        frame = self.grabber.read()
        await asyncio.sleep(0.04)
        print(f"VideoTrack: {frame.pts}, {frame.time_base}")
        return frame
    
time.sleep(1)
grabber = Grabber(ctx.video_receiver)
video_track = SimpleVideoTrack(grabber)

if ctx.state.playing:
    grabber.start()
    webrtc_streamer(
        key="DisplayTrack",
        mode=WebRtcMode.RECVONLY,
        source_video_track=video_track,
        desired_playing_state=True,
    )
else:
    try:
        grabber.stop()
    except:
        pass
