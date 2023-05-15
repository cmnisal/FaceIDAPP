import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import os
import av
from twilio.rest import Client
import queue
from typing import List, NamedTuple
import numpy as np


account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

token = client.tokens.create()

RTC_CONFIGURATION = {"iceServers": token.ice_servers}


class Stats(NamedTuple):
    framerate: float

# Set logging level to error (To avoid getting spammed by queue warnings etc.)
logging.basicConfig(level=logging.ERROR)


# Set page layout for streamlit to wide
st.set_page_config(layout="wide")

result_queue: "queue.Queue[Stats]" = queue.Queue()


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

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    start = time.time()
    img = frame.to_ndarray(format="rgb24")
    time.sleep(0.25)
    stop = time.time()
    result_queue.put(Stats(1 / (stop - start)))
    return av.VideoFrame.from_ndarray(img, format="rgb24")


# Streamlit app
st.title("FaceID App Demonstration")

ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


# KPI Section
st.markdown("**Stats**")
kpi = KPI(["**FrameRate**"])
st.markdown("---")

if ctx.state.playing:
    while True:
        stats = result_queue.get()
        kpi.update_kpi([stats.framerate])
        print(stats.framerate)
        time.sleep(0.1)
