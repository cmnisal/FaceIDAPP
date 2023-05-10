import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode


@st.cache_resource(experimental_allow_widgets=True)
def init_webcam(width=680):
    ctx = webrtc_streamer(
        key="FaceIDAppDemo",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={
            "video": {
                "width": {
                    "min": width,
                    "ideal": width,
                    "max": width,
                },
            },
            "audio": False,
        },
        video_receiver_size=1,
        async_processing=True,
    )
    return ctx.video_receiver
