import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import os
from twilio.rest import Client


account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

token = client.tokens.create()


RTC_CONFIGURATION={
  "iceServers": token.ice_servers
}


@st.cache_resource(experimental_allow_widgets=True)
def init_webcam(width=680):
    ctx = webrtc_streamer(
        key="FaceIDAppDemo",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
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

