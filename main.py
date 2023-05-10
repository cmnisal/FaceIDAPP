import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# Streamlit app
st.title("FaceID App Demonstration")

# Instantiate WebRTC (and show start button)
ctx = webrtc_streamer(
    key="FaceIDAppDemo",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_receiver_size=1,
    async_processing=True,
)

# Live Stream Display
image_loc = st.empty()

if ctx.video_receiver:
    while True:
        try:
            frame = ctx.video_receiver.get_frame(timeout=1)
            img = frame.to_ndarray(format="rgb24")
        except:
            continue

        # Display Live Frame
        image_loc.image(img)
