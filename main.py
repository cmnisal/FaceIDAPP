import streamlit as st
import time
from tools.webcam import init_webcam
import logging


# Set logging level to error (To avoid getting spammed by queue warnings etc.)
logging.basicConfig(level=logging.ERROR)


# Set page layout for streamlit to wide
st.set_page_config(layout="wide")


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


# -----------------------------------------------------------------------------------------------
# Streamlit App
st.title("FaceID App Demonstration")

# Get Access to Webcam
webcam = init_webcam()

# KPI Section
st.markdown("**Stats**")
kpi = KPI(["**FrameRate**"])
st.markdown("---")

# Live Stream Display
stream_display = st.empty()
st.markdown("---")

if webcam:
    prevTime = 0
    while True:
        try:
            # Get Frame from Webcam
            frame = webcam.get_frame(timeout=1)

            # Convert to OpenCV Image
            frame = frame.to_ndarray(format="rgb24")
        except:
            continue

        # DISPLAY THE LIVE STREAM --------------------------------------------------
        stream_display.image(
            frame, channels="RGB", caption="Live-Stream", use_column_width=True
        )

        # CALCULATE FPS -----------------------------------------------------------
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # UPDATE KPIS -------------------------------------------------------------
        kpi.update_kpi([fps])
