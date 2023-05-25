import streamlit as st
import tensorflow as tf
from tools.utils import get_file
import torch

URLS = {
    "mobileNet": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/mobileNet.tflite",
    "resNet": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/resNet.tflite",
    "FaceTransformerOctupletLoss": "https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/FaceTransformerOctupletLoss.pt",
    "ArcFaceOctupletLoss": "https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/ArcFaceOctupletLoss.tf.zip",
}

FILE_HASHES = {
    "mobileNet": "6c19b789f661caa8da735566490bfd8895beffb2a1ec97a56b126f0539991aa6",
    "resNet": "f4d8b0194957a3ad766135505fc70a91343660151a8103bbb6c3b8ac34dbb4e2",
    "FaceTransformerOctupletLoss": "f2c7cf1b074ecb17e546dc7043a835ad6944a56045c9675e8e1158817d662662",
    "ArcFaceOctupletLoss": "8603f374fd385081ce5ce80f5997e3363f4247c8bbad0b8de7fb26a80468eeea",
}

st.title("FaceIDLight")

filename = get_file(URLS["FaceTransformerOctupletLoss"], FILE_HASHES["FaceTransformerOctupletLoss"])
st.write(filename)

model = torch.load(filename)
st.write(model)