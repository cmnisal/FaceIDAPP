import numpy as np
import tflite_runtime.interpreter as tflite
from sklearn.metrics.pairwise import cosine_distances
import streamlit as st
import time


MODEL_PATHS = {
    "MobileNet": "./models/mobileNet.tflite",
    "ResNet": "./models/resNet.tflite",
}


#@st.cache_resource
def load_identification_model(name="MobileNet"):
    model = tflite.Interpreter(model_path=MODEL_PATHS[name])
    return model


def inference(imgs, model):
    if len(imgs) > 0:
        imgs = np.asarray(imgs).astype(np.float32) / 255
        model.resize_tensor_input(model.get_input_details()[0]["index"], imgs.shape)
        model.allocate_tensors()
        model.set_tensor(model.get_input_details()[0]["index"], imgs)
        model.invoke()
        embs = [model.get_tensor(elem["index"]) for elem in model.get_output_details()]
        return embs[0]
    else:
        return []


def identify(embs_src, embs_gal, labels_gal, imgs_gal, thresh=None):
    all_dists = cosine_distances(embs_src, embs_gal)
    ident_names, ident_dists, ident_imgs = [], [], []
    for dists in all_dists:
        idx_min = np.argmin(dists)
        if thresh and dists[idx_min] > thresh:
            dist = dists[idx_min]
            pred = None
        else:
            dist = dists[idx_min]
            pred = idx_min
        ident_names.append(labels_gal[pred] if pred is not None else "Unknown")
        ident_dists.append(dist)
        ident_imgs.append(imgs_gal[pred] if pred is not None else None)
    return ident_names, ident_dists, ident_imgs
