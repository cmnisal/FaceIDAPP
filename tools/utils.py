import logging
import os
import streamlit as st
from twilio.rest import Client
import os
import numpy as np
import hashlib
import tempfile
import os
import hashlib
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlopen


logger = logging.getLogger(__name__)


@st.cache_data
def get_ice_servers(name="twilio"):
    """Get ICE servers from Twilio.
    Returns:
        List of ICE servers.
    """
    if name == "twilio":
        # Ref: https://www.twilio.com/docs/stun-turn/api
        try:
            account_sid = os.environ["TWILIO_ACCOUNT_SID"]
            auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        except KeyError:
            logger.warning(
                "Twilio credentials are not set. Fallback to a free STUN server from Google."
            )
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

        client = Client(account_sid, auth_token)

        token = client.tokens.create()

        return token.ice_servers

    elif name == "metered":
        try:
            username = os.environ["METERED_USERNAME"]
            credential = os.environ["METERED_CREDENTIAL"]
        except KeyError:
            logger.warning(
                "Metered credentials are not set. Fallback to a free STUN server from Google."
            )
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

        ice_servers = [
            {"url": "stun:a.relay.metered.ca:80", "urls": "stun:a.relay.metered.ca:80"},
            {
                "url": "turn:a.relay.metered.ca:80",
                "username": username,
                "urls": "turn:a.relay.metered.ca:80",
                "credential": credential,
            },
            {
                "url": "turn:a.relay.metered.ca:80?transport=tcp",
                "username": username,
                "urls": "turn:a.relay.metered.ca:80?transport=tcp",
                "credential": credential,
            },
            {
                "url": "turn:a.relay.metered.ca:443",
                "username": username,
                "urls": "turn:a.relay.metered.ca:443",
                "credential": credential,
            },
            {
                "url": "turn:a.relay.metered.ca:443?transport=tcp",
                "username": username,
                "urls": "turn:a.relay.metered.ca:443?transport=tcp",
                "credential": credential,
            },
        ]
        return ice_servers
    else:
        raise ValueError(f"Unknown name: {name}")


# Function to format floats within a list
def format_dflist(val):
    if isinstance(val, list):
        return [format_dflist(num) for num in val]
    if isinstance(val, np.ndarray):
        return np.asarray([format_dflist(num) for num in val])
    if isinstance(val, np.float32):
        return f"{val:.2f}"
    if isinstance(val, float):
        return f"{val:.2f}"
    else:
        return val


def rgb(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def tflite_inference(model, img):
    """Inferences an image through the model with tflite interpreter on CPU
    :param model: a tflite.Interpreter loaded with a model
    :param img: image
    :return: list of outputs of the model
    """
    # Check if img is np.ndarray
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    
    # Check if dim is 4
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.resize_tensor_input(input_details[0]["index"], img.shape)
    model.allocate_tensors()
    model.set_tensor(input_details[0]["index"], img.astype(np.float32))
    model.invoke()
    return [model.get_tensor(elem["index"]) for elem in output_details]


def get_file(origin, file_hash, is_zip=False):
    tmp_file = os.path.join(tempfile.gettempdir(), "FaceIDLight", origin.split("/")[-1])
    os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
    if not os.path.exists(tmp_file):
        download = True
    else:
        hasher = hashlib.sha256()
        with open(tmp_file, "rb") as file:
            for chunk in iter(lambda: file.read(65535), b""):
                hasher.update(chunk)
        if not hasher.hexdigest() == file_hash:
            print(
                "A local file was found, but it seems to be incomplete or outdated because the file hash does not "
                "match the original value of " + file_hash + " so data will be downloaded."
            )
            download = True
        else:
            download = False

    if download:
        response = urlopen(origin)
        with tqdm.wrapattr(
            open(tmp_file, "wb"),
            "write",
            miniters=1,
            desc="Downloading " + origin.split("/")[-1] + " to: " + tmp_file,
            total=getattr(response, "length", None),
        ) as file:
            for chunk in response:
                file.write(chunk)
            file.close()
    if is_zip:
        with ZipFile(tmp_file, "r") as zipObj:
            zipObj.extractall(tmp_file.split(".")[0])
        tmp_file = os.path.join(tmp_file.split(".")[0])
    return tmp_file


def get_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(65535), b""):
            hasher.update(chunk)
    return hasher.hexdigest()