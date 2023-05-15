import logging
import os
import urllib.request
from pathlib import Path
import streamlit as st
from twilio.rest import Client
import os
import hashlib


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


def get_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(65535), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url, model_path: Path, file_hash=None):
    if model_path.exists():
        if file_hash:
            hasher = hashlib.sha256()
            with open(model_path, "rb") as file:
                for chunk in iter(lambda: file.read(65535), b""):
                    hasher.update(chunk)
            if not hasher.hexdigest() == file_hash:
                print(
                    "A local file was found, but it seems to be incomplete or outdated because the file hash does not "
                    "match the original value of "
                    + file_hash
                    + " so data will be downloaded."
                )
                download = True
            else:
                print("Using a verified local file.")
                download = False
    else:
        model_path.mkdir(parents=True, exist_ok=True)
        print("Downloading data ...")
        download = True

    if download:

        # These are handles to two visual elements to animate.
        weights_warning, progress_bar = None, None
        try:
            weights_warning = st.warning("Downloading %s..." % url)
            progress_bar = st.progress(0)
            with open(model_dir, "wb") as output_file:
                with urllib.request.urlopen(url) as response:
                    length = int(response.info()["Content-Length"])
                    counter = 0.0
                    MEGABYTES = 2.0**20.0
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        counter += len(data)
                        output_file.write(data)

                        # We perform animation by overwriting the elements.
                        weights_warning.warning(
                            "Downloading %s... (%6.2f/%6.2f MB)"
                            % (url, counter / MEGABYTES, length / MEGABYTES)
                        )
                        progress_bar.progress(min(counter / length, 1.0))

        # Finally, we remove these visual elements by calling .empty().
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()
