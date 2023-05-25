---
title: LiveFaceID
emoji: 🐢
colorFrom: gray
colorTo: pink
sdk: streamlit
sdk_version: 1.19.0
app_file: app.py
pinned: false
license: mit
---

## Description



## BUGS/TODOS:
- [ ] In online demo, the identy preview is only showing the first frame of the video
- [ ] Progress for download model files etc.
- [ ] Implement the possibility to play a video instead of webcam (for example BigBang and use sample Gallery Images from Bigbang)


# In network
When you run streamlit app on a linux machine:
```bash
wget https://github.com/suyashkumar/ssl-proxy/releases/download/v0.2.7/ssl-proxy-linux-amd64.tar.gz

gzip -d ssl-proxy-linux-amd64.tar.gz
tar -xvf ssl-proxy-linux-amd64.tar

./ssl-proxy-linux-amd64 -from 0.0.0.0:8502 to 0.0.0.0:8501
```

This runs the reverse proxy, to be able to access the streamlit app via https in a local network. 
