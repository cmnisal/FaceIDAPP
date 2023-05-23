from .face_detection import FaceDetection
from .face_recognition import FaceRecognition
from .nametypes import Identity
import cv2
import os
import numpy as np


def init_gallery(files, min_detections_conf=0.8, min_similarity=0.67, model_name="mobileNet"):
    face_detector = FaceDetection(min_detections_conf=min_detections_conf)
    face_recognizer = FaceRecognition(model_name=model_name, min_similarity=min_similarity)

    gallery = []
    for file in files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.cvtColor(
            cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        # Face Detection
        img, detections = face_detector(img)

        if detections == []:
            continue
        elif len(detections) > 1:
            detections = detections[:1]

        # Face Recognition
        identities = face_recognizer(img, detections)

        # Add to gallery
        gallery.append(
            Identity(
                name=os.path.splitext(file.name)[0],
                embedding=identities[0].embedding,
                face_aligned=identities[0].face_aligned,
            )
        )

    return gallery
