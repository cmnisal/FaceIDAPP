from .utils import tflite_inference
from .nametypes import Identity
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import os
import cv2
from skimage.transform import SimilarityTransform
from .utils import get_file
import tflite_runtime.interpreter as tflite
from typing import Literal


BASE_URL = "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/"

FILE_HASHES = {
    "mobileNet": "6c19b789f661caa8da735566490bfd8895beffb2a1ec97a56b126f0539991aa6",
    "resNet": "f4d8b0194957a3ad766135505fc70a91343660151a8103bbb6c3b8ac34dbb4e2",
}


class FaceRecognition:
    def __init__(
        self,
        thresh: float = 0.67,
        gallery_files: list = None,
        model_name: Literal["mobileNet", "resNet50"] = "mobileNet",
    ):
        self.gallery = None
        self.thresh = thresh
        self.gallery = self._initialize_gallery(gallery_files)

        self.model = tflite.Interpreter(
            model_path=get_file(
                BASE_URL + f"{model_name}.tflite", FILE_HASHES[model_name]
            )
        )

    def _initialize_gallery(self, files):
        if files is None:
            return []

        # TODO
        gallery = []

        return gallery

    def __call__(self, frame, detections):
        if len(detections) == 0 or len(self.gallery) == 0:
            return frame, []

        # Get Gallery Embeddings
        embs_gal = np.asarray([identity.embedding for identity in self.gallery])

        # Get Detections Embeddings
        faces_aligned = [self.align(detection.face) for detection in detections]
        embs_det = tflite_inference(faces_aligned)

        # Get Cosine Distances
        cos_distances = cosine_distances(embs_det, embs_gal)

        # Get Matching Identities
        identities = []
        for det_idx in range(len(detections)):
            idx_min = np.argmin(cos_distances[det_idx])
            if cos_distances[det_idx][idx_min] < self.thresh:
                identities.append(
                    Identity(
                        name=os.path.splittext(self.gallery[idx_min])[0],
                        embedding_match=self.gallery[idx_min].embedding,
                        face_match=self.gallery[idx_min].image,
                        dist=cos_distances[det_idx][idx_min],
                    )
                )

        return frame, identities

    @staticmethod
    def align(face, landmarks_source, target_size=(112, 112)):
        landmarks_target = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(landmarks_source, landmarks_target)
        tmatrix = tform.params[0:2, :]
        face_aligned = cv2.warpAffine(face, tmatrix, target_size, borderValue=0.0)
        return face_aligned
