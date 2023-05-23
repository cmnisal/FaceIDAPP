from .utils import tflite_inference
from .nametypes import Identity, Match
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
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
        min_similarity: float = 0.67,
        model_name: Literal["mobileNet", "resNet50"] = "mobileNet",
    ):
        self.min_similarity = min_similarity
        self.model = tflite.Interpreter(
            model_path=get_file(
                BASE_URL + f"{model_name}.tflite", FILE_HASHES[model_name]
            )
        )

    def __call__(self, frame, detections):
        # Align Faces
        faces, faces_aligned = [], []
        for detection in detections:
            face = frame[
                int(detection.bbox[0][1]) : int(detection.bbox[1][1]),
                int(detection.bbox[0][0]) : int(detection.bbox[1][0]),
            ]
            try:
                face = cv2.resize(face, (112, 112))
            except:
                face = np.zeros((112, 112, 3))

            faces.append(face)
            faces_aligned.append(self.align(frame, detection.landmarks))

        # Do Inference
        if len(faces_aligned) == 0:
            return []
        
        # Normalize images from [0, 255] to [0, 1]
        faces_aligned_norm = np.asarray(faces_aligned).astype(np.float32) / 255.0

        embs_det = tflite_inference(self.model, faces_aligned_norm)
        embs_det = np.asarray(embs_det[0])

        # Save Identities
        identities = []
        for idx, detection in enumerate(detections):
            identities.append(
                Identity(
                    detection_idx=detection.idx,
                    embedding=embs_det[idx],
                    face_aligned=faces_aligned[idx],
                )
            )
        return identities
        

    def find_matches(self, identities, gallery):
        if len(gallery) == 0 or len(identities) == 0:
            return []

        # Get Embeddings
        embs_gal = np.asarray([identity.embedding for identity in gallery])
        embs_det = np.asarray([identity.embedding for identity in identities])

        # Calculate Cosine Distances
        cos_distances = cosine_distances(embs_det, embs_gal)

        # Find Matches
        matches = []
        for ident_idx, identity in enumerate(identities):
            dist_to_identity = cos_distances[ident_idx]
            idx_min = np.argmin(dist_to_identity)
            if dist_to_identity[idx_min] < self.min_similarity:
                matches.append(
                    Match(
                        identity_idx=identity.detection_idx,
                        gallery_idx=idx_min,
                        distance=dist_to_identity[idx_min],
                        name=gallery[idx_min].name,
                    )
                )
        
        # Sort Matches by identity_idx
        matches = sorted(matches, key=lambda match: match.gallery_idx)

        return matches

    @staticmethod
    def align(img, landmarks_source, target_size=(112, 112)):
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
        face_aligned = cv2.warpAffine(img, tmatrix, target_size, borderValue=0.0)
        return face_aligned
