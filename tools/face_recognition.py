from .nametypes import Identity, Match
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import cv2
from skimage.transform import SimilarityTransform
from typing import Literal

class FaceRecognition:
    def __init__(
        self,
        min_similarity: float = 0.67,
        model_name: Literal["mobileNet", "resNet50"] = "mobileNet",
    ):
        self.min_similarity = min_similarity
        if model_name == "mobileNet":
            from .models import MobileNetV2
            self.model = MobileNetV2()
        elif model_name == "resNet50":
            from .models import ResNet50
            self.model = ResNet50()
        elif model_name == "ArcFaceOctupletLoss":
            from .models import ArcFaceOctupletLoss
            self.model = ArcFaceOctupletLoss()
        elif model_name == "FaceTransformerOctupletLoss":
            from .models import FaceTransformerOctupletLoss
            self.model = FaceTransformerOctupletLoss()
        else:
            raise ValueError(
                f"model_name must be one of ['mobileNet', 'resNet50', 'ArcFaceOctupletLoss', 'FaceTransformerOctupletLoss'], got {model_name}"
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

        embs_det = self.model(faces_aligned_norm)
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
        if isinstance(landmarks_source, list):
            landmarks_source = np.array(landmarks_source, dtype=np.float32)

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
