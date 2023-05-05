import numpy as np
import cv2
from skimage.transform import SimilarityTransform


FIVE_LANDMARKS = [470, 475, 1, 57, 287]


class FaceNormalizer:
    def __init__(self, target_size=(112, 112)):
        self.target_size = target_size

    def face_cropper(self, frame, detections):
        if not detections[0]:
            return []

        faces = []
        for detection in detections[0]:
            faces.append(self.normalize(frame, detection.landmark))
        return faces

    def normalize(self, img, landmarks):
        dst = np.array(
            [
                [landmarks[i].x * img.shape[1], landmarks[i].y * img.shape[0]]
                for i in FIVE_LANDMARKS
            ],
        )

        src = np.array(
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
        tform.estimate(dst, src)
        tmatrix = tform.params[0:2, :]
        return cv2.warpAffine(img, tmatrix, self.target_size, borderValue=0.0)
