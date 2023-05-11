import numpy as np
import cv2
from skimage.transform import SimilarityTransform


FIVE_LANDMARKS = [470, 475, 1, 57, 287]


def align(img, landmarks, target_size=(112, 112)):
    dst = np.array(
        [
            [
                landmarks.landmark[i].x * img.shape[1],
                landmarks.landmark[i].y * img.shape[0],
            ]
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
    return cv2.warpAffine(img, tmatrix, target_size, borderValue=0.0)



def align_faces(img, detections):
    aligned_faces = [align(img, detection.multi_face_landmarks) for detection in detections]
    return aligned_faces
