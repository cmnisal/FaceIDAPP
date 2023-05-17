from typing import NamedTuple, List
import numpy as np
import pandas as pd
from tools.utils import format_list


class Detection(NamedTuple):
    bbox: List[int]
    landmarks: List[List[int]]
    name: str = None
    face: np.ndarray = None
    embedding: np.ndarray = None
    embedding_match: np.ndarray = None
    face_match: np.ndarray = None
    distance: float = None


# Sample Detection:

detection = Detection(
    bbox=[0.1, 0.2, 0.3, 0.4],
    landmarks=[[0.1, 0.2], [0.3, 0.4]],
    name="John Doe",
    face=np.random.rand(100, 100, 3),
    embedding=np.random.rand(128),
    embedding_match=np.random.rand(128),
    face_match=np.random.rand(100, 100, 3),
)

detections = [detection, detection, detection]

df = pd.DataFrame(detections)

df = df.applymap(lambda x: (format_list(x)))


print(df)