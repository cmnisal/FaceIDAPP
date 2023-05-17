from typing import NamedTuple, List
import numpy as np


class Detection(NamedTuple):
    bbox: List[int]
    landmarks: List[List[int]]
    name: str = None
    face: np.ndarray = None
    embedding: np.ndarray = None
    embedding_match: np.ndarray = None
    face_match: np.ndarray = None
    distance: float = None


class Stats(NamedTuple):
    fps: float = 0
    resolution: List[int] = [None, None, None]
    num_faces: int = 0
    detection: float = None
    alignment: float = None
    inference: float = None
    recognition: float = None
    drawing: float = None
    

class Identity(NamedTuple):
    name: str
    embedding: np.ndarray
    image: np.ndarray
