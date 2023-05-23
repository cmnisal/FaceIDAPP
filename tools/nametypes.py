from typing import NamedTuple, List
import numpy as np


class Detection(NamedTuple):
    idx: int = None
    bbox: List[List[float]] = None
    landmarks: List[List[float]] = None
    confidence: float = None
    

class Identity(NamedTuple):
    detection_idx: int = None
    name: str = None
    embedding: np.ndarray = None
    face_aligned: np.ndarray = None


class Stats(NamedTuple):
    fps: float = 0
    resolution: List[int] = [None, None, None]
    num_faces: int = 0
    detection: float = None
    recognition: float = None
    matching: float = None
    annotation: float = None
    

class Match(NamedTuple):
    identity_idx: int = None
    gallery_idx: int = None
    distance: float = None
    name: str = None
