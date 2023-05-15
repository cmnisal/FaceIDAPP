from typing import NamedTuple, List
import numpy as np


class Detection(NamedTuple):
    bbox: List[int]
    landmarks: List[List[int]]
    name: str = None
    face: np.ndarray = None
    emdedding: np.ndarray = None
    distance: float = None


class Stats(NamedTuple):
    fps: float
    resolution: List[int]
    num_faces: int


class Timings(NamedTuple):
    detection: float
    normalization: float
    inference: float
    recognition: float
    drawing: float

class Identity(NamedTuple):
    name: str
    embedding: np.ndarray
    image: np.ndarray
