from __future__ import annotations
import numpy as np

def viterbi(
    score: np.ndarray,
    origin: np.ndarray,
    zvec: np.ndarray,
    yvec: np.ndarray,
    xvec: np.ndarray,
    dist_min: float,
    dist_max: float,
) -> tuple[np.ndarray, float]:
    ...
