from __future__ import annotations
import numpy as np


class ViterbiGrid:
    def __init__(
        self,
        score_array: np.ndarray,
        origin: np.ndarray,
        zvec: np.ndarray,
        yvec: np.ndarray,
        xvec: np.ndarray,
    ) -> None:
        ...
    
    def viterbi_simple(self, dist_min: float, dist_max: float) -> tuple[np.ndarray, float]: ...
    def viterbi(self, dist_min: float, dist_max: float, skew_max: float) -> tuple[np.ndarray, float]: ...

def alleviate(
    arr: np.ndarray,
    label: np.ndarray,
    nrise: int,
    iterations: int,
) -> None:
    ...