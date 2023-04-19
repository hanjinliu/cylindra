from __future__ import annotations
from typing import overload
import numpy as np
from numpy.typing import NDArray

class ViterbiGrid:
    def __init__(
        self,
        score_array: np.ndarray,
        origin: np.ndarray,
        zvec: np.ndarray,
        yvec: np.ndarray,
        xvec: np.ndarray,
    ) -> None: ...
    @overload
    def viterbi(
        self,
        dist_min: float,
        dist_max: float,
        /,
    ) -> tuple[NDArray[np.int32], float]: ...
    @overload
    def viterbi(
        self,
        dist_min: float,
        dist_max: float,
        skew_max: float,
        /,
    ) -> tuple[NDArray[np.int32], float]: ...
    def world_pos(self, n: int, z: int, y: int, x: int, /) -> NDArray[np.float64]: ...

class Sources:
    def has_longitudinal(self) -> bool: ...
    def has_lateral(self) -> bool: ...
    def __eq__(self, other: list[tuple[int, int]]) -> bool: ...

class Index:
    y: int
    a: int
    def __init__(self, y: int, a: int, /) -> None: ...
    def is_valid(self) -> bool: ...
    def __eq__(self, other: tuple[int, int]) -> bool: ...

class CylinderGeometry:
    def __init__(self, ny: int, na: int, nrise: int, /) -> None: ...
    def source_forward(self, y: int, a: int, /) -> Sources: ...
    def source_backward(self, y: int, a: int, /) -> Sources: ...
    def get_neighbors(self, y: int, a: int, /) -> list[Index]: ...
    def get_index(self, y: int, a: int, /) -> Index: ...
    def convert_angular(self, a: int, /) -> int: ...

def alleviate(
    arr: np.ndarray,
    label: np.ndarray,
    nrise: int,
    iterations: int,
    /,
) -> NDArray[np.float64]: ...
