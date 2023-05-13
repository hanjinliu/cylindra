from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

class ViterbiGrid:
    def __init__(
        self,
        score_array: NDArray[np.float32],
        origin: NDArray[np.float32],
        zvec: NDArray[np.float32],
        yvec: NDArray[np.float32],
        xvec: NDArray[np.float32],
    ) -> None: ...
    def viterbi(
        self,
        dist_min: float,
        dist_max: float,
        skew_max: float | None = None,
        /,
    ) -> tuple[NDArray[np.int32], float]: ...
    def world_pos(self, n: int, z: int, y: int, x: int, /) -> NDArray[np.float64]: ...

class Index:
    y: int
    a: int
    def __init__(self, y: int, a: int, /) -> None: ...
    def is_valid(self) -> bool: ...

class CylinderGeometry:
    def __init__(self, ny: int, na: int, nrise: int, /) -> None: ...
    def get_index(self, y: int, a: int, /) -> Index: ...

class CylindricAnnealingModel:
    def __init__(self, seed: int = 0) -> None: ...
    def simulate(self, niter: int = 10000, nsettle: int = 1000) -> None: ...
    def iteration(self) -> int: ...
    def with_seed(self, seed: int) -> CylindricAnnealingModel: ...
    def with_reject_limit(self, reject_limit: int) -> CylindricAnnealingModel: ...
    def optimization_state(self) -> str: ...
    def energy(self) -> float: ...
    def temperature(self) -> float: ...
    def set_reservoir(
        self, temperature: float, time_constant: float, min_temperature: float = 0.0
    ) -> CylindricAnnealingModel: ...
    def construct_graph(
        self, indices: NDArray[np.int32], npf: int, nrise: int
    ) -> CylindricAnnealingModel: ...
    def set_graph_coordinates(
        self,
        origin: NDArray[np.float32],
        zvec: NDArray[np.float32],
        yvec: NDArray[np.float32],
        xvec: NDArray[np.float32],
    ) -> CylindricAnnealingModel: ...
    def set_energy_landscape(
        self, energy: NDArray[np.float32]
    ) -> CylindricAnnealingModel: ...
    def set_box_potential(
        self,
        lon_dist_min: float,
        lon_dist_max: float,
        lat_dist_min: float,
        lat_dist_max: float,
    ) -> CylindricAnnealingModel: ...
    def shifts(self) -> NDArray[np.int32]: ...
    def longitudinal_distances(self) -> NDArray[np.float32]: ...
    def lateral_distances(self) -> NDArray[np.float32]: ...
    def get_edge_info(self) -> tuple[np.float32, np.float32, np.int32]: ...

def alleviate(
    arr: NDArray[np.float64],
    label: NDArray[np.int32],
    nrise: int,
) -> NDArray[np.float64]: ...
