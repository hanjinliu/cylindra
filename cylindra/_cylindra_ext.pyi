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
        angle_max: float | None = None,
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
    def simulate(self, niter: int = 10000) -> None: ...
    def cool_completely(self) -> None: ...
    def iteration(self) -> int: ...
    def with_seed(self, seed: int) -> CylindricAnnealingModel: ...
    def with_reject_limit(self, reject_limit: int) -> CylindricAnnealingModel: ...
    def optimization_state(self) -> str: ...
    def time_constant(self) -> float: ...
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
        lon_ang_max: float,
        cooling_rate: float = 1e-3,
    ) -> CylindricAnnealingModel: ...
    def shifts(self) -> NDArray[np.int32]: ...
    def set_shifts(self, shifts: NDArray[np.int32]) -> CylindricAnnealingModel: ...
    def local_shape(self) -> tuple[int, int, int]: ...
    def longitudinal_distances(self) -> NDArray[np.float32]: ...
    def lateral_distances(self) -> NDArray[np.float32]: ...
    def longitudinal_angles(self) -> NDArray[np.float32]: ...
    def lateral_angles(self) -> NDArray[np.float32]: ...
    def get_edge_info(self) -> tuple[np.float32, np.float32, np.int32]: ...
    def node_count(self) -> int: ...
    def init_shift_random(self) -> None: ...
    def binding_energies(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...

class CylindricArray:
    def __init__(
        nth: NDArray[np.int32],
        npf: NDArray[np.int32],
        values: NDArray[np.float32],
        nrise: int,
    ): ...
    def nrise(self) -> int: ...
    def asarray(self) -> NDArray[np.float32]: ...
    def as1d(self) -> NDArray[np.float32]: ...
    def with_values(self, values: NDArray[np.float32]) -> CylindricArray: ...
    def convolve(self, weight: NDArray[np.float32]) -> CylindricArray: ...
    def mean_filter(self, footprint: NDArray[np.bool_]) -> CylindricArray: ...
    def max_filter(self, footprint: NDArray[np.bool_]) -> CylindricArray: ...
    def min_filter(self, footprint: NDArray[np.bool_]) -> CylindricArray: ...
    def median_filter(self, footprint: NDArray[np.bool_]) -> CylindricArray: ...
    def label(self) -> CylindricArray: ...

def alleviate(
    arr: NDArray[np.float64],
    label: NDArray[np.int32],
    nrise: int,
) -> NDArray[np.float64]: ...
def oblique_coordinates(
    coords: NDArray[np.int32],
    tilts: tuple[float, float],
    intervals: tuple[float, float],
    offsets: tuple[float, float],
) -> NDArray[np.float32]: ...

class RegionProfiler:
    def __init__(
        self,
        image: NDArray[np.float32],
        label_image: NDArray[np.uint32],
        nrise: int,
    ) -> None: ...
    @classmethod
    def from_arrays(
        cls,
        image: NDArray[np.float32],
        label_image: NDArray[np.uint32],
        nrise: int,
    ) -> RegionProfiler: ...
    @classmethod
    def from_features(
        cls,
        nth: NDArray[np.int32],
        npf: NDArray[np.int32],
        values: NDArray[np.float32],
        labels: NDArray[np.uint32],
        per: int,
        nrise: int,
    ) -> RegionProfiler: ...
    def calculate(self, props: list[str]) -> dict[str, NDArray[np.float32]]: ...
