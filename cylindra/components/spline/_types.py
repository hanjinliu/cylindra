from __future__ import annotations

from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray


class Coords3D(TypedDict):
    """3D coordinates in list used in json."""

    z: list[float]
    y: list[float]
    x: list[float]


class SplineInfo(TypedDict, total=False):
    """Spline parameters used in json."""

    t: list[float]
    c: Coords3D
    k: int
    u: list[float]
    lims: tuple[float, float]
    localprops_window_size: dict[str, float]
    extrapolate: str
    config: dict[str, Any]


TCKType = tuple["NDArray[np.float32] | None", "NDArray[np.float32] | None", int]
PrepOutput = tuple[TCKType, NDArray[np.float32]]


class SplineFitResult(NamedTuple):
    params: PrepOutput
    curvature: float
    residuals: NDArray[np.float32]
    success: bool
