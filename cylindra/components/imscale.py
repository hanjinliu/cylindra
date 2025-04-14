from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import impy as ip
import numpy as np
from numpy.typing import NDArray


class ScaleOptimizerBase(ABC):
    """The base class for optimizing the image scale."""

    @abstractmethod
    def fit(self, img: ip.ImgArray, img_ref: ip.ImgArray) -> OptimizationResult: ...


class ScaleOptimizer(ScaleOptimizerBase):
    def __init__(
        self,
        zoom_min: float,
        zoom_max: float,
        *,
        num: int = 11,
        precision: float = 0.001,
    ) -> None:
        self._zoom_min = zoom_min
        self._zoom_max = zoom_max
        self._num = num
        self._precision = precision

    def fit(
        self,
        img: ip.ImgArray,
        img_ref: ip.ImgArray,
        mask: NDArray[np.float32] | None = None,
    ) -> OptimizationResult:
        scales = []
        scores = []
        img_ref_normed = img_ref - np.mean(img_ref)
        zmin, zmax = self._zoom_min, self._zoom_max
        diff = (zmax - zmin) / (self._num - 1)
        if mask is None:
            mask = 1
        while img.scale.x * diff > self._precision:
            cur_scores = []
            cur_factors = np.linspace(zmin, zmax, self._num)
            for zoom in cur_factors:
                imgold_zoomed = img.zoom(zoom, same_shape=True, mode="reflect") * mask
                img_zoomed_normed = imgold_zoomed - np.mean(imgold_zoomed)
                cur_scores.append(ip.ncc(img_zoomed_normed, img_ref_normed))
            scales.extend(img.scale.x / cur_factors)
            scores.extend(cur_scores)
            opt_factor = cur_factors[np.argmax(cur_scores)]
            diff = (zmax - zmin) / (self._num - 1)
            zmin = opt_factor - diff
            zmax = opt_factor + diff
        _scales = np.array(scales)
        _scores = np.array(scores)
        _order = np.argsort(_scales)
        return OptimizationResult(_scales[_order], _scores[_order])


class OptimizationResult(NamedTuple):
    scales: NDArray[np.float32]
    scores: NDArray[np.float32]

    @property
    def scale_optimal(self) -> float:
        return self.scales[np.argmax(self.scores)]

    @property
    def score_optimal(self) -> float:
        return np.max(self.scores)
