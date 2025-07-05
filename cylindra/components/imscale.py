from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import impy as ip
import numpy as np
from numpy.typing import NDArray

from cylindra.const import nm
from cylindra.utils import fit_to_shape


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
        mask: ip.ImgArray | None = None,
        freq_min: nm = 0.5,
        freq_max: nm = 100,
    ) -> OptimizationResult:
        scales = []
        scores = []
        if img.scale.x > img_ref.scale.x:
            img_ref = img_ref.zoom(img_ref.scale.x / img.scale.x, mode="reflect")
        elif img.scale.x < img_ref.scale.x:
            img = img.zoom(img.scale.x / img_ref.scale.x, mode="reflect")
        if mask is None:
            mask = 1
        else:
            if mask.scale.x != img.scale.x:
                mask = mask.zoom(mask.scale.x / img.scale.x, mode="reflect")
            mask = fit_to_shape(mask, img.shape)
        img_ref = fit_to_shape(img_ref, img.shape)
        img_ref_normed = (img_ref - np.mean(img_ref)) * mask
        zmin, zmax = self._zoom_min, self._zoom_max
        diff = (zmax - zmin) / (self._num - 1)

        fmax = img.scale.x / freq_min if freq_min > 0 else np.inf
        fmin = img.scale.x / freq_max

        while img.scale.x * diff > self._precision:
            cur_scores = []
            cur_factors = np.linspace(zmin, zmax, self._num)
            for zoom in cur_factors:
                imgold_zoomed = img.zoom(zoom, same_shape=True, mode="reflect")
                img_zoomed_normed = (imgold_zoomed - np.mean(imgold_zoomed)) * mask
                cur_scores.append(
                    fsc_mean(img_zoomed_normed, img_ref_normed, fmin, fmax)
                )
            scales.extend(img.scale.x * cur_factors)
            scores.extend(cur_scores)
            opt_factor = cur_factors[np.argmax(cur_scores)]
            diff = (zmax - zmin) / (self._num - 1)
            zmin = opt_factor - diff
            zmax = opt_factor + diff
        _scales = np.array(scales)
        _scores = np.array(scores)
        _order = np.argsort(_scales)
        return OptimizationResult(_scales[_order], _scores[_order])


def fsc_mean(img0: ip.ImgArray, img1: ip.ImgArray, freq_min: nm, freq_max: nm):
    """Calculate the mean FSC value within the specified frequency range."""
    freq, fsc = ip.fsc(img0, img1, dfreq=np.sqrt(3) / img0.shape[0])
    freq_range = (freq_min <= freq) & (freq <= freq_max)
    fsc_of_interest = fsc[freq_range]
    if fsc_of_interest.size == 0:
        raise ValueError(
            f"No frequencies found in the range {freq_min=}, {freq_max=}. Returned "
            f"frequency was {freq!r}. Please adjust the frequency range."
        )
    return np.max(fsc_of_interest)


class OptimizationResult(NamedTuple):
    scales: NDArray[np.float32]
    scores: NDArray[np.float32]

    @property
    def scale_optimal(self) -> float:
        return self.scales[np.argmax(self.scores)]

    @property
    def score_optimal(self) -> float:
        return np.max(self.scores)
