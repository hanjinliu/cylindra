from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import impy as ip
import numpy as np
import polars as pl
from acryo import Molecules, SubtomogramLoader
from dask import array as da
from numpy.typing import NDArray

from cylindra.const import MoleculesHeader as Mole
from cylindra.const import nm

if TYPE_CHECKING:
    from cylindra.components.tomogram import CylTomogram


class SeamSearcher(ABC):
    def __init__(self, npf: int):
        self._npf = npf

    @property
    def npf(self) -> int:
        return self._npf

    @abstractmethod
    def search(self, *args, **kwargs) -> SeamSearchResult:
        """Search for the seam position."""

    def label_with_seam(self, size: int) -> NDArray[np.bool_]:
        labels = list[NDArray[np.bool_]]()  # list of boolean arrays
        _id = np.arange(size)
        assert _id.size % self.npf == 0
        for pf in range(2 * self.npf):
            res = (_id - pf) // self.npf
            sl = res % 2 == 0
            labels.append(sl)
        return np.stack(labels, axis=0)

    def calc_averages(self, loader: SubtomogramLoader) -> ip.ImgArray:
        # prepare all the labels in advance
        labels = self.label_with_seam(loader.molecules.count())

        # here, dask_array is (N, Z, Y, X) array where dask_array[i] is i-th subtomogram.
        dask_array = loader.construct_dask()
        averaged_images = da.compute(
            [da.mean(dask_array[sl], axis=0) for sl in labels]
        )[0]
        return ip.asarray(np.stack(averaged_images, axis=0), axes="pzyx").set_scale(
            zyx=loader.scale
        )


@dataclass
class SeamSearchResult:
    scores: NDArray[np.floating]

    @property
    def seam_pos(self) -> int:
        return np.argmax(self.scores)

    @property
    def npf(self) -> int:
        return self.scores.size // 2

    def get_label(self, size: int) -> NDArray[np.uint8]:
        _id = np.arange(size)
        res = (_id - self.seam_pos) // self.npf
        # NOTE: value returned by SeamSearcher.label_with_seam is boolean array which
        # means that the true molecules match the seam position.
        return (res % 2 == 1).astype(np.uint8)

    def as_series(self, size: int) -> pl.Series:
        return pl.Series(Mole.isotype, self.get_label(size))

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({"scores": self.scores})


@dataclass
class CorrelationSeamSearchResult(SeamSearchResult):
    scores: NDArray[np.floating]
    averages: ip.ImgArray
    correlations: NDArray[np.floating]
    anti_correlations: NDArray[np.floating] | None

    def to_dataframe(self) -> pl.DataFrame:
        d = {
            "scores": self.scores,
            "correlations": self.correlations,
        }
        if self.anti_correlations is not None:
            d["anti_correlations"] = self.anti_correlations
        return pl.DataFrame(d)


class ManualSeamSearcher(SeamSearcher):
    def search(self, pos: int) -> SeamSearchResult:
        scores = np.zeros(self.npf * 2, dtype=np.float32)
        scores[pos] = 1.0
        return SeamSearchResult(scores)


class CorrelationSeamSearcher(SeamSearcher):
    """Seam searcher based on correlation with the template."""

    def search(
        self,
        loader: SubtomogramLoader,
        template: ip.ImgArray,
        anti_template: ip.ImgArray | None = None,
        mask: NDArray[np.float32] | None = None,
        cutoff: float = 0.5,
        loader_noise: SubtomogramLoader | None = None,
    ) -> CorrelationSeamSearchResult:
        if mask is None:
            mask = 1

        masked_template = self._lowpass_filter(template, mask, cutoff)
        has_anti_template = anti_template is not None
        has_noise_loader = loader_noise is not None
        loader = loader.replace(output_shape=template.shape)
        averaged_images = self.calc_averages(loader)

        if has_noise_loader:
            loader_noise = loader_noise.replace(output_shape=template.shape)
            noise_average_images = self.calc_averages(loader_noise)
        else:
            noise_average_images = None

        if has_anti_template:
            if anti_template.shape != template.shape:
                raise ValueError(
                    f"The shape of anti-template ({anti_template.shape}) must be the "
                    f"same as the shape of template ({template.shape})."
                )
            masked_anti_template = self._lowpass_filter(anti_template, mask, cutoff)
            anti_corrs = np.empty(averaged_images.shape[0], dtype=np.float32)
        else:
            masked_anti_template = None
            anti_corrs = None

        corrs = np.empty(averaged_images.shape[0], dtype=np.float32)
        for _i, avg in enumerate(averaged_images):
            avg: ip.ImgArray
            masked_avg = self._lowpass_filter(avg, mask, cutoff)
            corrs[_i] = ip.zncc(masked_avg, masked_template)
            if has_anti_template:
                anti_corrs[_i] = ip.zncc(masked_avg, masked_anti_template)

        if has_noise_loader:
            for _i, avg_noise in enumerate(noise_average_images):
                avg_noise: ip.ImgArray
                masked_avg_noise = self._lowpass_filter(avg_noise, mask, cutoff)
                corrs[_i] -= ip.zncc(masked_avg_noise, masked_template)
                if has_anti_template:
                    anti_corrs[_i] -= ip.zncc(masked_avg_noise, masked_anti_template)

        if has_anti_template:
            score = corrs - anti_corrs
        else:
            corr1, corr2 = corrs[: self.npf], corrs[self.npf :]
            score = np.empty_like(corrs, dtype=np.float32)
            score[: self.npf] = corr1 - corr2
            score[self.npf :] = corr2 - corr1

        return CorrelationSeamSearchResult(score, averaged_images, corrs, anti_corrs)

    @staticmethod
    def _lowpass_filter(
        img: ip.ImgArray, mask: NDArray[np.float32], cutoff: float
    ) -> ip.ImgArray:
        return (img * mask).lowpass_filter(cutoff, dims="zyx")


class BooleanSeamSearcher(SeamSearcher):
    def search(self, label: np.ndarray) -> SeamSearchResult:
        label = np.asarray(label)
        nmole = label.size
        unique_values = np.unique(label)
        if len(unique_values) != 2:
            raise ValueError(
                f"Label must have exactly two unique values, but got {unique_values}"
            )

        bin_label = self._binarize(label == unique_values[0])

        each_label = self.label_with_seam(nmole)
        scores = list[int]()
        for pf in range(self.npf):
            sl = self._binarize(each_label[pf])
            score = abs(np.sum(bin_label * sl))
            scores.append(score)
        return SeamSearchResult(np.array(scores))

    @staticmethod
    def _binarize(x: NDArray[np.bool_]) -> NDArray[np.int8]:
        return np.where(x, 1, -1)


def prep_seam_search_input(
    tomo: CylTomogram,
    mole: Molecules,
    npf: int | None,
    order: int,
    subtract_noise_by: Literal["none", "halves", "shift"] = "none",
    shift_to_background: tuple[nm, nm, nm] = (0, 0, 0),
) -> tuple[SubtomogramLoader, SubtomogramLoader | None, int]:
    if subtract_noise_by == "halves":
        loader_s, loader_n = tomo.get_subtomogram_and_noise_loader(mole, order=order)
    else:
        loader_s = tomo.get_subtomogram_loader(mole, order=order)
        if subtract_noise_by == "none":
            loader_n = None
        else:
            loader_n = loader_s.replace(
                molecules=mole.translate_internal(shift_to_background)
            )
    if npf is None:
        npf = mole.features[Mole.pf].unique().len()
    return loader_s, loader_n, npf
