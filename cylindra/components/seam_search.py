from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import impy as ip
from dask import array as da
from acryo import SubtomogramLoader


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
    averages: ip.ImgArray | None = None

    @property
    def seam_pos(self) -> int:
        return np.argmax(self.scores)

    @property
    def npf(self) -> int:
        return self.scores.size // 2

    def get_label(self, size: int) -> NDArray[np.uint8]:
        _id = np.arange(size)
        res = (_id - self.seam_pos) // self.npf
        return (res % 2 == 0).astype(np.uint8)


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
    ) -> SeamSearchResult:
        corrs = list[float]()

        if mask is None:
            mask = 1

        masked_template = (template * mask).lowpass_filter(cutoff, dims="zyx")
        has_anti_template = anti_template is not None
        if has_anti_template:
            if anti_template.shape != template.shape:
                raise ValueError(
                    f"The shape of anti-template ({anti_template.shape}) must be the "
                    f"same as the shape of template ({template.shape})."
                )
            masked_anti_template = (anti_template * mask).lowpass_filter(
                cutoff, dims="zyx"
            )
        else:
            masked_anti_template = None

        averaged_images = self.calc_averages(
            loader.replace(output_shape=template.shape)
        )

        corrs = np.empty(averaged_images.shape[0], dtype=np.float32)
        anti_corrs = np.empty(averaged_images.shape[0], dtype=np.float32)
        for _i, avg in enumerate(averaged_images):
            avg: ip.ImgArray
            masked_avg = (avg * mask).lowpass_filter(cutoff=cutoff, dims="zyx")
            corrs[_i] = ip.zncc(masked_avg, masked_template)
            if has_anti_template:
                anti_corrs[_i] = ip.zncc(masked_avg, masked_anti_template)

        if has_anti_template:
            score = corrs - anti_corrs
        else:
            corr1, corr2 = corrs[: self.npf], corrs[self.npf :]
            score = np.empty_like(corrs, dtype=np.float32)
            score[: self.npf] = corr1 - corr2
            score[self.npf :] = corr2 - corr1

        return SeamSearchResult(score, averaged_images)


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


class FiducialSeamSearcher(SeamSearcher):
    def search(
        self,
        loader: SubtomogramLoader,
        weight: NDArray[np.float32],
    ) -> SeamSearchResult:
        averaged_images = self.calc_averages(loader.replace(output_shape=weight.shape))

        scores = list[float]()
        for img in averaged_images:
            scores.append(np.mean(img * weight))
        return SeamSearchResult(np.array(scores), averaged_images)
