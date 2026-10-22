from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import impy as ip
import numpy as np
import polars as pl
from acryo import Molecules, SubtomogramLoader
from acryo.backend import Backend
from acryo.loader._misc import lazy_average
from dask import array as da
from numpy.typing import NDArray

from cylindra.const import MoleculesHeader as Mole

if TYPE_CHECKING:
    from acryo import BackProjectionLoader

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
        dask_array = loader.construct_wedge_weighted_ft()
        averaged_images = da.compute(
            [
                lazy_average(
                    dask_array[sl],
                    output_shape=loader.output_shape,
                    backend=Backend(),
                    wedge_weighting=True,
                )
                for sl in labels
            ]
        )[0]
        averaged_images = ip.asarray(np.stack(averaged_images, axis=0), axes="pzyx")
        return averaged_images.set_scale(zyx=loader.scale, unit="nm")


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


@dataclass
class MoleculeScoreSeamSearchResult(SeamSearchResult):
    """Seam search result based on the score of each molecule.

    Attributes
    ----------
    scores : (2 * npf,) array
        Score of each seam position. ``scores[s + npf] == -scores[s]``.
    molecule_scores : (N,) array
        Score of each molecule. Larger value means that the molecule is more similar
        to the template (than to the anti-template, if given).
    ring, pf : (N,) arrays
        Ring and protofilament indices of each molecule used for labeling.
    tilt_scores : (N, T) array, optional
        Score of each molecule in each tilt image, if the molecules are scored in
        the tilt images.
    """

    scores: NDArray[np.floating]
    molecule_scores: NDArray[np.floating]
    ring: NDArray[np.int32]
    pf: NDArray[np.int32]
    tilt_scores: NDArray[np.floating] | None = None

    def get_label(self, size: int) -> NDArray[np.uint8]:
        if size != self.ring.size:
            raise ValueError(
                f"Seam search was performed on {self.ring.size} molecules, but labels "
                f"for {size} molecules were requested."
            )
        parity = _seam_parity(self.ring, self.pf, self.seam_pos, self.npf)
        return (~parity).astype(np.uint8)


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
    ) -> CorrelationSeamSearchResult:
        if mask is None:
            mask = 1

        masked_template = self._lowpass_filter(template, mask, cutoff)
        has_anti_template = anti_template is not None
        loader = loader.replace(output_shape=template.shape)
        averaged_images = self.calc_averages(loader)

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

        if has_anti_template:
            diff = corrs - anti_corrs
        else:
            diff = corrs
        # averaged_images[i] and averaged_images[i + npf] are averages of the
        # complementary sets of molecules, so both of them should be considered.
        score = diff - np.roll(diff, self.npf)

        return CorrelationSeamSearchResult(score, averaged_images, corrs, anti_corrs)

    @staticmethod
    def _lowpass_filter(
        img: ip.ImgArray, mask: NDArray[np.float32], cutoff: float
    ) -> ip.ImgArray:
        return (img * mask).lowpass_filter(cutoff, dims="zyx")


class TiltScoreSeamSearcher(SeamSearcher):
    """Seam searcher based on the scores of each molecule in the tilt images.

    The templates are projected to each tilt image at the position and orientation of
    each molecule, and compared with the tilt images by ZNCC. Unlike the subtomograms,
    the tilt images have no missing wedge or interpolation of the reconstruction.
    """

    def search(
        self,
        loader: BackProjectionLoader,
        template: NDArray[np.float32],
        anti_template: NDArray[np.float32] | None = None,
        mask: NDArray[np.float32] | None = None,
        cutoff: float | None = None,
        ctf: bool = True,
    ) -> MoleculeScoreSeamSearchResult:
        """Search for the seam position.

        Parameters
        ----------
        loader : BackProjectionLoader
            Loader of the molecules from the tilt series. Molecules should have the
            "nth" and "pf-id" features, otherwise they are assumed to be sorted by
            (nth, pf-id).
        template : 3D array
            Template image in the pixel size of the loader.
        anti_template : 3D array, optional
            Anti-template image. If given, the score of each molecule is the
            correlation with the template minus that with the anti-template.
        mask : 3D array, optional
            Mask image of the templates. It is projected in the same way as the
            templates and applied to both the projections and the tilt images.
        cutoff : float, optional
            Cutoff frequency of the low-pass filter, in the same unit as the alignment.
            None or 0 means no filter.
        ctf : bool, default True
            If true, the CTF is applied to the projections of the templates, with the
            defocus of each molecule in each tilt.
        """
        templates = [template]
        if anti_template is not None:
            if anti_template.shape != template.shape:
                raise ValueError(
                    f"The shape of anti-template ({anti_template.shape}) must be the "
                    f"same as the shape of template ({template.shape})."
                )
            templates.append(anti_template)
        # (n_templates, N, T) scores in each tilt image
        scores = loader.score_on_tilt(
            templates, mask=mask, cutoff=cutoff or None, ctf=ctf
        )
        if anti_template is None:
            tilt_scores = scores[0]
        else:
            tilt_scores = scores[0] - scores[1]
        mole_scores = tilt_scores.mean(axis=1, dtype=np.float64)
        result = self._search_by_molecule_scores(loader.molecules, mole_scores)
        result.tilt_scores = tilt_scores.astype(np.float32)
        return result

    def _search_by_molecule_scores(
        self, mole: Molecules, mole_scores: NDArray[np.float64]
    ) -> MoleculeScoreSeamSearchResult:
        """Compare the class means of the molecule scores of all the seam positions."""
        ring, pf = self._get_ring_and_pf(mole)
        # labels[s, i] is True if the i-th molecule is in the template-like class
        # under the seam position s.
        seam_pos = np.arange(2 * self.npf)[:, np.newaxis]
        labels = _seam_parity(ring, pf, seam_pos, self.npf)
        cnt_true = labels.sum(axis=1)
        cnt_false = mole_scores.size - cnt_true
        if np.any(cnt_true == 0) or np.any(cnt_false == 0):
            raise ValueError("Molecules in both even and odd rings are needed.")
        sum_true = (labels * mole_scores).sum(axis=1)
        sum_false = mole_scores.sum() - sum_true
        scores = sum_true / cnt_true - sum_false / cnt_false

        return MoleculeScoreSeamSearchResult(
            scores=scores.astype(np.float32),
            molecule_scores=mole_scores.astype(np.float32),
            ring=ring,
            pf=pf,
        )

    def _get_ring_and_pf(
        self, mole: Molecules
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        feat = mole.features
        if Mole.nth in feat.columns and Mole.pf in feat.columns:
            ring = feat[Mole.nth].cast(pl.Int32).to_numpy()
            pf = feat[Mole.pf].cast(pl.Int32).to_numpy()
            ring = ring - ring.min()
        else:
            _id = np.arange(mole.count(), dtype=np.int32)
            if _id.size % self.npf != 0:
                raise ValueError(
                    f"Number of molecules ({_id.size}) is not a multiple of npf "
                    f"({self.npf})."
                )
            ring, pf = np.divmod(_id, self.npf)
        if pf.min() < 0 or pf.max() >= self.npf:
            raise ValueError(
                f"Protofilament IDs must be in [0, {self.npf}), got "
                f"[{pf.min()}, {pf.max()}]."
            )
        return ring, pf


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


def _seam_parity(
    ring: NDArray[np.integer],
    pf: NDArray[np.integer],
    seam_pos: int | NDArray[np.integer],
    npf: int,
) -> NDArray[np.bool_]:
    """True if the molecule is in the template-like class under the seam position.

    This is consistent with ``SeamSearcher.label_with_seam`` if molecules are sorted
    by (ring, pf).
    """
    return (ring + (pf - seam_pos) // npf) % 2 == 0


def prep_seam_search_input(
    tomo: CylTomogram,
    mole: Molecules,
    npf: int | None,
    order: int,
) -> tuple[SubtomogramLoader, int]:
    """Prepare the subtomogram loaders for seam search.

    Returns
    -------
    tuple[SubtomogramLoader, int]
        The subtomogram loader for signal and the number of protofilaments.
    """
    loader_s = tomo.get_subtomogram_loader(mole, order=order)
    if npf is None:
        npf = mole.features[Mole.pf].unique().len()
    return loader_s, npf
