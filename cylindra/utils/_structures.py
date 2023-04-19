from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import polars as pl
import impy as ip
from dask import array as da
from acryo import Molecules, SubtomogramLoader

from cylindra.const import (
    Mode,
    MoleculesHeader as Mole,
    GlobalVariables as GVar,
    PropertyNames as H,
)

from ._correlation import mirror_zncc

if TYPE_CHECKING:
    from cylindra.components import CylSpline


def try_all_seams(
    loader: SubtomogramLoader,
    npf: int,
    template: ip.ImgArray,
    mask: NDArray[np.float32] | None = None,
    cutoff: float = 0.5,
) -> tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]:
    """
    Try all the possible seam positions and compare correlations.

    Parameters
    ----------
    loader : SubtomogramLoader
        An aligned ``acryo.SubtomogramLoader`` object.
    npf : int
        Number of protofilament.
    template : ip.ImgArray
        Template image.
    mask : ip.ImgArray, optional
        Mask image.
    cutoff : float, default is 0.5
        Cutoff frequency applied before calculating correlations.

    Returns
    -------
    tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]
        Correlation, average and boolean array correspond to each seam position.
    """
    corrs: list[float] = []
    labels: list[np.ndarray] = []  # list of boolean arrays

    if mask is None:
        mask = 1

    masked_template = (template * mask).lowpass_filter(cutoff=cutoff, dims="zyx")
    _id = np.arange(len(loader.molecules))
    assert _id.size % npf == 0

    # prepare all the labels in advance (only takes up ~0.5 MB at most)
    for pf in range(2 * npf):
        res = (_id - pf) // npf
        sl = res % 2 == 0
        labels.append(sl)

    # here, dask_array is (N, Z, Y, X) array where dask_array[i] is i-th subtomogram.
    dask_array = loader.construct_dask(output_shape=template.shape)
    averaged_images = da.compute([da.mean(dask_array[sl], axis=0) for sl in labels])[0]
    averaged_images = ip.asarray(np.stack(averaged_images, axis=0), axes="pzyx")
    averaged_images.set_scale(zyx=loader.scale)

    corrs: list[float] = []
    for avg in averaged_images:
        avg: ip.ImgArray
        corr = ip.zncc((avg * mask).lowpass_filter(cutoff=cutoff), masked_template)
        corrs.append(corr)

    return np.array(corrs), averaged_images, labels


def centering(
    img: ip.ImgArray,
    point: np.ndarray,
    angle: float,
    drot: float = 5,
    nrots: int = 7,
    max_shifts: float | None = None,
):
    """
    Find the center of cylinder using self-correlation.

    Parameters
    ----------
    img : ip.ImgArray
        Target image.
    point : np.ndarray
        Current center of cylinder.
    angle : float
        The central angle of the cylinder.
    drot : float, default is 5
        Deviation of the rotation angle.
    nrots : int, default is 7
        Number of rotations to try.
    max_shifts : float, optional
        Maximum shift in pixel.

    """
    angle_deg2 = angle_corr(img, ang_center=angle, drot=drot, nrots=nrots)

    img_next_rot = img.rotate(-angle_deg2, cval=np.mean(img))
    proj = img_next_rot.proj("y")
    shift = mirror_zncc(proj, max_shifts=max_shifts)

    shiftz, shiftx = shift / 2
    shift = np.array([shiftz, 0, shiftx])
    rad = -np.deg2rad(angle_deg2)
    cos = np.cos(rad)
    sin = np.sin(rad)
    shift = shift @ [[1.0, 0.0, 0.0], [0.0, cos, sin], [0.0, -sin, cos]]
    point += shift
    return point


def angle_corr(
    img: ip.ImgArray, ang_center: float = 0, drot: float = 7, nrots: int = 29
):
    # img: 3D
    img_z = img.proj("z")
    mask = ip.circular_mask(img_z.shape.y / 2 + 2, img_z.shape)
    img_mirror: ip.ImgArray = img_z["x=::-1"]
    angs = np.linspace(ang_center - drot, ang_center + drot, nrots, endpoint=True)
    corrs = []
    f0 = np.sqrt(img_z.power_spectra(dims="yx", zero_norm=True))
    cval = np.mean(img_z)
    for ang in angs:
        img_mirror_rot = img_mirror.rotate(ang * 2, mode=Mode.constant, cval=cval)
        f1 = np.sqrt(img_mirror_rot.power_spectra(dims="yx", zero_norm=True))
        corr = ip.zncc(f0, f1, mask)
        corrs.append(corr)

    angle = angs[np.argmax(corrs)]
    return angle


def molecules_to_spline(mole: Molecules):
    """Convert well aligned molecule positions into a spline."""
    from cylindra.components import CylSpline

    spl = CylSpline(degree=GVar.splOrder)
    all_coords = _reshaped_positions(mole)
    mean_coords = np.mean(all_coords, axis=1)
    spl.fit_coa(mean_coords, min_radius=GVar.minCurvatureRadius)
    return spl


def _reshaped_positions(mole: Molecules) -> NDArray[np.float32]:
    try:
        pf_label = mole.features[Mole.pf]
        pos_list: list[NDArray[np.float32]] = []  # each shape: (y, ndim)
        for pf in range(pf_label.max() + 1):
            pos_list.append(mole.pos[pf_label == pf])
        pos = np.stack(pos_list, axis=1)  # shape: (y, pf, ndim)

    except Exception as e:
        raise TypeError(
            f"Reshaping failed. Molecules must be correctly labeled at {Mole.pf!r} "
            f"feature. Original error is\n{type(e).__name__}: {e}"
        ) from e
    return pos


def with_interval(mole: Molecules, spl: CylSpline) -> pl.DataFrame:
    _index_column_key = "__index_column"
    mole0 = mole.with_features([pl.arange(0, pl.count()).alias(_index_column_key)])
    _spl_len = spl.length()
    _subsets: list[Molecules] = []
    for _, sub in mole0.groupby(Mole.pf):
        _pos = sub.pos
        _interv_vec = np.diff(_pos, axis=0, append=0)
        _u = sub.features[Mole.position] / _spl_len
        _spl_vec = spl(_u, der=1)
        _spl_vec_len = np.linalg.norm(_spl_vec, axis=1)  # lengths of spline vectors
        _spl_vec_norm = _spl_vec / _spl_vec_len[:, np.newaxis]
        _y_interv = np.sum(
            _interv_vec * _spl_vec_norm, axis=1
        )  # projection by inner product
        _y_interv[-1] = -1.0  # fill invalid values with -1
        _subsets.append(sub.with_features(pl.Series(Mole.interval, _y_interv)))
    return (
        Molecules.concat(_subsets)
        .sort(_index_column_key)
        .drop_features(_index_column_key)
        .features
    )


def with_skew(mole: Molecules, spl: CylSpline) -> pl.DataFrame:
    _index_column_key = "__index_column"
    mole0 = mole.with_features([pl.arange(0, pl.count()).alias(_index_column_key)])
    _spl_len = spl.length()
    _subsets: list[Molecules] = []
    for _, sub in mole0.groupby(Mole.pf):
        _pos = sub.pos
        _interv_vec = np.diff(_pos, axis=0, append=0)
        _interv_vec_len = np.linalg.norm(_interv_vec, axis=1)
        _interv_vec_norm = _interv_vec / _interv_vec_len[:, np.newaxis]

        _u = sub.features[Mole.position] / _spl_len
        _spl_pos = spl(_u, der=0)
        _spl_vec = spl(_u, der=1)

        _mole_to_spl_vec = _spl_pos - _pos
        _radius = np.linalg.norm(_mole_to_spl_vec, axis=1)

        _spl_vec_len = np.linalg.norm(_spl_vec, axis=1)  # lengths of spline vectors
        _spl_vec_norm = _spl_vec / _spl_vec_len[:, np.newaxis]

        _skew_cross = np.cross(_interv_vec_norm, _spl_vec_norm, axis=1)  # cross product
        _inner = np.sum(_skew_cross * _mole_to_spl_vec, axis=1)
        _skew_sin = np.linalg.norm(_skew_cross, axis=1) * np.sign(_inner)

        _skew = np.rad2deg(2 * _interv_vec_len * _skew_sin / _radius)
        _skew = 0
        _subsets.append(sub.with_features(pl.Series(Mole.skew, _skew)))
    return (
        Molecules.concat(_subsets)
        .sort(_index_column_key)
        .drop_features(_index_column_key)
        .features
    )


def infer_seam_from_labels(label: np.ndarray, npf: int) -> int:
    label = np.asarray(label)
    nmole = label.size
    unique_values = np.unique(label)
    if len(unique_values) != 2:
        raise ValueError(
            f"Label must have exactly two unique values, but got {unique_values}"
        )

    def _binarize(x: NDArray[np.bool_]) -> NDArray[np.int8]:
        return np.where(x, 1, -1)

    bin_label = _binarize(label == unique_values[0])

    _id = np.arange(nmole)
    assert _id.size % npf == 0

    scores: list[int] = []
    for pf in range(npf):
        res = (_id - pf) // npf
        sl = _binarize(res % 2 == 0)
        score = abs(np.sum(bin_label * sl))
        scores.append(score)
    return np.argmax(scores)
