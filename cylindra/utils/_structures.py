from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import polars as pl
import impy as ip
from acryo import Molecules

from cylindra.const import (
    Mode,
    MoleculesHeader as Mole,
    GlobalVariables as GVar,
    PropertyNames as H,
)

from ._correlation import mirror_zncc

if TYPE_CHECKING:
    from cylindra.components import CylSpline


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
    corrs = list[float]()
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

    spl = CylSpline(degree=GVar.spline_degree)
    all_coords = _reshaped_positions(mole)
    mean_coords = np.mean(all_coords, axis=1)  # (N, ndim)
    return spl.fit(mean_coords, weight_ramp=(50, 0.5))


def _reshaped_positions(mole: Molecules) -> NDArray[np.float32]:
    try:
        pf_label = mole.features[Mole.pf]
        pos_list = list[NDArray[np.float32]]()  # each shape: (y, ndim)
        for pf in range(pf_label.max() + 1):
            pos_list.append(mole.pos[pf_label == pf])
        pos = np.stack(pos_list, axis=1)  # shape: (y, pf, ndim)

    except Exception as e:
        raise TypeError(
            f"Reshaping failed. Molecules must be correctly labeled at {Mole.pf!r} "
            f"feature. Original error is\n{type(e).__name__}: {e}"
        ) from e
    return pos


def infer_geometry_from_molecules(mole: Molecules) -> tuple[int, int, int]:
    """Infer cylinder geometry (ny, npf, nrise) from molecules."""
    columns = mole.features.columns
    if not (Mole.pf in columns and Mole.position in columns):
        raise ValueError(
            f"Molecules must have columns {Mole.pf!r} and {Mole.position!r}."
        )
    npf = mole.features[Mole.pf].max() + 1
    nmole = mole.pos.shape[0]
    ny, res = divmod(nmole, npf)
    if res != 0:
        raise ValueError("Molecules are not correctly labeled.")
    spl_pos = mole.features[Mole.position].to_numpy().reshape(ny, npf)
    dy = np.abs(np.mean(np.diff(spl_pos, axis=0)))
    drise = np.mean(np.diff(spl_pos, axis=1))
    nrise = -int(np.round(drise * npf / dy)) * GVar.rise_sign
    return ny, npf, nrise
