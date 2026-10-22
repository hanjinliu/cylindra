from __future__ import annotations

from typing import TYPE_CHECKING

import impy as ip
import numpy as np
from numpy.typing import NDArray

from cylindra import cylmeasure
from cylindra.const import nm

if TYPE_CHECKING:
    from cylindra.components.cyl_spline import CylSpline
    from cylindra.components.molecules import Molecules


def make_cylindrical_mask(
    spl: CylSpline,
    mole: Molecules,
    shape: tuple[int, int, int],
    scale: nm,
) -> NDArray[np.float32]:
    _inner = spl.config.thickness_inner / scale
    _outer = spl.config.thickness_outer / scale
    if spl.radius is None:
        _radius_nm: nm = cylmeasure.calc_radius(mole, spl).mean()
    else:
        _radius_nm = spl.radius
    _radius = _radius_nm / scale
    zz, xx = np.indices((shape[0], shape[2]), dtype=np.float32)
    xx -= (shape[2] - 1) / 2
    zz -= (shape[0] - 1) / 2 - _radius
    dd2 = zz**2 + xx**2
    _cyl_mask_2d = (max(_radius - _inner, 0) ** 2 < dd2) & (
        dd2 < (_radius + _outer) ** 2
    )
    mask_binary = ip.asarray(
        np.repeat(_cyl_mask_2d[:, np.newaxis, :], shape[1], axis=1), axes="zyx"
    )
    if scale < 0.55:
        mask_binary = mask_binary.smooth_mask(
            sigma=1 / scale, dilate_radius=0.5 / scale
        )
    else:
        mask_binary = mask_binary.smooth_mask(sigma=0.8, dilate_radius=1)
    return mask_binary.value


def make_tube_template(spl: CylSpline, length: nm, scale: nm) -> NDArray[np.float32]:
    """Bright tube along the y axis, of the radius and the wall of the spline.

    The edges of the wall are blurred by 1 nm, and the ends of the tube are tapered by
    cosine over a quarter of the length.
    """
    from scipy.special import erf

    assert spl.radius is not None
    r_inner = spl.radius - spl.config.thickness_inner
    r_outer = spl.radius + spl.config.thickness_outer
    half_d = int(np.ceil((r_outer + 2.0) / scale))
    half_l = int(np.ceil(length / 2 / scale))
    shape = (2 * half_d + 1, 2 * half_l + 1, 2 * half_d + 1)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    rho = np.hypot(zz - half_d, xx - half_d) * scale
    blur = np.sqrt(2) * 1.0
    wall = (erf((rho - r_inner) / blur) - erf((rho - r_outer) / blur)) / 2
    edge = np.clip((length / 2 - np.abs(yy - half_l) * scale) / (length / 4), 0, 1)
    taper = (1 - np.cos(np.pi * edge)) / 2
    return (wall * taper).astype(np.float32)
