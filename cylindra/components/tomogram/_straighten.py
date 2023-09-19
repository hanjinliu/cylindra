from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dask import array as da, delayed

import impy as ip

from cylindra.components.spline import CylSpline
from cylindra.components._ftprops import get_polar_image
from cylindra.const import nm
from cylindra.utils import map_coordinates, ceilint


def straighten(
    img: ip.ImgArray | ip.LazyImgArray,
    spl: CylSpline,
    range_: tuple[float, float],
    size: tuple[nm, nm] | nm,
    chunk_length: nm = 72,
):
    if size is None:
        rz = rx = 2 * (spl.radius + spl.config.thickness_outer)

    else:
        if isinstance(size, tuple):
            rz, rx = size
        elif hasattr(size, "__iter__"):
            rz, rx = size
        else:
            rz = rx = size

    length = spl.length(*range_)
    scale = img.scale.x
    coords = spl.cartesian(shape=(rz, rx), s_range=range_, scale=scale)
    nchunks = ceilint(length / chunk_length)
    if nchunks == 1:
        transformed = map_coordinates(img, coords)
    else:
        delayed_map_coordinates = delayed(map_coordinates)
        each_coords = _chunk_coords(coords, axis=1, nchunks=nchunks)
        tasks = [delayed_map_coordinates(img, crds) for crds in each_coords]
        transformed = np.concatenate(da.compute(*tasks), axis=1)
    axes = "zyx"
    out = ip.asarray(transformed, axes=axes)
    return out.set_scale({k: scale for k in axes}, unit="nm")


def straighten_cylindric(
    img: ip.ImgArray | ip.LazyImgArray,
    spl: CylSpline,
    range_: tuple[float, float],
    radii: tuple[float, float] | None,
    chunk_length: nm = 72,
):
    if spl.radius is None:
        raise ValueError("Radius has not been determined yet.")

    _scale = img.scale.x
    if radii is None:
        rmin, rmax = spl.radius_range()
    else:
        rmin, rmax = radii

    if rmax <= rmin:
        raise ValueError("radii[0] < radii[1] must be satisfied.")
    coords = spl.cylindrical(
        r_range=(rmin, rmax),
        s_range=range_,
        scale=_scale,
    )
    rc = (rmin + rmax) / 2
    length = spl.length(*range_)
    if length < chunk_length:
        transformed = get_polar_image(img, coords, radius=rc)
    else:
        delayed_get_polar_image = delayed(get_polar_image)
        nchunks = ceilint(length / chunk_length)
        each_coords = _chunk_coords(coords, axis=1, nchunks=nchunks)
        tasks = [delayed_get_polar_image(img, crds, rc) for crds in each_coords]
        transformed = np.concatenate(da.compute(*tasks), axis=1)
    return transformed


def _chunk_coords(
    coords: NDArray[np.float32],
    axis: int,
    nchunks: int,
) -> list[NDArray[np.float32]]:
    # NOTE: the first axis is the dimension axis
    axis_len = coords.shape[axis + 1]
    s0 = ceilint(axis_len / nchunks)
    indices = np.linspace(s0, axis_len, nchunks, dtype=np.int32, endpoint=False)
    return np.split(coords, indices, axis=axis + 1)
