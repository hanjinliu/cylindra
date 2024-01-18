from __future__ import annotations

from functools import lru_cache

import impy as ip
import numpy as np
from numpy.typing import NDArray

from cylindra._dask import Delayed, delayed
from cylindra.const import Mode, nm
from cylindra.utils import map_coordinates, map_coordinates_task, roundint


@lru_cache(maxsize=12)
def polar_coords_2d(
    r_start: float, r_stop: float, center=(0, 0)
) -> NDArray[np.float32]:
    n_angle = roundint((r_start + r_stop) * np.pi)
    n_radius = roundint(r_stop - r_start)
    r_, ang_ = np.indices((n_radius, n_angle))
    r_ = r_ + (r_start + r_stop - n_radius + 1) / 2
    # NOTE: r_.mean() is always (r_start + r_stop) / 2
    output_coords = np.column_stack([r_.ravel(), ang_.ravel()])
    coords = _linear_polar_mapping(
        np.array(output_coords),
        k_angle=n_angle / 2 / np.pi,
        k_radius=1,
        center=center[::-1],
    ).astype(np.float32)
    coords = coords.reshape(n_radius, n_angle, 2)  # V, H, 2

    # Here, the first coordinate should be theta=0, and theta moves anti-clockwise
    coords[:] = np.flip(coords, axis=2)  # flip y, x
    return coords


def get_polar_image(
    img: ip.ImgArray | ip.LazyImgArray,
    coords: NDArray[np.float32],
    radius: nm,
    order: int = 3,
):
    """Convert the input image into a polar image."""
    polar = map_coordinates(img, coords, order=order, mode=Mode.constant, cval=np.mean)
    return _post_process_polar_image(polar, img, radius).compute()


@delayed
def _post_process_polar_image(polar: ip.ImgArray, img: ip.ImgArray, radius: nm):
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    a_scale = 2 * np.pi * radius / polar.shape.a
    return polar.set_scale(r=img.scale.x, y=img.scale.x, a=a_scale, unit=img.scale_unit)


def get_polar_image_task(
    img: ip.ImgArray | ip.LazyImgArray,
    coords: NDArray[np.float32],
    radius: nm,
    order: int = 3,
) -> Delayed[ip.ImgArray]:
    """Convert the input image into a polar image."""
    polar = map_coordinates_task(
        img, coords, order=order, mode=Mode.constant, cval=np.mean
    )
    return _post_process_polar_image(polar, img, radius)


def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords
