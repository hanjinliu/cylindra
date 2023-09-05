from __future__ import annotations
from functools import lru_cache
from typing import Callable
import numpy as np
from numpy.typing import NDArray
import impy as ip
from cylindra.const import nm, Mode
from cylindra.utils import map_coordinates, roundint


class CylindricTransformer:
    def __init__(self, r_range: tuple[nm, nm]):
        self._rmin, self._rmax = r_range
        self._input_shape: tuple[int, int, int] | None = None

    @property
    def rrange(self) -> tuple[nm, nm]:
        return self._rmin, self._rmax

    @property
    def rcenter(self) -> nm:
        return (self._rmin + self._rmax) / 2

    def rrange_px(self, scale: nm) -> tuple[float, float]:
        return self._rmin / scale, self._rmax / scale

    def _get_transform_coords(self, img: ip.ImgArray) -> NDArray[np.float32]:
        ny = img.shape.y
        if img.axes != ["z", "y", "x"]:
            raise ValueError("Input image must have axes 'z', 'y', 'x'.")
        center = img.shape.z / 2 - 0.5, img.shape.x / 2 - 0.5
        coords = polar_coords_2d(
            *self.rrange_px(img.scale.x), center=center
        )  # R, A, D(Z, X)
        zeros = np.zeros(coords.shape[:-1], dtype=np.float32)
        coords_3d = np.stack(
            [coords[..., 0], zeros, coords[..., 1]],
            axis=2,
        )  # V, S, H, D
        stacked = np.repeat(coords_3d[:, np.newaxis], ny, axis=1)
        stacked[:, :, :, 1] = np.arange(ny)[np.newaxis, :, np.newaxis]
        return np.moveaxis(stacked, -1, 0)

    def transform(self, img: ip.ImgArray) -> ip.ImgArray:
        stacked = self._get_transform_coords(img)
        self._input_shape = img.shape
        return get_polar_image(img, stacked, self.rcenter)

    def _get_inv_transform_coords(
        self, polar: ip.ImgArray, shape: tuple[int, int, int]
    ) -> NDArray[np.float32]:
        zz, yy, xx = np.indices(shape)
        zc = shape[0] / 2 - 0.5
        xc = shape[2] / 2 - 0.5
        rr = np.sqrt((zz - zc) ** 2 + (xx - xc) ** 2) - self._rmin / polar.scale.r
        aa = np.arctan2(zz - zc, xx - xc) / 2 / np.pi * polar.shape.a
        aa[aa < -0.5] += polar.shape.a
        return np.stack([rr, yy, aa], axis=0)

    def inverse_transform(
        self,
        img: ip.ImgArray,
        shape: tuple[int, int, int] | None = None,
        cval: float | Callable[[ip.ImgArray], float] = np.min,
    ) -> ip.ImgArray:
        # img.axes == "rya"
        if img.axes != ["r", "y", "a"]:
            raise ValueError("Input image must have axes 'r', 'y', 'a'.")
        if shape is None:
            if self._input_shape is None:
                raise RuntimeError("Input shape is not set.")
            shape = self._input_shape
        coords = self._get_inv_transform_coords(img, shape)
        apad = 8
        img_input = img.pad(apad, mode="wrap", dims="a")
        coords[2] += apad
        out = map_coordinates(img_input, coords, order=3, mode=Mode.constant, cval=cval)
        return out.set_axes("zyx").set_scale(xyz=img.scale.r)


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
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    a_scale = 2 * np.pi * radius / polar.shape.a
    return polar.set_scale(r=img.scale.x, y=img.scale.x, a=a_scale, unit=img.scale_unit)


def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords
