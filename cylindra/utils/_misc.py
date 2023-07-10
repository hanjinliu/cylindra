from __future__ import annotations

import math
import numpy as np
import impy as ip
from typing import TypeVar, Callable
from cylindra.const import Mode
from contextlib import contextmanager


def roundint(a: float):
    return int(round(a))


def ceilint(a: float):
    return int(math.ceil(a))


def floorint(a: float):
    return int(math.floor(a))


@contextmanager
def set_gpu():
    """Use GPU within this context."""
    # NOTE: Support GPU in the future...
    yield


def make_slice_and_pad(z0: int, z1: int, size: int) -> tuple[slice, tuple[int, int]]:
    """
    This function calculates what slicing and padding are needed when an array is sliced
    by ``z0:z1``. Array must be padded when z0 is negative or z1 is outside the array size.
    """
    z0_pad = z1_pad = 0
    if z0 < 0:
        z0_pad = -z0
        z0 = 0
    elif size < z0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")

    if size < z1:
        z1_pad = z1 - size
        z1 = size
    elif z1 < 0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")

    return slice(z0, z1), (z0_pad, z1_pad)


def crop_tomogram(
    img: ip.ImgArray | ip.LazyImgArray, pos, shape: tuple[int, int, int]
) -> ip.ImgArray:
    """
    From large image ``img``, crop out small region centered at ``pos``.
    Image will be padded if needed.
    """
    z, y, x = pos
    rz, ry, rx = ((s - 1) / 2 for s in shape)
    sizez, sizey, sizex = img.sizesof("zyx")

    sl_z, pad_z = make_slice_and_pad(roundint(z - rz), roundint(z + rz + 1), sizez)
    sl_y, pad_y = make_slice_and_pad(roundint(y - ry), roundint(y + ry + 1), sizey)
    sl_x, pad_x = make_slice_and_pad(roundint(x - rx), roundint(x + rx + 1), sizex)
    reg = img[sl_z, sl_y, sl_x]
    if isinstance(reg, ip.LazyImgArray):
        reg = reg.compute()
    pads = [pad_z, pad_y, pad_x]
    if np.any(np.array(pads) > 0):
        reg = reg.pad(pads, dims="zyx", constant_values=reg.mean())

    return reg


def centroid(arr: np.ndarray, xmin: int, xmax: int) -> float:
    """
    Calculate the centroid of arr between xmin and xmax, for detection of subpixel maxima.
    """
    xmin = max(xmin, 0)
    xmax = min(xmax, arr.size)
    x = np.arange(xmin, xmax)
    input_arr = arr[xmin:xmax] - np.min(arr[xmin:xmax])
    return np.sum(input_arr * x) / np.sum(input_arr)


def rotational_average(img: ip.ImgArray, fold: int = 13):
    angles = np.arange(fold) * 360 / fold
    average_img = img.copy()
    for angle in angles[1:]:
        average_img.value[:] += img.rotate(angle, dims="zx", mode=Mode.nearest)
    average_img /= fold
    return average_img


def interval_divmod(value: float, interval: float) -> tuple[float, int]:
    """
    Calculate stop and n_segs, where satisfy:
    1. stop == interval * n_segs
    2. stop <= value
    3. stop is largest.
    """
    if interval == 0:
        raise ZeroDivisionError("Devided by zero.")
    n_segs, res = divmod(value + 1e-8, interval)
    return value - res, int(n_segs)


_A = TypeVar("_A", bound=np.ndarray)


def merge_images(img0: _A, img1: _A) -> _A:
    img0 = np.asarray(img0)
    img1 = np.asarray(img1)
    img0_norm = img0 - img0.min()
    img0_norm /= img0_norm.max()
    img1_norm = img1 - img1.min()
    img1_norm /= img1_norm.max()
    return np.stack([img0_norm, img1_norm, img0_norm], axis=-1)


def normalize_image(img: _A, outlier: float = 0.01) -> _A:
    min, max = np.quantile(img, [outlier, 1 - outlier])
    return (img - min) / (max - min)


def map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray,
    coordinates: np.ndarray,
    order: int = 3,
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
) -> np.ndarray:
    """
    Crop image at the edges of coordinates before calling map_coordinates to avoid
    loading entire array into memory.
    """
    coordinates = coordinates.copy()
    shape = input.shape
    sl = []
    for i in range(input.ndim):
        imin = int(np.min(coordinates[i])) - order
        imax = ceilint(np.max(coordinates[i])) + order + 1
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        coordinates[i] -= _sl.start

    img = input[tuple(sl)]
    if isinstance(img, ip.LazyImgArray):
        img = img.compute()
    if callable(cval):
        cval = cval(img)

    return img.map_coordinates(
        coordinates=coordinates,
        order=order,
        mode=mode,
        cval=cval,
    )


class Projections:
    """
    Class that stores projections of a 3D image, calculated lazily.

    Note
    ----
    We have to think thoroughly about the XYZ coordinate here.
    In right-handed coordinate system, the XYZ axes look like following.

        Z (parallel to sight)
       (x)------> X
        |
        |
        |
        v Y

    When the 3D image is projected along Y axis, that is, img.proj("y") in ``impy``,
    and viewed parallel to Y axis, the projection should look like following.

        X <-------(x) Y
                   |
                   |
                   |
                 Z v

    Therefore, if we use standard ``imshow`` functions like ``plt.imshow`` and those
    in ``pyqtgraph``, we must **flip along X axis**.
    """

    FLIP = ip.slicer.x[::-1]

    def __init__(self, image: ip.ImgArray | ip.LazyImgArray, npf: int = 13):
        self.yx = image.proj("z")
        self.zx = image.proj("y")[self.FLIP]
        self.zx_ave = None
        self.npf = int(npf)

        self.shape = image.shape

    def compute(self) -> Projections:
        """Compute the projection if needed."""
        if isinstance(self.yx, ip.LazyImgArray):
            self.yx = self.yx.compute()
        if isinstance(self.zx, ip.LazyImgArray):
            self.zx = self.zx.compute()
        if self.zx_ave is None and self.npf > 1:
            self.zx_ave = rotational_average(self.zx, fold=self.npf)
        return self
