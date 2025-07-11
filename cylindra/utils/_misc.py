from __future__ import annotations

import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import impy as ip
import numpy as np
import polars as pl
from acryo.tilt import NoWedge, TiltSeriesModel, dual_axis, single_axis
from dask import array as da
from numpy.typing import NDArray
from scipy import ndimage as ndi

from cylindra._dask import Delayed, compute, delayed
from cylindra.const import Mode


def roundint(a: float) -> int:
    return int(round(a))


def ceilint(a: float) -> int:
    return int(math.ceil(a))


def floorint(a: float) -> int:
    return int(math.floor(a))


def assert_column_exists(df: pl.DataFrame, col: str | list[str]):
    """Check if column exists in DataFrame, for better error message."""
    if isinstance(col, str):
        col = [col]
    not_exist = [c for c in col if c not in df.columns]
    if not_exist:
        raise ValueError(
            f"Column {not_exist!r} does not exist in the DataFrame. "
            f"Existing columns are: {df.columns!r}."
        )


def distance_matrix(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Return the distance matrix between two arrays.

    distance_matrix(a, b) will return a matrix of shape (a.shape[0], b.shape[0])
    """
    return np.linalg.norm(a[:, np.newaxis] - b[np.newaxis], axis=-1)


def interp(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    order: int = 1,
    axis: int = -1,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    from scipy.interpolate import interp1d

    match order:
        case 0:
            kind = "nearest"
        case 1:
            kind = "linear"
        case 3:
            kind = "cubic"
        case v:
            raise ValueError(f"`interpolation` must be 0, 1 or 3. Got {v}.")
    return interp1d(x, y, kind=kind, axis=axis)


@contextmanager
def set_gpu():
    """Use GPU within this context."""
    # NOTE: Support GPU in the future...
    yield


def make_slice_and_pad(z0: int, z1: int, size: int) -> tuple[slice, tuple[int, int]]:
    """
    Calculate slice and padding for array slicing.

    This function calculates what slicing and padding are needed when an array is
    sliced by ``z0:z1``. Array must be padded when z0 is negative or z1 is outside the
    array size.
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
    Crop tomogram at the integer borders.

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
    return reg.astype(np.float32, copy=False)


def crop_tomograms(
    img: ip.ImgArray | ip.LazyImgArray,
    positions: Sequence[tuple[int, int, int]],
    shape: tuple[int, int, int],
) -> ip.ImgArray:
    from dask import array as da

    regs = []
    is_lazy = isinstance(img, ip.LazyImgArray)
    pad_fn = da.pad if is_lazy else np.pad
    for z, y, x in positions:
        rz, ry, rx = ((s - 1) / 2 for s in shape)
        sizez, sizey, sizex = img.sizesof("zyx")

        sl_z, pad_z = make_slice_and_pad(roundint(z - rz), roundint(z + rz + 1), sizez)
        sl_y, pad_y = make_slice_and_pad(roundint(y - ry), roundint(y + ry + 1), sizey)
        sl_x, pad_x = make_slice_and_pad(roundint(x - rx), roundint(x + rx + 1), sizex)
        reg = img.value[sl_z, sl_y, sl_x]
        pads = [pad_z, pad_y, pad_x]
        if np.any(np.array(pads) > 0):
            reg = pad_fn(reg, pads, mode="constant", constant_values=reg.mean())
        regs.append(reg)
    if isinstance(img, ip.ImgArray):
        return ip.asarray(np.stack(regs, axis=0), axes="pzyx", dtype=np.float32)
    else:
        return ip.asarray(np.stack(compute(*regs), axis=0), dtype=np.float32)


def centroid(arr: NDArray[np.floating], xmin: int, xmax: int) -> float:
    """
    Centroid of 1D array.

    Calculate the centroid of arr between xmin and xmax, for detection of the
    maximal index at subpixel precision.
    """
    xmin = max(xmin, 0)
    xmax = min(xmax, arr.size)
    x = np.arange(xmin, xmax)
    input_arr = arr[xmin:xmax] - np.min(arr[xmin:xmax])
    return float(np.sum(input_arr * x) / np.sum(input_arr))


def centroid_2d(
    arr: NDArray[np.floating], slice_y: slice, slice_x: slice
) -> tuple[float, float]:
    """Centroid of 2D array."""
    ind_y = np.arange(slice_y.start, slice_y.stop)
    ind_x = np.arange(slice_x.start, slice_x.stop)
    input_arr = arr[slice_y, slice_x] - np.min(arr[slice_y, slice_x])
    yy, xx = np.meshgrid(ind_y, ind_x, indexing="ij")
    ymax = float(np.sum(input_arr * yy) / np.sum(input_arr))
    xmax = float(np.sum(input_arr * xx) / np.sum(input_arr))
    return ymax, xmax


def rotational_average(img: ip.ImgArray, fold: int = 13):
    """Rotational average of 2D image supposing `fold` PFs."""
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
    n_segs, res = divmod(value + 1e-6, interval)
    return value - res, int(n_segs)


_A = TypeVar("_A", bound=np.ndarray)


def normalize_image(img: _A, outlier: float = (0.01, 0.01)) -> _A:
    min, max = np.quantile(img, [outlier[0], 1 - outlier[1]])
    return (img - min) / (max - min)


def map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray,
    coordinates: NDArray[np.float32],
    order: int = 3,
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
    prefilter: bool | None = None,
) -> ip.ImgArray:
    """
    Crop image at the edges of coordinates before calling map_coordinates to avoid
    loading entire array into memory.
    """
    return map_coordinates_task(
        input, coordinates, order, mode, cval, prefilter
    ).compute()


@delayed
def _delayed_map_coordinates(
    img: NDArray[np.float32],
    coordinates: NDArray[np.float32],
    order: int = 3,
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
    prefilter: bool | None = None,
    like: ip.ImgArray | ip.LazyImgArray | None = None,
):
    img = img.astype(np.float32, copy=False)
    if callable(cval):
        cval = cval(img)
    if prefilter is None:
        prefilter = order > 1
    out = ndi.map_coordinates(
        img,
        coordinates=coordinates,
        order=order,
        mode=mode,
        prefilter=prefilter,
        cval=cval,
    )
    return ip.asarray(out, like=like)


def map_coordinates_task(
    input: ip.ImgArray | ip.LazyImgArray,
    coordinates: np.ndarray,
    order: int = 3,
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
    prefilter: bool | None = None,
) -> Delayed[ip.ImgArray]:
    coordinates = coordinates.copy()
    shape = input.shape
    sl: list[slice] = []
    for i in range(input.ndim):
        imin = int(np.min(coordinates[i])) - order
        imax = ceilint(np.max(coordinates[i])) + order + 1
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        coordinates[i] -= _sl.start

    img_cropped = input[tuple(sl)]
    task = _delayed_map_coordinates(
        img_cropped.value,
        coordinates,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
        like=input,
    )
    return task


def find_changing_point(arr: NDArray[np.floating]) -> int:
    from cylindra._cylindra_ext import find_changing_point as _find_changing_point_rs

    return _find_changing_point_rs(np.asarray(arr, dtype=np.float32))


def with_columns(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    """More robust version of df.with_columns(other)."""
    if df.shape[0] == 0:
        if not isinstance(other, pl.DataFrame):
            return pl.DataFrame(other)
        return other
    return df.with_columns(other)


def read_tilt_angles_from_mdoc(path: str) -> NDArray[np.float32]:
    text = Path(path).read_text()
    pattern = re.compile(r"TiltAngle\s*=\s*(-?\d+\.\d+)")
    angles = [float(m.group(1)) for m in pattern.finditer(text)]
    return np.asarray(angles, dtype=np.float32)


def read_tilt_angles_from_rawtlt(path: str) -> NDArray[np.float32]:
    angles = [float(val) for val in Path(path).read_text().splitlines()]
    return np.asarray(angles, dtype=np.float32)


def find_tilt_angles(dirpath: Path) -> NDArray[np.float32] | None:
    for path in dirpath.glob("*.mdoc"):
        try:
            tilt_angle = read_tilt_angles_from_mdoc(path)
        except Exception:
            pass
        else:
            return tilt_angle
    for path in dirpath.glob("*.rawtlt"):
        try:
            tilt_angle = read_tilt_angles_from_rawtlt(path)
        except Exception:
            pass
        else:
            return tilt_angle


def parse_tilt_model(tilt) -> TiltSeriesModel:
    if not isinstance(tilt, TiltSeriesModel):
        if isinstance(tilt, dict):
            match tilt["kind"]:
                case "none":
                    tilt = NoWedge()
                case "x" | "y":
                    tilt = single_axis(tilt["range"], tilt["kind"])
                case "dual":
                    tilt = dual_axis(tilt["yrange"], tilt["xrange"])
                case _:  # pragma: no cover
                    raise ValueError(f"Tilt model {tilt!r} not in a correct format.")
        else:
            tilt = single_axis(tilt)
    return tilt


def fit_to_shape(img: ip.ImgArray, shape: tuple[int, int, int]) -> ip.ImgArray:
    """Fit image to the specified shape.

    If the image is larger than the specified shape, it will be cropped.
    If the image is smaller, it will be padded with zero.
    """
    if img.shape == shape:
        return img
    img = img.copy()
    shifts = [0] * img.ndim
    tasks: list[tuple[str, Any]] = []
    for i, (s, sh) in enumerate(zip(img.shape, shape, strict=False)):
        ds = sh - s
        if ds > 0:
            if ds % 2 == 0:
                tasks.append(("pad", [(ds // 2, ds // 2)]))
            else:
                tasks.append(("pad", [(ds // 2, ds // 2 + 1)]))
                shifts[i] = 0.5
        elif ds < 0:
            if ds % 2 == 0:
                sl = [slice(None)] * img.ndim
                sl[i] = slice(-ds // 2, -ds // 2 + sh)
                tasks.append(("slice", tuple(sl)))
            else:
                sl = [slice(None)] * img.ndim
                sl[i] = slice(-ds // 2, -ds // 2 + sh)
                tasks.append(("slice", tuple(sl)))
                shifts[i] = -0.5
    if any(s != 0 for s in shifts):
        img = img.shift(shifts, dims=img.axes, mode=Mode.constant, cval=0.0)
    for i, (task, args) in enumerate(tasks):
        if task == "pad":
            img = img.pad(args, dims=img.axes[i], constant_values=0)
        elif task == "slice":
            img = img[args]
        else:  # pragma: no cover
            raise RuntimeError(f"Unknown task {task!r} in fit_to_shape.")
    return img


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

    When the 3D image is projected along Y axis, that is, img.mean("y") in ``impy``,
    and viewed parallel to Y axis, the projection should look like following.

        X <-------(x) Y
                   |
                   |
                   |
                 Z v

    Therefore, if we use standard ``imshow`` functions like ``plt.imshow`` and those
    in ``pyqtgraph``, we must **flip along X axis**.
    """

    def __init__(self, image: da.Array, npf: int = 13):
        self.yx = image.mean(axis=0, dtype=np.float32)
        self.zx = image.mean(axis=1, dtype=np.float32)[:, ::-1]
        self.zx_ave = None
        self.npf = int(npf)

        self.shape = image.shape

    def compute(self) -> Projections:
        """Compute the projection if needed."""
        if isinstance(self.yx, da.Array) and isinstance(self.zx, da.Array):
            self.yx, self.zx = compute(self.yx, self.zx)
        if self.zx_ave is None and self.npf > 1:
            self.zx_ave = rotational_average(self.zx, fold=self.npf)
        return self
