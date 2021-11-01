from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from typing import Any
from ._dependencies import impy as ip

def roundint(a: float):
    return int(round(a))

def ceilint(a: float):
    return int(np.ceil(a))

def make_slice_and_pad(z0: int, z1: int, size: int) -> tuple[slice, tuple[int, int]]:
    z0_pad = z1_pad = 0
    if z0 < 0:
        z0_pad = -z0
        z0 = 0
    if size < z1:
        z1_pad = z1 - size
        z1 = size

    return slice(z0, z1), (z0_pad, z1_pad)

def load_a_subtomogram(img, pos, shape: tuple[int, int, int], dask:bool=True):
    """
    From large image ``img``, crop out small region centered at ``pos``.
    Image will be padded if needed.
    """
    z, y, x = pos
    rz, ry, rx = [(s-1)/2 for s in shape]
    sizez, sizey, sizex = img.sizesof("zyx")

    sl_z, pad_z = make_slice_and_pad(roundint(z - rz), roundint(z + rz + 1), sizez)
    sl_y, pad_y = make_slice_and_pad(roundint(y - ry), roundint(y + ry + 1), sizey)
    sl_x, pad_x = make_slice_and_pad(roundint(x - rx), roundint(x + rx + 1), sizex)
    reg = img[sl_z, sl_y, sl_x]
    if dask:
        reg = reg.data
    with ip.SetConst("SHOW_PROGRESS", False):
        pads = [pad_z, pad_y, pad_x]
        if np.any(np.array(pads) > 0):
            reg = reg.pad(pads, dims="zyx", constant_values=np.median(reg))
    return reg


def centroid(arr: np.ndarray, xmin: int, xmax: int) -> float:
    """
    Calculate the centroid of arr between xmin and xmax, for detection of subpixel maxima.
    """    
    xmin = max(xmin, 0)
    xmax = min(xmax, arr.size)
    x = np.arange(xmin, xmax)
    input_arr = arr[xmin:xmax] - np.min(arr[xmin:xmax])
    return np.sum(input_arr*x)/np.sum(input_arr)


def rotational_average(img, fold:int=13):
    angles = np.arange(fold)*360/fold
    average_img = img.copy()
    with ip.SetConst("SHOW_PROGRESS", False):
        for angle in angles[1:]:
            average_img.value[:] += img.rotate(angle, dims="zx")
    average_img /= fold
    return average_img

def interval_divmod(value: float, interval: float) -> tuple[float, int]:
    """
    Calculate stop and n_segs, where satisfy:
    1. stop == interval * n_segs
    2. stop <= value
    3. stop is largest.
    """    
    n_segs, res = divmod(value + 1e-8, interval)
    return value - res, int(n_segs)

def map_coordinates(input, coordinates: np.ndarray, order: int = 3, mode: str = "constant",
                    cval: float = 0):
    """
    Crop image at the edges of coordinates before calling map_coordinates to avoid
    loading entire array into memory.
    """    
    coordinates = coordinates.copy()
    shape = input.shape
    sl = []
    pad = []
    for i in range(3):
        imin = int(np.min(coordinates[i]))
        imax = int(np.max(coordinates[i])) + 2
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        pad.append(_pad)
        coordinates[i] -= _sl.start
    sl = tuple(sl)
    img = input[sl].data
    
    if np.any(np.array(pad) > 0):
        img = img.pad(pad, dims="zyx", constant_values=np.median(img))
        
    return ndi.map_coordinates(img,
                               coordinates,
                               order=order,
                               mode=mode, 
                               cval=cval,
                               prefilter=order>1
                               )

def oblique_meshgrid(shape: tuple[int, int], 
                     rise: float = 0.,
                     tilt: float = 0., 
                     offset: tuple[float, float] = (0., 0.)) -> np.ndarray:
    v0 = np.array([1, tilt], dtype=np.float32)
    v1 = np.array([rise, 1], dtype=np.float32)
    n0, n1 = shape
    out = np.empty((n0, n1, 2), dtype=np.float32)
    for i in range(n0):
        for j in range(n1):
            out[i, j, :] = (v0 * i + v1 * j)
    out[:, :, 0] += offset[0]
    out[:, :, 1] += offset[1]
    return out