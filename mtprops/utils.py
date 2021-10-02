import numpy as np
from ._dependencies import impy as ip

def make_slice_and_pad(center:int, radius:int, size:int):
    z0 = center - radius
    z1 = center + radius + 1
    z0_pad = z1_pad = 0
    if z0 < 0:
        z0_pad = -z0
        z0 = 0
    elif size < z1:
        z1_pad = z1 - size
        z1 = size

    return slice(z0, z1), (z0_pad, z1_pad)

def load_a_subtomogram(img, pos, radius:tuple[int, int, int], dask:bool=True):
    """
    From large image ``img``, crop out small region centered at ``pos``.
    Image will be padded if needed.
    """
    z, y, x = pos.astype(np.int32)
    rz, ry, rx = radius
    sizez, sizey, sizex = img.sizesof("zyx")

    sl_z, pad_z = make_slice_and_pad(z, rz, sizez)
    sl_y, pad_y = make_slice_and_pad(y, ry, sizey)
    sl_x, pad_x = make_slice_and_pad(x, rx, sizex)
    reg = img[sl_z, sl_y, sl_x]
    if dask:
        reg = reg.data
    with ip.SetConst("SHOW_PROGRESS", False):
        pads = [pad_z, pad_y, pad_x]
        if np.any(np.array(pads) > 0):
            reg = reg.pad(pads, dims="zyx", constant_values=np.median(reg))
    return reg


def centroid(arr: np.ndarray, xmin: int, xmax: int) -> float:
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