from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
import impy as ip
from typing import TYPE_CHECKING, Callable

from mtprops.const import Mode

if TYPE_CHECKING:
    from .spline import Spline3D

def roundint(a: float):
    return int(round(a))

def ceilint(a: float):
    return int(np.ceil(a))

def no_verbose():
    return ip.SetConst("SHOW_PROGRESS", False)

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
        raise ValueError(f"Out of bound: {z0}:{z1}")
    
    if size < z1:
        z1_pad = z1 - size
        z1 = size
    elif z1 < 0:
        raise ValueError(f"Out of bound: {z0}:{z1}")
    
    return slice(z0, z1), (z0_pad, z1_pad)

def load_a_subtomogram(img: ip.ImgArray | ip.LazyImgArray, 
                       pos,
                       shape: tuple[int, int, int]):
    """
    From large image ``img``, crop out small region centered at ``pos``.
    Image will be padded if needed.
    """
    # TODO: inefficient if using cupy
    z, y, x = pos
    rz, ry, rx = [(s-1)/2 for s in shape]
    sizez, sizey, sizex = img.sizesof("zyx")

    sl_z, pad_z = make_slice_and_pad(roundint(z - rz), roundint(z + rz + 1), sizez)
    sl_y, pad_y = make_slice_and_pad(roundint(y - ry), roundint(y + ry + 1), sizey)
    sl_x, pad_x = make_slice_and_pad(roundint(x - rx), roundint(x + rx + 1), sizex)
    reg = img[sl_z, sl_y, sl_x]
    if isinstance(reg, ip.LazyImgArray):
        reg = reg.compute()
    with no_verbose():
        pads = [pad_z, pad_y, pad_x]
        if np.any(np.array(pads) > 0):
            reg = reg.pad(pads, dims="zyx", constant_values=np.mean(reg))
    
    return reg


def load_a_rot_subtomogram(img: ip.ImgArray, length_px: int, width_px: int, spl: "Spline3D", u):
    plane_shape = (width_px, width_px)
    axial_size = length_px
    out = []
    with no_verbose():
        coords = spl.local_cartesian(plane_shape, axial_size, u)
        coords = np.moveaxis(coords, -1, 0)
        out = map_coordinates(img, coords, order=3, mode=Mode.constant, cval=np.mean)
    out = ip.asarray(out, axes="zyx")
    out.set_scale(img)
    return out


def load_rot_subtomograms(img: ip.ImgArray | ip.LazyImgArray, length_px: int, width_px: int,
                          spl: "Spline3D"):
    plane_shape = (width_px, width_px)
    axial_size = length_px
    out = []
    with no_verbose():
        for u in spl.anchors:
            coords = spl.local_cartesian(plane_shape, axial_size, u)
            coords = np.moveaxis(coords, -1, 0)
            out.append(map_coordinates(img, coords, order=3, mode=Mode.constant, cval=np.mean))
    out = ip.asarray(np.stack(out, axis=0), axes="pzyx")
    out.set_scale(img)
    return out


def centroid(arr: np.ndarray, xmin: int, xmax: int) -> float:
    """
    Calculate the centroid of arr between xmin and xmax, for detection of subpixel maxima.
    """    
    xmin = max(xmin, 0)
    xmax = min(xmax, arr.size)
    x = np.arange(xmin, xmax)
    input_arr = arr[xmin:xmax] - np.min(arr[xmin:xmax])
    return np.sum(input_arr*x)/np.sum(input_arr)


def rotational_average(img: ip.ImgArray, fold: int = 13):
    angles = np.arange(fold)*360/fold
    average_img = img.copy()
    with no_verbose():
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


def mirror_pcc(img0: ip.ImgArray, mask=None):
    """
    Phase cross correlation of an image and its mirror image.
    Identical to ``ip.pcc_maximum(img0, img0[::-1, ::-1])``
    FFT of the mirror image can efficiently calculated from FFT of the original image.
    """    
    ft0 = img0.fft()
    
    return mirror_ft_pcc(ft0, mask)


def mirror_ft_pcc(ft0: ip.ImgArray, mask=None):
    """
    Phase cross correlation of an image and its mirror image.
    Identical to ``ip.ft_pcc_maximum(img0, img0[::-1, ::-1])``
    ``ft0`` must be FFT of ``img0``.
    """    
    shape = ft0.shape
    ind = np.indices(shape)
    phase = np.sum([ix/n for ix, n in zip(ind, shape)])
    weight = np.exp(1j*2*np.pi*phase)
    
    ft1 = weight*ft0.conj()
    return ip.ft_pcc_maximum(ft0, ft1, mask) + 1
    

def map_coordinates(input: ip.ImgArray | ip.LazyImgArray, 
                    coordinates: np.ndarray,
                    order: int = 3, 
                    mode: str = Mode.constant,
                    cval: float | Callable[[ip.ImgArray], float] = 0.0) -> np.ndarray:
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
    
    return ndi.map_coordinates(
        img.value,
        coordinates=coordinates,
        order=order,
        mode=mode, 
        cval=cval,
        prefilter=order>1
    )

def multi_map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray, 
    coordinates: np.ndarray,
    order: int = 3, 
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
    chunksize: int = 1,
) -> list[np.ndarray]:
    """
    Crop image at the edges of coordinates before calling map_coordinates to avoid
    loading entire array into memory.
    """    
    shape = input.shape
    
    out: list[np.ndarray] = []
    chunk_offset = 0
    while chunk_offset < coordinates.shape[0]:
        crds = coordinates[chunk_offset : chunk_offset+chunksize].copy()
        sl = []
        for i in range(input.ndim):
            imin = int(np.min(crds[:, i])) - order
            imax = ceilint(np.max(crds[:, i])) + order + 1
            _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
            sl.append(_sl)
            crds[:, i] -= _sl.start
        
        img = input[tuple(sl)]
        if isinstance(img, ip.LazyImgArray):
            img = img.compute()
        if callable(cval):
            cval = cval(img)
        input_img = img.value
        
        for each_crds in crds:
            out.append(
                ndi.map_coordinates(
                    input_img,
                    coordinates=each_crds,
                    order=order,
                    mode=mode, 
                    cval=cval,
                    prefilter=order>1
                )
            )
        
        chunk_offset += chunksize
    
    return out


def oblique_meshgrid(shape: tuple[int, int], 
                     tilts: tuple[float, float] = (0., 0.),
                     intervals: tuple[float, float] = (0., 0.),
                     offsets: tuple[float, float] = (0., 0.)) -> np.ndarray:
    """
    Construct 2-D meshgrid in oblique coordinate system.

    Parameters
    ----------
    shape : tuple[int, int]
        Output shape. If ``shape = (a, b)``, length of the output mesh will be ``a`` along
        the first axis, and will be ``b`` along the second one.
    tilts : tuple[float, float], optional
        Tilt tangents of each axis in world coordinate. Positive tangent means that the 
        corresponding axis tilt toward the line "y=x".
    intervals : tuple[float, float], optional
        The intervals (or scale) of new axes. 
    offsets : tuple[float, float], optional
        The origin of new coordinates.

    Returns
    -------
    np.ndarray
        World coordinates of lattice points of new coordinates.
    """
    tan0, tan1 = tilts
    d0, d1 = intervals
    c0, c1 = offsets
    n0, n1 = shape
    
    v0 = np.array([1, tan0], dtype=np.float32)
    v1 = np.array([tan1, 1], dtype=np.float32)

    out = np.empty((n0, n1, 2), dtype=np.float32)
    
    for i in range(n0):
        for j in range(n1):
            out[i, j, :] = v0 * i + v1 * j
    
    out[:, :, 0] = out[:, :, 0] * d0 + c0
    out[:, :, 1] = out[:, :, 1] * d1 + c1
    return out


class Projections:
    """
    Class that stores projections of a 3D image.
    
    .. note::
    
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
    def __init__(self, image: ip.ImgArray):
        with no_verbose():
            self.yx = image.proj("z")
            self.zx = image.proj("y")["x=::-1"]
        self.zx_ave = None
        
        self.shape = image.shape
    
    def rotational_average(self, npf: int):
        self.zx_ave = rotational_average(self.zx, fold=int(npf))
        return self.zx_ave
    