from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from dask import array as da, delayed
import impy as ip
from typing import TYPE_CHECKING, Callable
from .const import Mode

if TYPE_CHECKING:
    from .components import Spline

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
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")
    
    if size < z1:
        z1_pad = z1 - size
        z1 = size
    elif z1 < 0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")
    
    return slice(z0, z1), (z0_pad, z1_pad)

def crop_tomogram(
    img: ip.ImgArray | ip.LazyImgArray, 
    pos,
    shape: tuple[int, int, int]
) -> ip.ImgArray:
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


def mirror_pcc(img0: ip.ImgArray, mask=None, max_shifts=None):
    """
    Phase cross correlation of an image and its mirror image.
    Identical to ``ip.pcc_maximum(img0, img0[::-1, ::-1])``
    FFT of the mirror image can efficiently calculated from FFT of the original image.
    """    
    ft0 = img0.fft()
    
    return mirror_ft_pcc(ft0, mask=mask, max_shifts=max_shifts)


def mirror_ft_pcc(ft0: ip.ImgArray, mask=None, max_shifts=None):
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
    return ip.ft_pcc_maximum(ft0, ft1, mask=mask, max_shifts=max_shifts) + 1
    

def map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray,
    coordinates: np.ndarray,
    order: int = 3, 
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0
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
    
    return ndi.map_coordinates(
        np.asarray(img),
        coordinates=coordinates,
        order=order,
        mode=mode, 
        cval=cval,
        prefilter=order > 1,
    )


@delayed
def lazy_map_coordinates(
    input: np.ndarray,
    coordinates: np.ndarray,
    order: int = 3, 
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0
) -> np.ndarray:
    """Delayed version of ndi.map_coordinates."""
    return ndi.map_coordinates(
        input,
        coordinates=coordinates,
        order=order,
        mode=mode, 
        cval=cval,
        prefilter=order > 1,
    )

def multi_map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray, 
    coordinates: np.ndarray,
    order: int = 3, 
    mode: str = Mode.constant,
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
) -> list[np.ndarray]:
    shape = input.shape
    coordinates = coordinates.copy()
    
    if coordinates.ndim != input.ndim + 2:
        if coordinates.ndim == input.ndim + 1:
            coordinates = coordinates[np.newaxis]
        else:
            raise ValueError(
                f"Coordinates have wrong dimension: {coordinates.shape}."
                )
    
    sl = []
    for i in range(coordinates.shape[1]):
        imin = int(np.min(coordinates[:, i])) - order
        imax = ceilint(np.max(coordinates[:, i])) + order + 1
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        coordinates[:, i] -= _sl.start
    
    img = input[tuple(sl)]
    if isinstance(img, ip.LazyImgArray):
        img = img.compute()
    if callable(cval):
        cval = cval(img)
    input_img = np.asarray(img)
    
    tasks = []
    for crds in coordinates:
        mapped = lazy_map_coordinates(
            input_img,
            coordinates=crds,
            order=order,
            mode=mode, 
            cval=cval,
        )
        
        tasks.append(da.from_delayed(mapped, coordinates.shape[2:], dtype=np.float32))
    
    out = da.compute(tasks)[0]

    return np.stack(out, axis=0)

def oblique_meshgrid(
    shape: tuple[int, int], 
    tilts: tuple[float, float] = (0., 0.),
    intervals: tuple[float, float] = (0., 0.),
    offsets: tuple[float, float] = (0., 0.)
) -> np.ndarray:
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

def angle_uniform_filter(input, size, mode=Mode.mirror, cval=0):
    """Uniform filter of angles."""
    phase = np.exp(1j*input)
    out = ndi.convolve1d(phase, np.ones(size), mode=mode, cval=cval)
    return np.angle(out)


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
        if npf > 1:
            self.zx_ave = rotational_average(self.zx, fold=int(npf))
        return self.zx_ave
