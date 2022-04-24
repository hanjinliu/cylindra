from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
import impy as ip
from typing import Callable, ContextManager
from .const import Mode, GVar
try:
    from . import _cpp_ext
except ImportError:
    # In case build failed
    pass

def roundint(a: float):
    return int(round(a))

def ceilint(a: float):
    return int(np.ceil(a))

def set_gpu() -> ContextManager:
    """Use GPU within this context."""
    if GVar.GPU:
        return ip.use("cupy")
    else:
        return ip.use("numpy")

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
    z, y, x = pos
    rz, ry, rx = [(s-1)/2 for s in shape]
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
    return np.sum(input_arr*x)/np.sum(input_arr)


def rotational_average(img: ip.ImgArray, fold: int = 13):
    angles = np.arange(fold)*360/fold
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
    
    with set_gpu():
        shift = ip.ft_pcc_maximum(ft0, ft1, mask=mask, max_shifts=max_shifts) + 1
    return shift

def mirror_zncc(img0: ip.ImgArray, max_shifts=None):
    """
    Shift correction using zero-mean normalized cross correlation of an image 
    and its mirror image. Identical to ``ip.zncc_maximum(img0, img0[::-1, ::-1])``.
    """    
    img1 = img0[(slice(None, None, -1),) * img0.ndim]
    return ip.zncc_maximum(img0, img1, max_shifts=max_shifts)

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
    
    return img.map_coordinates(
        coordinates=coordinates,
        order=order,
        mode=mode, 
        cval=cval,
    )


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

def pad_template(template: ip.ImgArray, shape: tuple[int, ...]) -> ip.ImgArray:
    """
    Pad (or crop) template image to the specified shape.
    
    This function will not degenerate template information unless template is cropped.
    Template image is inserted into a empty image with correct shape with integer
    shift.

    Parameters
    ----------
    template : ip.ImgArray
        Template image.
    shape : tuple[int, ...]
        Desired output shape.

    Returns
    -------
    ip.ImgArray
        New image with desired shape.
    """
    template_shape = template.shape
    if len(template_shape) != len(shape):
        raise ValueError(
            f"Size mismatch between template shape {tuple(template_shape)!r} "
            f"and required shape {shape!r}"
        )
    
    needed_size_up = np.array(shape) - np.array(template_shape)
    pads: list[tuple[int, int]] = []
    for i, s in enumerate(needed_size_up):
        dw = s//2
        if s < 0:
            template = template[(slice(None),)*i, slice(dw, -s+dw)]
            pads.append((0, 0))
        else:
            pads.append((dw, s - dw))
    
    return template.pad(pads, dims=template.axes, constant_values=np.min(template))


def pad_sheared_edges(
    arr: np.ndarray,
    pad_width: tuple(int, int),
    axial_mode: str = Mode.reflect,
    cval: float = 0.0,
    start: int = -3,
) -> np.ndarray:
    if start < 0:
        start = -start
        arr = arr[:, ::-1]
    
    if arr.ndim != 2:
        raise ValueError("Only 2D arrays are supported.")
    
    # arr shape: (y, pf)
    pad_kwargs = dict(mode=axial_mode)
    if axial_mode == Mode.constant:
        pad_kwargs.update(constant_values=cval)
    
    pad_long, pad_lat = pad_width
    s0, s1 = arr.shape
    
    if pad_lat > arr.shape[1]:
        raise ValueError("Cannot pad wider than array width.")
    
    # `arr` will be padded like below, suppose arr has seven columns (=protofilaments)
    # and `start == 2`.`

    #   */////*///////*/////*  <--+
    #   */////*///////*******     | 
    #   */////*///////*12345*     |
    #   */////*********     *     |
    #   */////*1234567*rpad *     |
    #   *******       *     *     |  
    #   *34567*  arr  *     *     +- These blanks disappears if pad length is shorter
    #   *     *       *ABCDE*     |  than `start`.
    #   * lpad*       *******     |
    #   *     *ABCDEFG*/////*     |
    #   *     *********/////*     |
    #   *CDEFG*///////*/////*     |
    #   *******///////*/////*     |
    #   */////*///////*/////*  <--+
    #      ^      ^      ^
    #      |      |      |
    #      |      |      +-- rpad_padded
    #      |      +--------- arr_padded
    #      +---------------- lpad_padded

    # where shaded area by slash "/" means that the area needs padding.
    
    lpad = arr[:, s1-pad_lat:]
    rpad = arr[:, :pad_lat]
    
    if pad_long < start:
        lpad_padded = np.pad(
            lpad[:s0-(start-pad_long)], 
            pad_width=[(pad_long + start, 0), (0, 0)], 
            **pad_kwargs
        )
        rpad_padded = np.pad(
            rpad[(start-pad_long):],
            pad_width=[(0, pad_long + start), (0, 0)], 
            **pad_kwargs
        )
    else:
        lpad_padded = np.pad(
            lpad, 
            pad_width=[(pad_long + start, pad_long - start), (0, 0)], 
            **pad_kwargs
        )
        rpad_padded = np.pad(
            rpad, 
            pad_width=[(pad_long - start, pad_long + start), (0, 0)], 
            **pad_kwargs
        )
    
    arr_padded = np.pad(arr, [(pad_long, pad_long), (0, 0)], **pad_kwargs)
    
    out = np.concatenate([lpad_padded, arr_padded, rpad_padded], axis=1)
    
    if start < 0:
        out = arr[:, ::-1]
    return out

def sheared_convolve(
    input: np.ndarray,
    weights: np.ndarray, 
    axial_mode: str = Mode.reflect,
    cval: float = 0.0,
    start: int = -3,
):
    """Convolution of an image with sheared boundary."""
    input = np.asarray(input)
    weights = np.asarray(weights)
    filter_length, filter_width = weights.shape
    l_ypad = filter_length // 2
    l_apad = filter_width // 2
    input_padded = pad_sheared_edges(
        input, (l_ypad, l_apad), axial_mode=axial_mode, cval=cval, start=start
    )
    out: np.ndarray = ndi.convolve(input_padded, weights)
    ly, lx = out.shape
    out_unpadded = out[l_ypad:ly - l_ypad, l_apad:lx - l_apad]
    return out_unpadded

def interval_filter(
    pos: np.ndarray,
    vec: np.ndarray,
    filter_length: int,
    filter_width: int,
    start: int
) -> np.ndarray:
    
    # For instance, if (filter_length, filter_width) = (3, 3), momoner intervals
    # will be averaged in the following way.
    #
    # - (*) ... center monomer
    # - (o) ... adjacent monomers
    #
    # (o) (o) (o)
    #  :   :   :   <---+---- These six intervals will be averaged.
    # (o) (*) (o)      |
    #  :   :   :   <---+
    # (o) (o) (o)
    #
    # 
    # calculated interval
    #     |
    #  |<-+-->|
    #        (o)
    # (o)
    #  -------> vec
    
    ny, npf, ndim = pos.shape
    
    # equivalent to padding mode "reflect"
    pitch_vec = np.diff(pos, axis=0, append=(2*pos[-1] - pos[-2])[np.newaxis])  
    
    vec_norm: np.ndarray = vec / np.sqrt(np.sum(vec**2, axis=1))[:, np.newaxis]
    vec_norm = vec_norm.reshape(-1, npf, ndim)
    y_interval: np.ndarray = np.sum(pitch_vec * vec_norm, axis=2)  # inner product

    ker = np.ones((filter_length, filter_width))
    ker[-1, :] = 0
    ker /= ker.sum()
    y_interval = sheared_convolve(y_interval, ker, start=start)
    
    return y_interval


def viterbi(
    score: np.ndarray,
    origin: np.ndarray,
    zvec: np.ndarray,
    yvec: np.ndarray,
    xvec: np.ndarray,
    dist_min: float,
    dist_max: float,
) -> tuple[np.ndarray, float]:
    """
    One-dimensional Viterbi algorithm for contexted subtomogram alignment.

    Parameters
    ----------
    score : (N, Nz, Ny, Nx) array
        Array of score landscape.
    origin : (N, 3) array
        World coordinates of origin of local coordinates.
    zvec : (N, 3) array
        World coordinate vectors of z-axis.
    yvec : (N, 3) array
        World coordinate vectors of y-axis.
    xvec : (N, 3) array
        World coordinate vectors of x-axis.
    dist_min : float
        Minimum distance between subtomograms.
    dist_max : float
        Maximum distance between subtomograms.

    Returns
    -------
    (N, 3) int array and float
        Optimal indices and optimal score.
    """
    nmole, nz, ny, nx = score.shape
    if origin.shape != (nmole, 3):
        raise ValueError(f"Shape of 'origin' must be ({nmole}, 3) but got {origin.shape}.")
    if zvec.shape != (nmole, 3):
        raise ValueError(f"Shape of 'zvec' must be ({nmole}, 3) but got {zvec.shape}.")
    if yvec.shape != (nmole, 3):
        raise ValueError(f"Shape of 'yvec' must be ({nmole}, 3) but got {yvec.shape}.")
    if xvec.shape != (nmole, 3):
        raise ValueError(f"Shape of 'xvec' must be ({nmole}, 3) but got {xvec.shape}.")
    if dist_min >= dist_max:
        raise ValueError("'dist_min' must be smaller than 'dist_max'.")
    
    return _cpp_ext.viterbi(score, origin, zvec, yvec, xvec, dist_min, dist_max)
    

class Projections:
    """Class that stores projections of a 3D image."""

    # We have to think thoroughly about the XYZ coordinate here.
    # In right-handed coordinate system, the XYZ axes look like following.

    #     Z (parallel to sight)
    #    (x)------> X
    #     |
    #     |
    #     |
    #     v Y
    
    # When the 3D image is projected along Y axis, that is, img.proj("y") in ``impy``,
    # and viewed parallel to Y axis, the projection should look like following.
    
    #     X <-------(x) Y
    #                |
    #                |
    #                |
    #              Z v

    # Therefore, if we use standard ``imshow`` functions like ``plt.imshow`` and those
    # in ``pyqtgraph``, we must **flip along X axis**.

    def __init__(self, image: ip.ImgArray):
        self.yx = image.proj("z")
        self.zx = image.proj("y")["x=::-1"]
        self.zx_ave = None
        
        self.shape = image.shape
    
    def rotational_average(self, npf: int):
        if npf > 1:
            self.zx_ave = rotational_average(self.zx, fold=int(npf))
        return self.zx_ave
