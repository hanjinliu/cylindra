from __future__ import annotations
import numpy as np
import impy as ip
from scipy import ndimage as ndi
from ..const import Mode


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
