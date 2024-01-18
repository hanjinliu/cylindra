from __future__ import annotations

from typing import Iterable

import impy as ip
import numpy as np
from numpy.typing import NDArray

from cylindra.utils._misc import set_gpu


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
    phase = np.sum([ix / n for ix, n in zip(ind, shape, strict=True)])
    weight = np.exp(1j * 2 * np.pi * phase)

    ft1 = weight * ft0.conj()

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


def rotated_auto_zncc(
    img0: ip.ImgArray,
    degrees: Iterable[float],
    max_shifts: tuple[float, float, float] | None = None,
) -> NDArray[np.floating]:
    results = list[tuple[np.ndarray, float, float]]()
    for deg in degrees:
        img1 = img0.rotate(deg, mode="constant", dims=2)
        results.append(
            (*ip.zncc_maximum_with_corr(img0, img1, max_shifts=max_shifts), deg)
        )

    shift, corr, optimal_deg = max(results, key=lambda x: x[1])

    rad = np.deg2rad(optimal_deg - 180.0) / 2
    cos, sin = np.cos(rad), np.sin(rad)
    rot = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)

    return shift @ rot / (2 * cos)


def angle_corr(
    img: ip.ImgArray, ang_center: float = 0, drot: float = 7, nrots: int = 29
):
    # img: 3D
    img_z = img.mean(axis="z")
    mask = ip.circular_mask(img_z.shape.y / 2 + 2, img_z.shape)
    img_mirror: ip.ImgArray = img_z["x=::-1"]
    angs = np.linspace(ang_center - drot, ang_center + drot, nrots, endpoint=True)
    corrs = list[float]()
    f0 = np.sqrt(img_z.power_spectra(dims="yx", zero_norm=True))
    cval = np.mean(img_z)
    for ang in angs:
        img_mirror_rot = img_mirror.rotate(ang * 2, mode="constant", cval=cval)
        f1 = np.sqrt(img_mirror_rot.power_spectra(dims="yx", zero_norm=True))
        corr = ip.zncc(f0, f1, mask)
        corrs.append(corr)

    angle = angs[np.argmax(corrs)]
    return angle
