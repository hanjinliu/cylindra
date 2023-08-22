from __future__ import annotations

from functools import cached_property
from cylindra.utils import ceilint, floorint
import numpy as np
import impy as ip


def int_translate(img: ip.ImgArray, shift: int, axis="a") -> ip.ImgArray:
    if shift == 0:
        return img
    sl = ip.slicer(axis)
    img0 = img[sl[shift:]]
    img1 = img[sl[:shift]]
    return np.concatenate([img0, img1], axis=-1)


class PeakDetector:
    """A power spectrum peak detector for a given image."""

    def __init__(self, img: ip.ImgArray, nsamples: int = 8):
        self._img = img
        self._nsamples = nsamples

    def dft(self, key, upsample_factor: int) -> ip.ImgArray:
        power_spectra = []
        sample_slope = self._img.shape.a / self._nsamples
        for i in range(self._nsamples):
            img = int_translate(self._img, int(sample_slope * i))
            pw = img.local_power_spectra(
                key=key,
                upsample_factor=upsample_factor,
                dims="rya",
            ).proj("r")
            power_spectra.append(pw)
        return np.stack(power_spectra, axis=0).mean(axis=0)

    def get_peak(
        self,
        range_y: tuple[float, float],
        range_a: tuple[float, float],
        up_y: int = 1,
        up_a: int = 1,
    ):
        """
        Get the peak in the power spectrum of the image in subpixel precision.

        This method runs a local power spectrum for the given area and returns the
        peak info.

        Parameters
        ----------
        range_y : (float, float)
            Range of power spectrum analysis in y-direction.
        range_a : (float, float)
            Range of power spectrum analysis in a-direction.
        up_y : int, default is 1
            Upsampling factor in y-direction.
        up_a : int, default is 1
            Upsampling factor in a-direction.

        Returns
        -------
        PeakInfo
            The peak info object.
        """
        y0i, y1i, y_pad0, y_pad1 = _get_boundary(range_y, up_y)
        a0i, a1i, a_pad0, a_pad1 = _get_boundary(range_a, up_a)
        power = self.dft(
            key=ip.slicer.y[y0i:y1i].a[a0i:a1i],
            upsample_factor=[1, up_y, up_a],
        )
        ylen, alen = power.shape.y, power.shape.a
        power_input = power[y_pad0 : ylen - y_pad1, a_pad0 : alen - a_pad1]
        ymax, amax = np.unravel_index(
            np.argmax(power_input),
            shape=power_input.shape,
        )
        return PeakInfo(
            ymax + y0i * up_y + y_pad0,
            amax + a0i * up_a + a_pad0,
            (self._img.shape.y, self._img.shape.a),
            (up_y, up_a),
        )


def _get_boundary(rng: tuple[float, float], up_y: int):
    # 0   y0i  y0      y1  y_pad1
    # |-----|--|--------|--|----->
    y0, y1 = rng
    y0i, y1i = floorint(y0), ceilint(y1)
    y1i = max(y1i, y0i + 1)
    # these should be >0
    y_pad0 = ceilint((y0 - y0i) * up_y)
    y_pad1 = floorint((y1i - y1) * up_y)
    return y0i, y1i, y_pad0, y_pad1


class PeakInfo:
    """Peak info object that will be returned by PeakDetector.get_peak"""

    def __init__(
        self, y: int, a: int, shape: tuple[int, int], upsampling: tuple[int, int]
    ):
        self._y_abs = y
        self._a_abs = a
        self._shape = shape
        self._upsampling = upsampling

    @cached_property
    def y(self) -> float:
        """The y peak position in subpixel unit."""
        return self._y_abs / self._upsampling[0]

    @cached_property
    def a(self) -> float:
        """The a peak position in subpixel unit."""
        return self._a_abs / self._upsampling[1]

    @cached_property
    def yfreq(self) -> float:
        """The y peak frequency."""
        size = self._shape[0] * self._upsampling[0]
        return np.fft.fftfreq(size)[self._y_abs]

    @cached_property
    def afreq(self) -> float:
        """The a peak frequency."""
        size = self._shape[1] * self._upsampling[1]
        return np.fft.fftfreq(size)[self._a_abs]
