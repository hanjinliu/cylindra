from __future__ import annotations

from functools import cached_property
from cylindra.utils import ceilint, floorint
import numpy as np
import impy as ip


class PeakDetector:
    """A power spectrum peak detector for a given image."""

    def __init__(self, img: ip.ImgArray):
        self._img = img

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
        y0, y1 = range_y
        a0, a1 = range_a
        y0i, y1i = floorint(y0), ceilint(y1)
        a0i, a1i = floorint(a0), ceilint(a1)
        y1i = max(y1i, y0i + 1)
        a1i = max(a1i, a0i + 1)
        power = self._img.local_power_spectra(
            key=ip.slicer.y[y0i:y1i].a[a0i:a1i],
            upsample_factor=[1, up_y, up_a],
            dims="rya",
        ).proj("r")

        # these should be >0
        y_pad0 = ceilint((y0 - y0i) * up_y)
        y_pad1 = floorint((y1i - y1) * up_y)
        a_pad0 = ceilint((a0 - a0i) * up_a)
        a_pad1 = floorint((a1i - a1) * up_a)

        ymax, amax = np.unravel_index(
            np.argmax(
                power[y_pad0 : power.shape.y - y_pad1, a_pad0 : power.shape.a - a_pad1]
            ),
            shape=power.shape,
        )

        return PeakInfo(
            ymax + y0i * up_y + y_pad0,
            amax + a0i * up_a + a_pad0,
            (self._img.shape.y, self._img.shape.a),
            (up_y, up_a),
        )


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
    def yfreq(self):
        """The y peak frequency."""
        size = self._shape[0] * self._upsampling[0]
        return np.fft.fftfreq(size)[self._y_abs]

    @cached_property
    def afreq(self):
        """The a peak frequency."""
        size = self._shape[1] * self._upsampling[1]
        return np.fft.fftfreq(size)[self._a_abs]
