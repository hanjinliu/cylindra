from __future__ import annotations

from typing import NamedTuple

import impy as ip
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from cylindra.utils import ceilint, centroid, floorint


def int_translate(img: ip.ImgArray, shift: int, axis: str = "a") -> ip.ImgArray:
    """Translate the image by integer pixels in the given axis."""
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
            ).mean(axis="r")
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
        up_y : int, default 1
            Upsampling factor in y-direction.
        up_a : int, default 1
            Upsampling factor in a-direction.

        Returns
        -------
        PeakInfo
            The peak info object.
        """
        ps, y0, a0 = self._local_ps_and_offset(range_y, range_a, up_y, up_a)
        ymax, amax = np.unravel_index(np.argmax(ps), ps.shape)
        (ymaxsub, amaxsub), value = find_peak(ps, index=(ymax, amax), nrepeat=1)
        return FTPeakInfo(
            ymaxsub + y0,
            amaxsub + a0,
            (self._img.shape.y, self._img.shape.a),
            (up_y, up_a),
        )

    def get_local_power_spectrum(
        self,
        range_y: tuple[float, float],
        range_a: tuple[float, float],
        up_y: int = 1,
        up_a: int = 1,
    ) -> ip.ImgArray:
        """Get the up-sampled local power spectruc of this peak."""
        return self._local_ps_and_offset(range_y, range_a, up_y, up_a)[0]

    def _local_ps_and_offset(
        self,
        range_y: tuple[float, float],
        range_a: tuple[float, float],
        up_y: int = 1,
        up_a: int = 1,
    ) -> tuple[ip.ImgArray, float, float]:
        y0i, y1i, y_pad0, y_pad1 = _get_boundary(range_y, up_y)
        a0i, a1i, a_pad0, a_pad1 = _get_boundary(range_a, up_a)
        ps0 = self.dft(
            key=ip.slicer.y[y0i:y1i].a[a0i:a1i],
            upsample_factor=[1, up_y, up_a],
        )
        ylen, alen = ps0.shape.y, ps0.shape.a
        ps = ps0[y_pad0 : ylen - y_pad1, a_pad0 : alen - a_pad1]
        return ps, y0i * up_y + y_pad0, a0i * up_a + a_pad0


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


class FTPeakInfo:
    """Peak info object that will be returned by PeakDetector.get_peak"""

    def __init__(
        self,
        y: float,
        a: float,
        shape: tuple[int, int],
        upsampling: tuple[int, int] = (1, 1),
    ):
        self._y_abs = y
        self._a_abs = a
        self._shape = shape
        self._upsampling = upsampling

    @property
    def y(self) -> float:
        """The y peak position in subpixel unit."""
        return self._y_abs / self._upsampling[0]

    @property
    def a(self) -> float:
        """The a peak position in subpixel unit."""
        return self._a_abs / self._upsampling[1]

    @property
    def yfreq(self) -> float:
        """The y peak frequency."""
        size = self._shape[0] * self._upsampling[0]
        return get_fftfreq(self._y_abs, size)

    @property
    def afreq(self) -> float:
        """The a peak frequency."""
        size = self._shape[1] * self._upsampling[1]
        return get_fftfreq(self._a_abs, size)

    def shift_to_center(self) -> FTPeakInfo:
        """Shift the peak to the center of the image."""
        fcenter0 = (self._shape[0] * self._upsampling[0] - 1) // 2 + 1
        y1 = self._y_abs + fcenter0
        if y1 >= self._shape[0] * self._upsampling[0]:
            y1 -= self._shape[0] * self._upsampling[0]
        return FTPeakInfo(
            shift_to_center(self._y_abs, self._shape[0] * self._upsampling[0]),
            shift_to_center(self._a_abs, self._shape[1] * self._upsampling[1]),
            self._shape,
            self._upsampling,
        )


def get_fftfreq(f: float, size: int) -> float:
    """Equivalent to np.fft.fftfreq(size)[f] but allow f to be float."""
    fcenter = (size - 1) // 2 + 1
    if f < fcenter:
        return f / size
    else:
        return (f - fcenter - size // 2) / size


def shift_to_center(idx: int, size: int):
    fcenter = size // 2
    pos = idx + fcenter
    if pos >= size:
        pos -= size
    return pos


class NDPeak(NamedTuple):
    """A peak in a nD array."""

    pos: tuple[float, ...]
    value: float


def find_peak(
    arr: NDArray[np.float32],
    index: NDArray[np.float32] | None = None,
    nrepeat: int = 3,
    n: int = 11,
    dx: float = 0.45,
) -> NDPeak:
    """Iteratively sample sub-meshes to find the peak of 3D array."""
    if index is None:
        index = np.unravel_index(np.argmax(arr), arr.shape)
    argmax = index
    if nrepeat > 0:
        value = 0.0
        arr_filt = ndi.spline_filter(arr, output=np.float32, mode="reflect")
        for _ in range(nrepeat):
            argmax, value = _find_peak_once(arr_filt, argmax, arr.shape, dx=dx, n=n)
            dx /= n
    else:
        value = arr[tuple(argmax)]
    return NDPeak(tuple(argmax), value)


def _find_peak_once(
    arr: NDArray[np.float32],
    index: NDArray[np.float32],
    shape: tuple[int, ...],
    dx: float = 1.0,
    n: int = 11,
):
    # width of the mesh subpixels is 2 * dx / n
    mesh = np.stack(
        np.meshgrid(
            *[np.linspace(-dx + x, dx + x, n, dtype=np.float32) for x in index],
            indexing="ij",
        ),
        axis=0,
    )

    mapped = ndi.map_coordinates(
        arr,
        mesh,
        order=3,
        mode="reflect",
        prefilter=False,
    )
    argmax = np.unravel_index(np.argmax(mapped), mapped.shape)
    value = mapped[argmax]
    center = (np.array(mapped.shape) - 1) / 2
    argmax_centered = np.array(argmax, dtype=np.float32) - center
    peak = argmax_centered * 2 * dx / (n - 1) + index
    peak = np.minimum(peak, np.array(shape) - 1)
    peak = np.maximum(peak, 0.0)
    return peak, value


def find_centroid_peak(prof: NDArray[np.float32], inner: float, outer: float) -> float:
    imax = np.nanargmax(prof)
    imax_sub = -1.0
    count = 0
    while abs(imax - imax_sub) > 1e-2:
        if imax_sub > 0:
            imax = imax_sub
        imax_sub = centroid(prof, ceilint(imax - inner), int(imax + outer) + 1)
        count += 1
        if count > 100:
            break
        if imax_sub != imax_sub:
            raise ValueError(f"Centroid encountered NaN in the {count}-th trial.")
    return imax_sub
