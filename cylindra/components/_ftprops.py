from __future__ import annotations

from typing import NamedTuple
import math
import numpy as np
from numpy.typing import NDArray
import impy as ip
import polars as pl
from cylindra.const import nm, PropertyNames as H, Mode
from cylindra.components._peak import PeakDetector
from cylindra.components.spline import SplineConfig
from cylindra.utils import map_coordinates, ceilint, roundint, floorint


class LatticeParams(NamedTuple):
    """Lattice parameters."""

    rise_angle: float
    rise_length: nm
    spacing: nm
    skew_tilt: float
    skew_angle: float
    npf: int
    start: float

    def to_polars(self) -> pl.DataFrame:
        """Convert named tuple into a polars DataFrame."""
        return pl.DataFrame([self], schema=self.polars_schema())

    @staticmethod
    def polars_schema() -> list[tuple[str, type[pl.DataType]]]:
        """Return the schema of the polars DataFrame."""
        return [
            (H.rise, pl.Float32),
            (H.rise_length, pl.Float32),
            (H.spacing, pl.Float32),
            (H.skew_tilt, pl.Float32),
            (H.skew, pl.Float32),
            (H.npf, pl.UInt8),
            (H.start, pl.Float32),
        ]


class LatticeAnalyzer:
    def __init__(self, config: SplineConfig):
        self._cfg = config

    def polar_ft_params(
        self, img: ip.ImgArray, radius: nm, nsamples: int = 8
    ) -> LatticeParams:
        """Detect the peak position and calculate the local lattice parameters."""
        img = img - float(img.mean())  # normalize.
        up_a = 40
        peak_det = PeakDetector(img, nsamples=nsamples)
        ya_scale_ratio = img.scale.y / img.scale.a

        # y-axis
        # ^           + <- peakv
        # |
        # |  +      +      + <- peakh
        # |
        # |       +
        # +--------------------> a-axis

        # Transformation around `peakh``.
        # This analysis measures skew angle and protofilament number.
        spacing_arr = np.array(self._cfg.spacing_range.aslist())[np.newaxis]
        y_factor = np.abs(radius / spacing_arr / img.shape.a * img.shape.y / 2)
        tan_skew_min, tan_skew_max, tan_rise_min, tan_rise_max = (
            math.tan(math.radians(s))
            for s in [self._cfg.skew_range.aslist() + self._cfg.rise_range.aslist()]
        )
        npf_min_max = np.array(self._cfg.npf_range.aslist())

        peakh = peak_det.get_peak(
            range_y=(
                np.min(tan_skew_min * y_factor * npf_min_max),
                np.max(tan_skew_max * y_factor * npf_min_max),
            ),
            range_a=self.get_arange(img),
            up_y=max(int(21600 / img.shape.y), 1),
            up_a=up_a,
        )
        npf_f = peakh.a
        npf = roundint(npf_f)

        # Transformation around `peakv`.
        _y_min = img.shape.y * img.scale.y / self._cfg.spacing_range.max
        _y_max = img.shape.y * img.scale.y / self._cfg.spacing_range.min
        _a_min = _y_max * tan_rise_min * img.shape.a / img.shape.y / ya_scale_ratio
        _a_max = _y_min * tan_rise_max * img.shape.a / img.shape.y / ya_scale_ratio
        _a_min, _a_max = sorted(
            [_a_min * self._cfg.rise_sign, _a_max * self._cfg.rise_sign]
        )
        peakv = peak_det.get_peak(
            range_y=(_y_min, _y_max),
            range_a=(
                floorint(max(-npf_f / 2, _a_min)),
                ceilint(min(npf_f / 2, _a_max)) + 1,
            ),
            up_y=max(int(6000 / img.shape.y), 1),
            up_a=up_a,
        )

        tan_rise = peakv.afreq / peakv.yfreq * ya_scale_ratio * self._cfg.rise_sign
        tan_skew_tilt = peakh.yfreq / peakh.afreq / ya_scale_ratio

        # NOTE: Values dependent on peak{x}.afreq are not stable against radius change.
        # peak{x}.afreq * radius is stable. r-dependent ones are marked as "f(r)" here.
        perimeter = 2 * np.pi * radius
        rise = math.atan(tan_rise)  # f(r)
        rise_len = tan_rise * perimeter / npf
        yspace = img.scale.y / peakv.yfreq
        skew_tilt = math.atan(tan_skew_tilt)  # f(r)
        skew = tan_skew_tilt * 2 * yspace / radius
        start = perimeter * tan_rise / (yspace * (1 + tan_rise * tan_skew_tilt))

        return LatticeParams(
            rise_angle=math.degrees(rise),
            rise_length=rise_len,
            spacing=yspace,
            skew_tilt=math.degrees(skew_tilt),
            skew_angle=math.degrees(skew),
            npf=npf,
            start=start,
        )

    def ft_params(
        self,
        img: ip.ImgArray | ip.LazyImgArray,
        coords: NDArray[np.float32],
        radius: nm,
        nsamples: int = 8,
    ) -> LatticeParams:
        """Calculate the local lattice parameters from a Cartesian input."""
        return self.polar_ft_params(
            get_polar_image(img, coords, radius), radius, nsamples
        )

    def get_yrange(self, img: ip.ImgArray) -> tuple[int, int]:
        """Get the range of y-axis in the polar image."""
        ylength_nm = img.shape.y * img.scale.y
        return (
            ceilint(ylength_nm / self._cfg.spacing_range.max) - 1,
            ceilint(ylength_nm / self._cfg.spacing_range.min),
        )

    def get_arange(self, img: ip.ImgArray) -> tuple[int, int]:
        """Get the range of a-axis in the polar image."""
        return self._cfg.npf_range.min, self._cfg.npf_range.max + 1

    def mask_spectra(self, polar: ip.ImgArray) -> ip.ImgArray:
        """Mask the spectra of the polar image."""
        polar_ft = polar.fft(shift=False, dims="rya")
        mask = ip.zeros(polar.shape, dtype=np.bool_, axes="rya")
        mask[ip.slicer.y[slice(*self.get_yrange(polar))]] = True
        mask[ip.slicer.a[slice(*self.get_arange(polar))]] = True
        polar_ft[~mask] = 0.0
        return polar_ft.ifft(shift=False, dims="rya")


def get_polar_image(
    img: ip.ImgArray | ip.LazyImgArray,
    coords: NDArray[np.float32],
    radius: nm,
    order: int = 3,
):
    """Convert the input image into a polar image."""
    polar = map_coordinates(img, coords, order=order, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    a_scale = 2 * np.pi * radius / polar.shape.a
    return polar.set_scale(r=img.scale.x, y=img.scale.x, a=a_scale, unit=img.scale_unit)
