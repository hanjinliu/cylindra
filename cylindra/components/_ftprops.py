from __future__ import annotations

from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray
import impy as ip
import polars as pl
from cylindra.const import nm, PropertyNames as H, GlobalVariables as GVar, Mode
from cylindra.components._peak import PeakDetector
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


def polar_ft_params(img: ip.ImgArray, radius: nm, nsamples: int = 8) -> LatticeParams:
    """Detect the peak position and calculate the local lattice parameters."""
    img = img - img.mean()  # normalize.
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
    spacing_arr = np.array([GVar.spacing_min, GVar.spacing_max])[np.newaxis]
    y_factor = np.abs(radius / spacing_arr / img.shape.a * img.shape.y / 2)
    tan_skew_min, tan_skew_max = np.tan(np.deg2rad([GVar.skew_min, GVar.skew_max]))
    tan_rise_min, tan_rise_max = np.tan(np.deg2rad([GVar.rise_min, GVar.rise_max]))
    npf_min_max = np.array([GVar.npf_min, GVar.npf_max])

    peakh = peak_det.get_peak(
        range_y=(
            np.min(tan_skew_min * y_factor * npf_min_max),
            np.max(tan_skew_max * y_factor * npf_min_max),
        ),
        range_a=get_arange(img),
        up_y=max(int(21600 / img.shape.y), 1),
        up_a=up_a,
    )
    npf_f = peakh.a
    npf = roundint(npf_f)

    # Transformation around `peakv`.
    _y_min = img.shape.y * img.scale.y / GVar.spacing_max
    _y_max = img.shape.y * img.scale.y / GVar.spacing_min
    _a_min = _y_max * tan_rise_min * img.shape.a / img.shape.y / ya_scale_ratio
    _a_max = _y_min * tan_rise_max * img.shape.a / img.shape.y / ya_scale_ratio
    _a_min, _a_max = np.sort([_a_min * GVar.rise_sign, _a_max * GVar.rise_sign])
    peakv = peak_det.get_peak(
        range_y=(_y_min, _y_max),
        range_a=(
            floorint(max(-npf_f / 2, _a_min)),
            ceilint(min(npf_f / 2, _a_max)) + 1,
        ),
        up_y=max(int(6000 / img.shape.y), 1),
        up_a=up_a,
    )

    tan_rise = peakv.afreq / peakv.yfreq / ya_scale_ratio * GVar.rise_sign
    tan_skew_tilt = peakh.yfreq / peakh.afreq * ya_scale_ratio

    # NOTE: Values dependent on peak{x}.afreq are not stable against radius change.
    # peak{x}.afreq * radius is stable. r-dependent ones are marked as "f(r)" here.
    rise = np.arctan(tan_rise)  # f(r)
    rise_len = tan_rise * 2 * np.pi * radius / npf
    yspace = img.scale.y / peakv.yfreq
    skew_tilt = np.arctan(tan_skew_tilt)  # f(r)
    skew = tan_skew_tilt * 2 * yspace / radius
    start = 2 * np.pi * radius * tan_rise / (yspace * (1 + tan_rise * tan_skew_tilt))

    return LatticeParams(
        rise_angle=np.rad2deg(rise),
        rise_length=rise_len,
        spacing=yspace,
        skew_tilt=np.rad2deg(skew_tilt),
        skew_angle=np.rad2deg(skew),
        npf=npf,
        start=start,
    )


def ft_params(
    img: ip.ImgArray | ip.LazyImgArray,
    coords: NDArray[np.float32],
    radius: nm,
    nsamples: int = 8,
) -> LatticeParams:
    """Calculate the local lattice parameters from a Cartesian input."""
    return polar_ft_params(get_polar_image(img, coords, radius), radius, nsamples)


def get_polar_image(
    img: ip.ImgArray | ip.LazyImgArray,
    coords: NDArray[np.float32],
    radius: nm,
    order: int = 3,
):
    polar = map_coordinates(img, coords, order=order, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    a_scale = 2 * np.pi * radius / polar.shape.a
    return polar.set_scale(r=img.scale.x, y=img.scale.x, a=a_scale, unit=img.scale_unit)


def get_yrange(img: ip.ImgArray) -> tuple[int, int]:
    """Get the range of y-axis in the polar image."""
    ylength_nm = img.shape.y * img.scale.y
    return (
        ceilint(ylength_nm / GVar.spacing_max) - 1,
        ceilint(ylength_nm / GVar.spacing_min),
    )


def get_arange(img: ip.ImgArray) -> tuple[int, int]:
    """Get the range of a-axis in the polar image."""
    return GVar.npf_min, GVar.npf_max + 1


def mask_spectra(polar: ip.ImgArray) -> ip.ImgArray:
    """Mask the spectra of the polar image."""
    polar_ft = polar.fft(shift=False, dims="rya")
    mask = ip.zeros(polar.shape, dtype=np.bool_, axes="rya")
    mask[ip.slicer.y[slice(*get_yrange(polar))]] = True
    mask[ip.slicer.a[slice(*get_arange(polar))]] = True
    polar_ft[~mask] = 0.0
    return polar_ft.ifft(shift=False, dims="rya")
