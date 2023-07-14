from __future__ import annotations

from typing import NamedTuple
import numpy as np
import impy as ip
import polars as pl
from cylindra.const import nm, PropertyNames as H, GlobalVariables as GVar, Mode
from cylindra.components._peak import PeakDetector
from cylindra.components.cyl_spline import rise_to_start

from cylindra.utils import map_coordinates, ceilint, roundint


class LocalParams(NamedTuple):
    """Local lattice parameters."""

    rise: float
    spacing: nm
    skew: float
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
            (H.spacing, pl.Float32),
            (H.skew, pl.Float32),
            (H.npf, pl.UInt8),
            (H.start, pl.Float32),
        ]


def polar_ft_params(img: ip.ImgArray, radius: nm) -> LocalParams:
    """Detect the peak position and calculate the local lattice parameters."""
    perimeter: nm = 2 * np.pi * radius
    img = img - img.mean()  # normalize.

    peak_det = PeakDetector(img)

    # y-axis
    # ^           + <- peak0
    # |
    # |  +      +      + <- peak1
    # |
    # |       +
    # +--------------------> a-axis

    # First transform around the expected length of y-pitch.
    npfrange = ceilint(GVar.npf_max / 2)
    peakv = peak_det.get_peak(
        range_y=(
            img.shape.y * img.scale.y / GVar.spacing_max,
            img.shape.y * img.scale.y / GVar.spacing_min,
        ),
        range_a=(-npfrange, npfrange + 1),
        up_y=max(int(6000 / img.shape.y), 1),
        up_a=20,
    )

    rise = np.arctan(peakv.afreq / peakv.yfreq) * GVar.rise_sign
    yspace = 1.0 / peakv.yfreq * img.scale.y

    # Second, transform around 13 pf lateral periodicity.
    # This analysis measures skew angle and protofilament number.
    y_factor = abs(radius / yspace / img.shape.a * img.shape.y / 2)
    tan_min = np.tan(np.deg2rad(GVar.skew_min))
    tan_max = np.tan(np.deg2rad(GVar.skew_max))
    npf_arr = np.array([GVar.npf_min, GVar.npf_max])

    peakh = peak_det.get_peak(
        range_y=(
            np.min(tan_min * y_factor * npf_arr),
            np.max(tan_max * y_factor * npf_arr),
        ),
        range_a=get_arange(img),
        up_y=max(int(21600 / img.shape.y), 1),
        up_a=20,
    )

    skew = np.arctan(peakh.yfreq / peakh.afreq * 2 * yspace / radius)
    start = rise_to_start(rise, yspace, skew=skew, perimeter=perimeter)
    npf = peakh.a

    return LocalParams(
        rise=np.rad2deg(rise),
        spacing=yspace,
        skew=np.rad2deg(skew),
        npf=roundint(npf),
        start=start,
    )


def ft_params(
    img: ip.ImgArray | ip.LazyImgArray, coords: np.ndarray, radius: nm
) -> LocalParams:
    return polar_ft_params(get_polar_image(img, coords), radius)


def get_polar_image(
    img: ip.ImgArray | ip.LazyImgArray, coords: np.ndarray, order: int = 3
):
    polar = map_coordinates(img, coords, order=order, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x, unit=img.scale_unit)
    return polar


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
