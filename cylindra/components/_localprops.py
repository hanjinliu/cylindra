from __future__ import annotations

from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray
import impy as ip
import polars as pl
from scipy import ndimage as ndi
from dask import delayed
from cylindra.const import nm, PropertyNames as H, GlobalVariables as GVar, Mode
from cylindra.components._peak import PeakDetector
from cylindra.components.cyl_spline import rise_to_start

from cylindra.utils import (
    map_coordinates,
    ceilint,
    roundint,
)


def tandg(x):
    """Tangent in degree."""
    return np.tan(np.deg2rad(x))


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
            (H.nPF, pl.UInt8),
            (H.start, pl.Float32),
        ]


def polar_ft_params(img: ip.ImgArray, radius: nm) -> LocalParams:
    perimeter: nm = 2 * np.pi * radius
    npfmin = GVar.npf_min
    npfmax = GVar.npf_max
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
    ylength_nm = img.shape.y * img.scale.y
    npfrange = ceilint(npfmax / 2)

    peak0 = peak_det.get_peak(
        range_y=(
            ceilint(ylength_nm / GVar.spacing_max) - 1,
            ceilint(ylength_nm / GVar.spacing_min),
        ),
        range_a=(-npfrange, npfrange + 1),
        up_y=max(int(6000 / img.shape.y), 1),
        up_a=20,
    )

    rise = np.arctan(-peak0.afreq / peak0.yfreq)
    yspace = 1.0 / peak0.yfreq * img.scale.y

    # Second, transform around 13 pf lateral periodicity.
    # This analysis measures skew angle and protofilament number.
    y_factor = abs(radius / yspace / img.shape.a * img.shape.y / 2)

    peak1 = peak_det.get_peak(
        range_y=(
            ceilint(tandg(GVar.skew_min) * y_factor * npfmin) - 1,
            ceilint(tandg(GVar.skew_max) * y_factor * npfmax),
        ),
        range_a=(npfmin, npfmax),
        up_y=max(int(21600 / img.shape.y), 1),
        up_a=20,
    )

    skew = np.arctan(peak1.yfreq / peak1.afreq * 2 * yspace / radius)
    start = rise_to_start(rise, yspace, skew=skew, perimeter=perimeter)
    npf = peak1.a

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


@delayed
def try_all_npf(img: ip.ImgArray | ip.LazyImgArray, coords: np.ndarray):
    polar = get_polar_image(img, coords)
    prof = ndi.spline_filter1d(
        polar.proj("ry").value, output=np.float32, mode=Mode.wrap
    )
    score = list[float]()
    npf_list = list(range(GVar.npf_min, GVar.npf_max + 1))
    for npf in npf_list:
        single_shift = prof.size / npf
        sum_img = prof + sum(ndi.shift(prof, single_shift * i) for i in range(1, npf))
        avg: NDArray[np.float32] = sum_img / npf
        score.append(avg.max() - avg.min())
    return npf_list[np.argmax(score)]


def get_polar_image(img: ip.ImgArray | ip.LazyImgArray, coords: np.ndarray):
    polar = map_coordinates(img, coords, order=3, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x, unit=img.scale_unit)
    return polar
