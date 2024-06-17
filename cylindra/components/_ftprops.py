from __future__ import annotations

import math
from typing import NamedTuple

import impy as ip
import numpy as np
import polars as pl
from numpy.typing import NDArray

from cylindra._dask import Delayed, delayed
from cylindra.components._cylinder_params import CylinderParameters
from cylindra.components._peak import FTPeakInfo, PeakDetector
from cylindra.components.spline import SplineConfig
from cylindra.const import PropertyNames as H
from cylindra.const import nm
from cylindra.cyltransform import get_polar_image, get_polar_image_task
from cylindra.utils import ceilint, floorint, roundint


class LatticeParams(NamedTuple):
    """Lattice parameters."""

    rise_angle: float
    rise_length: nm
    pitch: nm
    spacing: nm
    skew: float
    twist: float
    npf: int
    start: int

    def to_polars(self) -> pl.DataFrame:
        """Convert named tuple into a polars DataFrame."""
        return pl.DataFrame([self], schema=self.polars_schema())

    @staticmethod
    def polars_schema() -> list[tuple[str, type[pl.DataType]]]:
        """Return the schema of the polars DataFrame."""
        return [
            (H.rise, pl.Float32),
            (H.rise_length, pl.Float32),
            (H.pitch, pl.Float32),
            (H.spacing, pl.Float32),
            (H.skew, pl.Float32),
            (H.twist, pl.Float32),
            (H.npf, pl.UInt8),
            (H.start, pl.Int8),
        ]


class LatticeAnalyzer:
    def __init__(self, config: SplineConfig):
        self._cfg = config

    @property
    def config(self) -> SplineConfig:
        return self._cfg

    def estimate_lattice_params(
        self,
        img: ip.ImgArray | ip.LazyImgArray,
        coords: NDArray[np.float32],
        radius: nm,
        nsamples: int = 8,
    ) -> LatticeParams:
        """Estimate lattice parameters from a Cartesian input."""
        pol = get_polar_image(img, coords, radius)
        return self.estimate_lattice_params_polar(pol, radius, nsamples)

    def estimate_lattice_params_task(
        self,
        img: ip.ImgArray | ip.LazyImgArray,
        coords: NDArray[np.float32],
        radius: nm,
        nsamples: int = 8,
    ) -> Delayed[LatticeParams]:
        task = get_polar_image_task(img, coords, radius)
        return self.estimate_lattice_params_polar_delayed(task, radius, nsamples)

    def estimate_lattice_params_polar(
        self, img: ip.ImgArray, radius: nm, nsamples: int = 8
    ) -> LatticeParams:
        """Estimate lattice parameters from a cylindric input."""
        img = img - float(img.mean())  # normalize.
        peak_det = PeakDetector(img, nsamples=nsamples)
        peakh = self.get_peak_h(peak_det, img, radius)
        peakv = self.get_peak_v(peak_det, img, peakh.a)

        cparams = self.get_params(img, peakh, peakv, radius)
        return LatticeParams(
            rise_angle=cparams.rise_angle,
            rise_length=cparams.rise_length,
            pitch=cparams.pitch,
            spacing=cparams.spacing,
            skew=cparams.skew,
            twist=cparams.twist,
            npf=cparams.npf,
            start=cparams.start,
        )

    estimate_lattice_params_polar_delayed = delayed(estimate_lattice_params_polar)

    # y-axis
    # ^           + <- peakv
    # |
    # |  +      +      + <- peakh
    # |
    # |       +
    # +--------------------> a-axis

    # Transformation around `peakh``.
    # This analysis measures skew angle and protofilament number.

    def get_peak_h(self, peak_det: PeakDetector, img: ip.ImgArray, radius: nm):
        return peak_det.get_peak(**self._params_h(img, radius))

    def get_peak_v(self, peak_det: PeakDetector, img: ip.ImgArray, npf: float):
        return peak_det.get_peak(**self._params_v(img, npf))

    def get_ps_h(self, peak_det: PeakDetector, img: ip.ImgArray, radius: nm):
        return peak_det.get_local_power_spectrum(**self._params_h(img, radius))

    def get_ps_v(self, peak_det: PeakDetector, img: ip.ImgArray, npf: float):
        return peak_det.get_local_power_spectrum(**self._params_v(img, npf))

    def _params_h(self, img: ip.ImgArray, radius: nm):
        spacing_arr = self._cfg.spacing_range.asarray()[np.newaxis]
        y_factor = np.abs(radius / spacing_arr / img.shape.a * img.shape.y)
        tan_twist_min, tan_twist_max = (
            math.tan(math.radians(s)) for s in self._cfg.twist_range.aslist()
        )
        npf_min_max = self._cfg.npf_range.asarray()
        return {
            "range_y": (
                np.min(tan_twist_min * y_factor * npf_min_max),
                np.max(tan_twist_max * y_factor * npf_min_max),
            ),
            "range_a": self.get_arange(img),
            "up_y": max(int(21600 / img.shape.y), 1),
            "up_a": 20,
        }

    def _params_v(self, img: ip.ImgArray, npf: float):
        tan_rise_min, tan_rise_max = (
            math.tan(math.radians(s)) for s in self._cfg.rise_range.aslist()
        )
        ya_scale_ratio = img.scale.y / img.scale.a
        _y_min = img.shape.y * img.scale.y / self._cfg.spacing_range.max
        _y_max = img.shape.y * img.scale.y / self._cfg.spacing_range.min
        _a_min = _y_max * tan_rise_min * img.shape.a / img.shape.y / ya_scale_ratio
        _a_max = _y_min * tan_rise_max * img.shape.a / img.shape.y / ya_scale_ratio
        _a_min, _a_max = sorted(
            [_a_min * self._cfg.rise_sign, _a_max * self._cfg.rise_sign]
        )
        return {
            "range_y": (_y_min, _y_max),
            "range_a": (
                floorint(max(-npf / 2, _a_min)),
                ceilint(min(npf / 2, _a_max)) + 1,
            ),
            "up_y": max(int(6000 / img.shape.y), 1),
            "up_a": 20,
        }

    def get_params(
        self,
        img: ip.ImgArray,
        peakh: FTPeakInfo,
        peakv: FTPeakInfo,
        radius: nm,
    ) -> CylinderParameters:
        npf_f = peakh.a
        npf = roundint(npf_f)
        ya_scale_ratio = img.scale.y / img.scale.a

        tan_rise = peakv.afreq / peakv.yfreq * ya_scale_ratio
        tan_skew = peakh.yfreq / peakh.afreq / ya_scale_ratio

        # NOTE: Values dependent on peak{x}.afreq are not stable against radius change.
        # peak{x}.afreq * radius is stable. r-dependent ones are marked as "f(r)" here.
        return CylinderParameters(
            skew=math.degrees(math.atan(tan_skew)),  # f(r)
            rise_angle_raw=math.degrees(math.atan(tan_rise)),  # f(r)
            pitch=img.scale.y / peakv.yfreq,
            radius=radius,
            npf=npf,
            rise_sign=self._cfg.rise_sign,
        )

    def params_to_peaks(
        self,
        img: ip.ImgArray,
        params: CylinderParameters,
    ) -> tuple[FTPeakInfo, FTPeakInfo]:
        ya_scale_ratio = img.scale.y / img.scale.a
        tan_skew = math.tan(math.radians(params.skew))
        peakv = FTPeakInfo(
            y=img.scale.y / params.pitch * img.shape.y,
            a=params.start * params.rise_sign,
            shape=img.shape,
        )
        peakh = FTPeakInfo(
            y=tan_skew * ya_scale_ratio * params.npf / img.shape.a * img.shape.y,
            a=params.npf,
            shape=img.shape,
        )
        return peakh, peakv

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
