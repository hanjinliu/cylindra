from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Sequence,
    TypeVar,
    overload,
)

import impy as ip
import numpy as np
import polars as pl
from acryo import Molecules, SubtomogramLoader
from acryo.tilt import TiltSeriesModel
from dask import array as da
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from scipy.fft import fft2, ifft2
from scipy.spatial.transform import Rotation
from typing_extensions import Concatenate, ParamSpec

from cylindra._dask import Delayed, compute, delayed
from cylindra.components._ftprops import LatticeAnalyzer, LatticeParams, get_polar_image
from cylindra.components._peak import find_centroid_peak
from cylindra.components.spline import CylSpline
from cylindra.components.tomogram import _straighten
from cylindra.components.tomogram._spline_list import SplineList
from cylindra.components.tomogram._tomo_base import Tomogram
from cylindra.const import ExtrapolationMode, Mode, Ori, nm
from cylindra.const import PropertyNames as H
from cylindra.utils import (
    angle_corr,
    ceilint,
    centroid_2d,
    crop_tomograms,
    map_coordinates_task,
    rotated_auto_zncc,
    roundint,
    set_gpu,
)

if TYPE_CHECKING:
    from typing_extensions import Literal, Self

    from cylindra.components.cylindric import CylinderModel
    from cylindra.components.spline import SplineConfig

    Degenerative = Callable[[ArrayLike], Any]


LOGGER = logging.getLogger("cylindra")


_P = ParamSpec("_P")
_R = TypeVar("_R")


class BatchCallable(Protocol[_P, _R]):
    """
    This protocol enables static type checking of methods decorated with ``@batch_process``.
    The parameter specifier ``_KW`` does not add any information but currently there is not
    quick solution.
    """

    @overload
    def __call__(
        self, i: Literal[None], *args: _P.args, **kwargs: _P.kwargs
    ) -> list[_R]:
        ...

    @overload
    def __call__(self, i: int, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        ...

    @overload
    def __call__(
        self, i: Iterable[int] | None, *args: _P.args, **kwargs: _P.kwargs
    ) -> list[_R]:
        ...

    def __call__(self, i, *args, **kwargs):
        ...


def batch_process(
    func: Callable[Concatenate[CylTomogram, Any, _P], _R]
) -> BatchCallable[_P, _R]:
    """Enable running function for every splines."""

    @wraps(func)
    def _func(self: CylTomogram, i=None, **kwargs):
        if isinstance(i, int):
            out = func(self, i=i, **kwargs)
            return out

        # Determine along which spline function will be executed.
        if i is None:
            i_list = range(len(self.splines))
        elif not hasattr(i, "__iter__"):
            raise TypeError("'i' must be int or iterable of int if specified")
        else:
            i_list = []
            for i_ in i:
                if -len(self.splines) <= i_ < 0:
                    i_list.append(i_ + len(self.splines))
                elif 0 <= i_ < len(self.splines):
                    i_list.append(i_)
                else:
                    raise ValueError(f"Index {i_} is out of bound")

            if len(i_list) > len(set(i_list)):
                raise ValueError("Indices cannot contain duplicated values.")

        # Run function along each spline
        out = []
        for i_ in i_list:
            try:
                result = func(self, i=i_, **kwargs)
            except Exception as e:
                e.args = (f"{e} (Exception at spline-{i_})",)
                raise e
            else:
                out.append(result)

        return out

    return _func


class FitResult:
    def __init__(self, residual: NDArray[np.float64]):
        self.residual = residual

    @property
    def rmsd(self) -> float:
        """Root mean square deviation."""
        return np.sqrt(np.sum(self.residual**2) / self.residual.shape[0])


class CylTomogram(Tomogram):
    """Tomogram with cylindrical splines."""

    def __init__(self):
        super().__init__()
        self._splines = SplineList()

    @property
    def splines(self) -> SplineList:
        """List of splines."""
        return self._splines

    def add_spline(
        self,
        coords: ArrayLike,
        *,
        order: int = 3,
        err_max: nm = 0.5,
        extrapolate: ExtrapolationMode | str = ExtrapolationMode.linear,
        config: SplineConfig | dict[str, Any] = {},
    ) -> None:
        """
        Add spline path to tomogram.

        Parameters
        ----------
        coords : array-like
            (N, 3) array of coordinates. A spline curve that fit it well is added.
        order : int, optional
            Order of spline curve.
        extrapolate : str, optional
            Extrapolation mode of the spline.
        config : SplineConfig or dict, optional
            Configuration for spline fitting.
        """
        _coords = np.asarray(coords)
        ncoords = _coords.shape[0]
        spl = CylSpline(
            order=order,
            config=config,
            extrapolate=extrapolate,
        ).fit(_coords, err_max=err_max)
        interval: nm = 30.0
        length = spl.length()

        n = int(length / interval) + 1
        fit = spl.map(np.linspace(0, 1, n))
        if ncoords <= spl.order and ncoords < fit.shape[0]:
            return self.add_spline(
                fit,
                order=order,
                err_max=err_max,
                extrapolate=extrapolate,
                config=config,
            )

        self.splines.append(spl)
        return None

    @batch_process
    def make_anchors(
        self,
        i: int = None,
        *,
        interval: nm | None = None,
        n: int | None = None,
        max_interval: nm | None = None,
    ):
        """
        Make anchors on spline object(s).

        Parameters
        ----------
        interval : nm, optional
            Anchor intervals.
        n : int, optional
            Number of anchors
        max_interval : nm, optional
            Maximum interval between anchors.

        """
        self.splines[i].make_anchors(interval=interval, n=n, max_interval=max_interval)
        return None

    def align_to_polarity(self, orientation: Ori | str = Ori.MinusToPlus) -> Self:
        """
        Align all the splines in the direction parallel to the given polarity.

        Parameters
        ----------
        orientation : Ori or str, default Ori.MinusToPlus
            To which direction splines will be aligned.

        Returns
        -------
        Tomogram object
            Same object with updated splines.
        """
        orientation = Ori(orientation)
        if orientation is Ori.none:
            raise ValueError("Must be PlusToMinus or MinusToPlus.")
        for i, spl in enumerate(self.splines):
            if spl.orientation is Ori.none:
                raise ValueError(f"Spline-{i} has no orientation.")
            if spl.orientation != orientation:
                try:
                    self.splines[i] = spl.invert()
                except Exception as e:
                    raise type(e)(f"Cannot invert spline-{i}: {e}")
        return self

    @batch_process
    def fit(
        self,
        i: int = None,
        *,
        max_interval: nm = 30.0,
        degree_precision: float = 0.5,
        binsize: int = 1,
        err_max: nm = 1.0,
        edge_sigma: nm = 2.0,
        max_shift: nm = 5.0,
        n_rotations: int = 5,
    ) -> FitResult:
        """
        Roughly fit splines to cylindrical structures.

        Subtomograms will be sampled at every ``max_interval`` nm. In dense mode,
        Subtomograms will be masked relative to XY-plane, using sigmoid function.
        Sharpness of the sigmoid function is determined by ``dense_mode_sigma``
        (``dense_mode_sigma=0`` corresponds to a step function).

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to fit.
        max_interval : nm, default 30.0
            Maximum interval of sampling points in nm unit.
        degree_precision : float, default 0.5
            Precision of xy-tilt degree in angular correlation.
        binsize : int, default 1
            Multiscale bin size used for fitting.
        edge_sigma : nm, default 2.0
            Sharpness of mask at the edges. If not None, fitting will be executed after regions
            outside the cylinder are masked. Soft mask is important for precision because sharp
            changes in intensity cause strong correlation at the edges.
        max_shift: nm, default 5.0
            Maximum shift from the true center of the cylinder. This parameter is used in phase
            cross correlation.
        n_rotations : int, default 5
            Number of rotations to be tested during finding the cylinder center.

        Returns
        -------
        FitResult
            Result of fitting.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.fit, i={i}")
        spl = self.splines[i]
        anc = spl.prep_anchor_positions(max_interval=max_interval)
        subtomograms, interval, scale = self._prep_fit_spline(spl, anc, binsize)

        with set_gpu():
            subtomograms = _soft_mask_edges(subtomograms, spl, anc, scale, edge_sigma)
            ds = spl.map(anc, der=1)
            yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
            degree_max = 14.0
            nrots = roundint(degree_max / degree_precision) + 1

            # Angular correlation
            out = dask_angle_corr(subtomograms, yx_tilt, nrots=nrots)
            refined_tilt_deg = np.array(out)
            refined_tilt_rad = np.deg2rad(refined_tilt_deg)

            # If subtomograms are sampled at short intervals, angles should be smoothened to
            # avoid overfitting.
            size = 2 * roundint(48.0 / interval) + 1
            if size > 1:
                # Mirror-mode padding is "a b c d | c b a".
                refined_tilt_rad = angle_uniform_filter(
                    refined_tilt_rad, size=size, mode=Mode.mirror
                )
                refined_tilt_deg = np.rad2deg(refined_tilt_rad)

            # Rotate subtomograms in YX plane
            for _j, img in enumerate(subtomograms):
                img: ip.ImgArray
                angle = refined_tilt_deg[_j]
                img.rotate(-angle, cval=0, update=True)

            # zx-shift correction by self-ZNCC
            subtomo_proj = subtomograms.mean(axis="y")

            if edge_sigma is not None:
                # Regions outside the mask don't need to be considered.
                xc = int(subtomo_proj.shape.x / 2)
                w = int(spl.config.fit_width / scale / 2)
                subtomo_proj = subtomo_proj[ip.slicer.x[xc - w : xc + w + 1]]

            max_shift_px = max_shift / scale * 2
            pf_ang = 360 / spl.config.npf_range.center
            degrees = np.linspace(-pf_ang / 2, pf_ang / 2, n_rotations) + 180
            shifts = _multi_rotated_auto_zncc(subtomo_proj, degrees, max_shift_px)

        # Update spline coordinates.
        # Because centers of subtomogram are on lattice points of pixel coordinate,
        # coordinates that will be shifted should be converted to integers.
        coords_px = self.nm2pixel(spl.map(anc), binsize=binsize).astype(np.float32)
        coords_px_new = _shift_coords(coords_px, shifts, refined_tilt_rad)
        coords = coords_px_new * scale + self.multiscale_translation(binsize)

        # Update spline parameters
        self.splines[i] = spl.fit(coords, err_max=err_max)
        result = FitResult(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @batch_process
    def fit_centroid(
        self,
        i: int = None,
        *,
        max_interval: nm = 30.0,
        binsize: int = 1,
        err_max: nm = 1.0,
        max_shift: nm = 5.0,
    ) -> FitResult:
        LOGGER.info(f"Running: {self.__class__.__name__}.fit_centroid, i={i}")
        spl = self.splines[i]
        anc = spl.prep_anchor_positions(max_interval=max_interval)
        scale = self.scale * binsize

        # sample subtomograms
        loader = _prep_loader_for_refine(self, spl, anc, binsize)
        subtomograms = ip.asarray(loader.asnumpy(), axes="pzyx").mean(axis="y")[
            ip.slicer.x[::-1]
        ]
        num, lz, lx = subtomograms.shape
        dpx = ceilint(max_shift / scale)
        sl_z = slice(max((lz - 1) // 2 - dpx, 0), min(lz // 2 + dpx + 1, lz))
        sl_x = slice(max((lx - 1) // 2 - dpx, 0), min(lx // 2 + dpx + 1, lx))
        centers = np.stack(
            [centroid_2d(patch, sl_z, sl_x) for patch in subtomograms], axis=0
        )
        shifts = centers - np.column_stack(
            [
                np.full(num, (sl_z.start + sl_z.stop - 1) / 2),
                np.full(num, (sl_x.start + sl_x.stop - 1) / 2),
            ]
        )
        self.splines[i] = spl.shift(anc, shifts=shifts * scale, err_max=err_max)
        result = FitResult(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @batch_process
    def refine(
        self,
        i: int = None,
        *,
        max_interval: nm = 30.0,
        binsize: int = 1,
        err_max: nm = 1.0,
        corr_allowed: float = 0.9,
        max_shift: nm = 2.0,
        n_rotations: int = 3,
    ) -> FitResult:
        """
        Spline refinement using global lattice structural parameters.

        Refine spline using the result of previous fit and the global structural parameters.
        During refinement, Y-projection of XZ cross section of cylinder is rotated with the
        twist angles, thus is much more precise than the coarse fitting.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to fit.
        max_interval : nm, default 24.0
            Maximum interval of sampling points in nm unit.
        binsize : int, default 1
            Multiscale bin size used for refining.
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top
            90% will be used.
        max_shift: nm, default 2.0
            Maximum shift from the true center of the cylinder. This parameter is used in
            phase cross correlation.
        n_rotations : int, default 3
            Number of rotations to be tested during finding the cylinder center.

        Returns
        -------
        FitResult
            Result of fitting.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.refine, i={i}")
        spl = self.splines[i]
        _required = [H.spacing, H.twist, H.npf]
        if not spl.props.has_glob(_required):
            if (radius := spl.radius) is None:
                radius = self.measure_radius(
                    i=i,
                    binsize=binsize,
                    positions="auto",
                    update=False,
                )
            with spl.props.temp_glob(radius=radius):
                gprops = self.global_cft_params(
                    i=i, binsize=binsize, nsamples=1, update=False
                )
        else:
            gprops = spl.props.glob.select(_required)
        gdict = {k: float(gprops[k][0]) for k in _required}
        ancs = spl.prep_anchor_positions(max_interval=max_interval)

        # Calculate Fourier parameters by cylindrical transformation along spline.
        # Skew angles are divided by the angle of single protofilament and the residual
        # angles are used, considering missing wedge effect.
        space = gdict[H.spacing]
        twist = gdict[H.twist]
        npf = roundint(gdict[H.npf])

        LOGGER.info(f" >> Parameters: spacing = {space:.2f} nm, twist = {twist:.3f} deg, PF = {npf}")  # fmt: skip

        # complement twisting
        pf_ang = 360 / npf
        twists = _get_twists(spl.length(), ancs.size, space, twist, npf)
        scale = self.scale * binsize
        loader = _prep_loader_for_refine(self, spl, ancs, binsize, twists)
        subtomograms = ip.asarray(loader.asnumpy(), axes="pzyx")
        subtomograms[:] -= subtomograms.mean()  # normalize
        subtomograms.set_scale(zyx=scale)

        degrees = np.linspace(-pf_ang / 2, pf_ang / 2, n_rotations) + 180
        max_shift_px = max_shift / scale
        with set_gpu():
            inputs = subtomograms.mean(axis="y")[ip.slicer.x[::-1]]

            # Align twist-corrected images
            shifts_loc = _multi_rotated_auto_zncc(inputs, degrees, max_shift_px)
            tasks = [
                _delayed_translate(inputs[_j], shifts_loc[_j])
                for _j in range(ancs.size)
            ]
            imgs_aligned = _filter_by_corr(
                np.stack(compute(*tasks), axis="p"),
                corr_allowed,
            )

            # Make 2D template using coarse aligned images.
            imgcory = imgs_aligned.mean(axis="p")
            shift = rotated_auto_zncc(
                imgcory, degrees=degrees, max_shifts=max_shift_px * 2
            )
            template = imgcory.affine(translation=shift, mode=Mode.constant, cval=0.0)

            # Align twist-corrected images to the template
            quat = loader.molecules.quaternion()
            tasks = [
                _delayed_zncc_maximum(
                    inputs[_j],
                    _mask_missing_wedge(template, self.tilt_model, quat[_j]),
                    max_shift_px,
                    twists[_j],
                )
                for _j in range(ancs.size)
            ]
            shifts = np.stack(compute(*tasks), axis=0)

        # Update spline parameters
        self.splines[i] = spl.shift(ancs, shifts=shifts * scale, err_max=err_max)
        result = FitResult(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @batch_process
    def measure_radius(
        self,
        i: int = None,
        *,
        binsize: int = 1,
        positions: NDArray[np.float32] | Literal["auto", "anchor"] = "auto",
        min_radius: nm = 1.0,
        update: bool = True,
    ) -> nm:
        """
        Measure radius using radial profile from the center.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to measure.
        binsize : int, default 1
            Multiscale bin size used for radius calculation.
        positions : array-like or "auto" or "anchor", default "auto"
            Sampling positions (between 0 and 1) to calculate radius. If "anchor"
            is given, anchors of the spline will be used. If "auto" is given,
            three positions along the spline will be used.
        min_radius : nm, default 1.0
            Minimum radius of the cylinder.
        update : bool, default True
            If True, global radius property will be updated.

        Returns
        -------
        float (nm)
            Cylinder radius.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.measure_radius, i={i}")
        spl = self.splines[i]

        if isinstance(positions, str) and positions == "auto":
            nanchor = 3
            pos = 1 / nanchor * np.arange(nanchor) + 0.5 / nanchor
        elif isinstance(positions, str) and positions == "anchor":
            pos = spl.anchors
        else:
            pos = np.asarray(positions, dtype=np.float32)

        input_img = self._get_multiscale_or_original(binsize)

        depth = spl.config.fit_depth
        _scale = input_img.scale.x
        min_radius_px = min_radius / _scale
        max_radius = spl.config.fit_width / 2
        max_radius_px = max_radius / _scale
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        tasks = [
            _get_radial_prof(input_img, spl_trans, anc, (min_radius, max_radius), depth)
            for anc in pos
        ]
        profs: list[NDArray[np.float32]] = compute(*tasks)
        prof = np.stack(profs, axis=0).mean(axis=0)
        imax_sub = find_centroid_peak(prof, *_get_thickness(spl, _scale))
        offset_px = _get_radius_offset(min_radius_px, max_radius_px)
        radius = (imax_sub + offset_px) * _scale
        if update:
            spl.radius = radius
        LOGGER.info(f" >> Radius = {radius:.3f} nm")
        return radius

    @batch_process
    def local_radii(
        self,
        *,
        i: int = None,
        depth: nm = 50.0,
        binsize: int = 1,
        min_radius: nm = 1.0,
        update: bool = True,
        update_glob: bool = True,
    ) -> pl.Series:
        """
        Measure the local radii along the splines.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        depth : nm, default 50.0
            Longitudinal length of subtomograms for calculation.
        binsize : int, default 1
            Multiscale binsize to be used.
        min_radius : nm, default 1.0
            Minimum radius of the cylinder.
        update : bool, default True
            If True, spline properties will be updated.
        update_glob : bool, default True
            If True, global properties will be updated using the mean of the local
            radii.

        Returns
        -------
        pl.Series
            Radii along the spline.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.local_radii, i={i}")
        spl = self.splines[i]

        input_img = self._get_multiscale_or_original(binsize)

        depth = spl.config.fit_depth
        _scale = input_img.scale.x
        thickness = _get_thickness(spl, _scale)
        min_radius_px = min_radius / _scale
        max_radius = spl.config.fit_width / 2
        max_radius_px = max_radius / _scale
        offset_px = _get_radius_offset(min_radius_px, max_radius_px)
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        tasks = []
        for anc in spl_trans.anchors:
            task = _get_radial_prof(
                input_img, spl_trans, anc, (min_radius, max_radius), depth
            )
            tasks.append(task)
        profs: list[NDArray[np.float32]] = compute(*tasks)
        radii = list[float]()
        for prof in profs:
            imax_sub = find_centroid_peak(prof, *thickness)
            radii.append((imax_sub + offset_px) * _scale)

        out = pl.Series(H.radius, radii, dtype=pl.Float32)
        if update:
            spl.props.update_loc([out], depth)
        if update_glob:
            spl.props.update_glob([pl.Series(H.radius, [out.mean()])])
        return out

    @batch_process
    def local_cft_params(
        self,
        *,
        i: int = None,
        depth: nm = 50.0,
        binsize: int = 1,
        radius: nm | Literal["local", "global"] = "global",
        nsamples: int = 8,
        update: bool = True,
        update_glob: bool = False,
    ) -> pl.DataFrame:
        """
        Calculate local lattice parameters from cylindrical Fourier space.

        To determine the peaks upsampled discrete Fourier transformation is used
        for every subtomogram.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        depth : nm, default 50.0
            Length of subtomogram for calculation of local parameters.
        binsize : int, default 1
            Multiscale bin size used for calculation.
        radius : str, default "global"
            If "local", use the local radius for the analysis. If "global", use the
            global radius. If a float, use the given radius.
        nsamples : int, default 8
            Number of cylindrical coordinate samplings for Fourier transformation. Multiple
            samplings are needed because up-sampled discrete Fourier transformation does not
            return exactly the same power spectra with shifted inputs, unlike FFT. Larger
            ``nsamples`` reduces the error but is slower.
        update : bool, default True
            If True, spline properties will be updated.
        update_glob : bool, default False
            If True, global properties will be updated using the mean or mode of the local
            properties.

        Returns
        -------
        polars.DataFrame
            Local properties.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.local_cft_params, i={i}")
        spl = self.splines[i]
        radii = _prepare_radii(spl, radius)
        input_img = self._get_multiscale_or_original(binsize)
        _scale = input_img.scale.x
        tasks: list[Delayed[LatticeParams]] = []
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        _analyze_fn = LatticeAnalyzer(spl.config).estimate_lattice_params_task
        for anc, r0 in zip(spl_trans.anchors, radii, strict=True):
            rmin, rmax = spl.radius_range(r0)
            rc = (rmin + rmax) / 2
            coords = spl_trans.local_cylindrical((rmin, rmax), depth, anc, scale=_scale)
            tasks.append(_analyze_fn(input_img, coords, rc, nsamples=nsamples))

        lprops = pl.DataFrame(
            compute(*tasks),
            schema=LatticeParams.polars_schema(),
        )
        if update:
            spl.props.update_loc(lprops, depth)
        if update_glob:
            gprops = lprops.select(
                pl.col(H.spacing).mean(),
                pl.col(H.pitch).mean(),
                pl.col(H.twist).mean(),
                pl.col(H.skew).mean(),
                pl.col(H.rise).mean(),
                pl.col(H.rise_length).mean(),
                pl.col(H.npf).mode().first(),
                pl.col(H.start).mode().first(),
            )
            spl.props.update_glob(gprops)

        return lprops

    @batch_process
    def local_cft(
        self,
        *,
        i: int = None,
        depth: nm = 50.0,
        pos: int | None = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Calculate non-upsampled local cylindric Fourier transormation along spline.

        This function does not up-sample.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        depth : nm, default 50.0
            Length of subtomogram for calculation of local parameters.
        pos : int, optional
            Only calculate at ``pos``-th anchor if given.
        binsize : int, default 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            FT images stacked along "p" axis.
        """
        spl = self.splines[i]
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")

        input_img = self._get_multiscale_or_original(binsize)
        _scale = input_img.scale.x
        rmin, rmax = spl.radius_range()
        rc = (rmin + rmax) / 2
        out = list[ip.ImgArray]()
        if pos is None:
            anchors = spl.anchors
        else:
            anchors = [spl.anchors[pos]]
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        with set_gpu():
            for anc in anchors:
                coords = spl_trans.local_cylindrical(
                    (rmin, rmax), depth, anc, scale=_scale
                )
                polar = get_polar_image(input_img, coords, rc)
                polar[:] -= np.mean(polar)
                out.append(polar.fft(dims="rya"))

        return np.stack(out, axis="p")

    @batch_process
    def local_cps(
        self,
        *,
        i: int = None,
        depth: nm = 50.0,
        pos: int | None = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Calculate non-upsampled local cylindric power spectra along spline.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        depth : nm, default 50.0
            Length of subtomogram for calculation of local parameters.
        pos : int, optional
            Only calculate at ``pos``-th anchor if given.
        binsize : int, default 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            FT images stacked along "p" axis.
        """
        cft = self.local_cft(i=i, depth=depth, pos=pos, binsize=binsize)
        return cft.real**2 + cft.imag**2

    @batch_process
    def global_cft_params(
        self,
        *,
        i: int = None,
        binsize: int = 1,
        nsamples: int = 8,
        update: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate global lattice parameters.

        This function transforms tomogram using cylindrical coordinate system along
        spline. This function calls ``straighten`` beforehand, so that Fourier space is
        distorted if the cylindrical structure is curved.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default 1
            Multiscale bin size used for calculation.
        nsamples : int, default 8
            Number of cylindrical coordinate samplings for Fourier transformation.
            Multiple samplings are needed because up-sampled discrete Fourier
            transformation does not return exactly the same power spectra with shifted
            inputs, unlike FFT. Larger ``nsamples`` reduces the error but is slower.
        update : bool, default True
            If True, spline properties will be updated.

        Returns
        -------
        pl.DataFrame
            Global properties.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.global_cft_params, i={i}")
        spl = self.splines[i]
        rmin, rmax = spl.radius_range()
        img_st = self.straighten_cylindric(i, radii=(rmin, rmax), binsize=binsize)
        rc = (rmin + rmax) / 2
        analyzer = LatticeAnalyzer(spl.config)
        out = analyzer.estimate_lattice_params_polar(
            img_st, rc, nsamples=nsamples
        ).to_polars()
        if update:
            spl.props.glob = spl.props.glob.with_columns(out)
        return out

    @batch_process
    def global_cft(self, i: int = None, binsize: int = 1) -> ip.ImgArray:
        """
        Calculate global cylindrical fast Fourier tranformation.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            Complex image.
        """
        img_st: ip.ImgArray = self.straighten_cylindric(i, binsize=binsize)
        img_st -= np.mean(img_st)
        return img_st.fft(dims="rya")

    @batch_process
    def infer_polarity(
        self,
        i: int = None,
        *,
        binsize: int = 1,
        depth: nm = 40,
        mask_freq: bool = True,
        update: bool = True,
    ) -> Ori:
        """
        Infer spline polarities using polar 2D image.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default 1
            Multiscale bin size used for calculation.
        depth : nm, default 40.0
            Depth of images used to infer polarities.

        Returns
        -------
        Ori
            Orientation of corresponding splines.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.infer_polarity, i={i}")
        current_scale = self.scale * binsize

        if binsize > 1:
            imgb = self.get_multiscale(binsize)
        else:
            try:
                imgb = self.get_multiscale(1)
            except ValueError:
                imgb = self.image

        spl = self.splines[i]
        cfg = spl.config
        ori_clockwise = Ori(cfg.clockwise)
        ori_counterclockwise = Ori.invert(ori_clockwise, allow_none=False)
        if spl.radius is None:
            r_range = 0.5 * current_scale, cfg.fit_width / 2
        else:
            r_range = spl.radius_range()
        point = 0.5  # the sampling point
        coords = spl.local_cylindrical(r_range, depth, point, scale=current_scale)
        polar = get_polar_image(imgb, coords, spl.radius, order=1)
        if mask_freq:
            polar = LatticeAnalyzer(cfg).mask_spectra(polar)
        img_flat = polar.mean(axis="y")

        if (npf := spl.props.get_glob(H.npf, None)) is None:
            # if the global properties are already calculated, use it
            # otherwise, calculate the number of PFs from the power spectrum
            ft = img_flat.fft(shift=False, dims="ra")
            pw = ft.real**2 + ft.imag**2
            img_pw = np.mean(pw, axis=0)
            npf = np.argmax(img_pw[cfg.npf_range.asslice()]) + cfg.npf_range.min

        pw_peak = img_flat.local_power_spectra(
            key=ip.slicer.a[npf - 1 : npf + 2],
            upsample_factor=20,
            dims="ra",
        ).mean(axis="a")
        r_argmax = np.argmax(pw_peak)
        clkwise = r_argmax - (pw_peak.size + 1) // 2 <= 0
        ori = ori_clockwise if clkwise else ori_counterclockwise

        # logging
        _val = pw_peak[r_argmax]
        pw_non_peak = np.delete(pw_peak, r_argmax)
        _ave, _std = np.mean(pw_non_peak), np.std(pw_non_peak, ddof=1)
        LOGGER.info(f" >> polarity = {ori.name} (peak intensity={_val:.2g} compared to {_ave:.2g} ± {_std:.2g})")  # fmt: skip
        if update:
            spl.orientation = ori
        return ori

    @batch_process
    def straighten(
        self,
        i: int = None,
        *,
        size: nm | tuple[nm, nm] = None,
        range_: tuple[float, float] = (0.0, 1.0),
        chunk_length: nm | None = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Straightening by building curved coordinate system around splines. Currently
        Cartesian coordinate system and cylindrical coordinate system are supported.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to straighten.
        size : float (nm), optional
            Vertical/horizontal box size.
        range_ : tuple[float, float], default (0.0, 1.0)
            Range of spline domain.
        chunk_length : nm, optional
            If spline is longer than this, it will be first split into chunks,
            straightened respectively and all the straightened images are concatenated
            afterward, to avoid loading entire image into memory.
        binsize : int, default 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """
        spl = self.splines[i]
        input_img = self._get_multiscale_or_original(binsize)
        chunk_length = _normalize_chunk_length(input_img, chunk_length)
        return _straighten.straighten(input_img, spl, range_, size)

    @batch_process
    def straighten_cylindric(
        self,
        i: int = None,
        *,
        radii: tuple[nm, nm] | None = None,
        range_: tuple[float, float] = (0.0, 1.0),
        chunk_length: nm | None = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Straightening by building curved coordinate system around splines. Currently
        Cartesian coordinate system and cylindrical coordinate system are supported.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to straighten.
        radii : tuple of float (nm), optional
            Lower/upper limit of radius.
        range_ : tuple[float, float], default (0.0, 1.0)
            Range of spline domain.
        chunk_length : nm, optional
            If spline is longer than this, it will be first split into chunks,
            straightened respectively and all the straightened images are concatenated
            afterward, to avoid loading entire image into memory.
        binsize : int, default 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """
        spl = self.splines[i]
        input_img = self._get_multiscale_or_original(binsize)
        chunk_length = _normalize_chunk_length(input_img, chunk_length)
        return _straighten.straighten_cylindric(input_img, spl, range_, radii)

    @batch_process
    def map_centers(
        self,
        i: int = None,
        *,
        interval: nm = 1.0,
        orientation: Ori | str | None = None,
        rotate_molecules: bool = True,
    ) -> Molecules:
        """
        Mapping molecules along the center of a cylinder.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that mapping will be calculated.
        interval : float (nm), optional
            Interval of molecules.
        rotate_molecules : bool, default True
            If True, twist the molecule orientations according to the spline twist.

        Returns
        -------
        Molecules
            Molecules object with mapped coordinates and angles.
        """
        spl = self.splines[i]
        u = spl.prep_anchor_positions(interval=interval)
        if rotate_molecules:
            spacing = spl.props.get_glob(H.spacing)
            twist = spl.props.get_glob(H.twist) / 2
            rotation = np.deg2rad(spl.distances(u) / spacing * twist)
        else:
            rotation = None
        mole = spl.anchors_to_molecules(u, rotation=rotation)
        if spl._need_rotation(orientation):
            mole = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
        return mole

    def get_cylinder_model(
        self,
        i: int,
        offsets: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> CylinderModel:  # fmt: skip
        """
        Return the cylinder model at the given spline ID.

        Parameters
        ----------
        i : int
            Spline ID from which model will be created.
        offsets : tuple of float, optional
            Offset of the model. See :meth:`map_monomers` for details.

        Returns
        -------
        CylinderModel
            The cylinder model.
        """
        return self.splines[i].cylinder_model(offsets=offsets, **kwargs)

    @batch_process
    def map_monomers(
        self,
        i: int = None,
        *,
        offsets: tuple[nm, float] | None = None,
        orientation: Ori | str | None = None,
        extensions: tuple[int, int] = (0, 0),
        **kwargs,
    ) -> Molecules:
        """
        Map monomers in a regular cylinder shape.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that mapping will be calculated.
        offsets : tuple of float, optional
            The offset of origin of oblique coordinate system to map monomers.
        orientation : Ori or str, optional
            Orientation of the y-axis of each molecule.

        Returns
        -------
        Molecules
            Object that represents monomer positions and angles.
        """
        model = self.get_cylinder_model(i, offsets=offsets, **kwargs)
        ny, na = model.shape
        ext0, ext1 = extensions
        if ny + ext0 + ext1 < 0:
            raise ValueError("The number of monomers is negative.")
        yy, aa = np.indices((ny + ext0 + ext1, na), dtype=np.int32)
        yy -= ext0
        coords = np.stack([yy.ravel(), aa.ravel()], axis=1)
        spl = self.splines[i]
        mole = model.locate_molecules(spl, coords)
        if spl._need_rotation(orientation):
            mole = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
        return mole

    @batch_process
    def map_on_grid(
        self,
        i: int = None,
        coords: NDArray[np.int32] = (),
        *,
        offsets: tuple[nm, float] | None = None,
        orientation: Ori | str | None = None,
        **kwargs,
    ) -> Molecules:
        """
        Map monomers in a regular cylinder shape.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that mapping will be calculated.
        coords : ndarray
            Integer coordinates on the cylinder surface.
        offsets : tuple of float, optional
            The offset of origin of oblique coordinate system to map monomers.
        orientation : Ori or str, optional
            Orientation of the y-axis of each molecule.

        Returns
        -------
        Molecules
            Object that represents monomer positions and angles.
        """
        model = self.get_cylinder_model(i, offsets=offsets, **kwargs)
        coords = np.asarray(coords, dtype=np.int32)
        spl = self.splines[i]
        mole = model.locate_molecules(spl, coords)
        if spl._need_rotation(orientation):
            mole = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
        return mole

    @batch_process
    def map_pf_line(
        self,
        i: int = None,
        *,
        interval: nm = 1.0,
        offsets: tuple[nm, float] = (0.0, 0.0),
        orientation: Ori | str | None = None,
    ) -> Molecules:
        """
        Mapping molecules along a protofilament line.

        This method is useful when you want to visualize seam or protofilament, or
        assign molecules for subtomogram averaging of seam binding protein or doublet
        microtubule.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that mapping will be calculated.
        offsets : (float, float), default (0.0, 0.0)
            Axial offset in nm and angular offset in degree.

        Returns
        -------
        Molecules
            Object that represents protofilament positions and angles.
        """
        spl = self.splines[i]
        spacing = spl.props.get_glob(H.spacing)
        twist = spl.props.get_glob(H.twist) / 2

        ny = roundint(spl.length() / interval)
        skew_rad = np.deg2rad(twist) * interval / spacing

        yoffset, aoffset = offsets
        rcoords = np.full(ny, spl.radius)
        ycoords = np.arange(ny) * interval + yoffset
        acoords = np.arange(ny) * skew_rad + np.deg2rad(aoffset)
        coords = np.stack([rcoords, ycoords, acoords], axis=1)
        mole = spl.cylindrical_to_molecules(coords)
        if spl._need_rotation(orientation):
            mole = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
        return mole

    def _prep_fit_spline(self, spl: CylSpline, anc: NDArray[np.float32], binsize: int):
        npoints = anc.size
        interval = spl.length() / (npoints - 1)
        depth_px = self.nm2pixel(spl.config.fit_depth, binsize=binsize)
        width_px = self.nm2pixel(spl.config.fit_width, binsize=binsize)

        # If subtomogram region is rotated by 45 degree, its XY-width will be
        # (length + width) / sqrt(2)
        if binsize > 1:
            centers = spl.map(anc) - self.multiscale_translation(binsize)
        else:
            centers = spl.map(anc)
        center_px = self.nm2pixel(centers, binsize=binsize)
        size_px = (width_px,) + (roundint((width_px + depth_px) / 1.414),) * 2
        input_img = self._get_multiscale_or_original(binsize)

        subtomograms = crop_tomograms(input_img, center_px, size_px)
        subtomograms[:] -= subtomograms.mean()
        scale = self.scale * binsize
        return subtomograms, interval, scale


def dask_angle_corr(
    imgs, ang_centers, drot: float = 7, nrots: int = 29
) -> NDArray[np.float32]:
    _angle_corr = delayed(partial(angle_corr, drot=drot, nrots=nrots))
    tasks = []
    for img, ang in zip(imgs, ang_centers, strict=True):
        tasks.append(da.from_delayed(_angle_corr(img, ang), shape=(), dtype=np.float32))
    return da.compute(tasks)[0]


def _prepare_radii(
    spl: CylSpline, radius: nm | Literal["local", "global"]
) -> Sequence[float]:
    if isinstance(radius, str):
        if radius == "global":
            if spl.radius is None:
                raise ValueError("Global radius is not measured yet.")
            radii = np.full(spl.anchors.size, spl.radius, dtype=np.float32)
        elif radius == "local":
            if not spl.props.has_loc(H.radius):
                raise ValueError("Local radii is not measured yet.")
            radii = spl.props.loc[H.radius].to_numpy()
        else:
            raise ValueError("`radius` must be 'local' or 'global' if string.")
    else:
        if radius <= 0:
            raise ValueError("`radius` must be a positive float.")
        radii = np.full(spl.anchors.size, radius, dtype=np.float32)
    return radii


def angle_uniform_filter(input, size, mode=Mode.mirror, cval=0):
    """Uniform filter of angles."""
    phase = np.exp(1j * input)
    out = ndi.convolve1d(phase, np.ones(size), mode=mode, cval=cval)
    return np.angle(out)


def _soft_mask_edges(
    subtomograms: ip.ImgArray,
    spl: CylSpline,
    anc: NDArray[np.float32],
    scale: float,
    edge_sigma: float | None,
) -> ip.ImgArray:
    if edge_sigma is None:
        return subtomograms
    # mask XY-region outside the cylinders with sigmoid function.
    yy, xx = np.indices(subtomograms.sizesof("yx"))
    yc, xc = np.array(subtomograms.sizesof("yx")) / 2 - 0.5
    yr = yy - yc
    xr = xx - xc
    for _j, ds in enumerate(spl.map(anc, der=1)):
        _, vy, vx = ds
        distance: NDArray[np.float64] = (
            np.abs(-xr * vy + yr * vx) / np.sqrt(vx**2 + vy**2) * scale
        )
        distance_cutoff = spl.config.fit_width / 2
        if edge_sigma == 0:
            mask_yx = (distance > distance_cutoff).astype(np.float32)
        else:
            mask_yx = 1 / (1 + np.exp((distance - distance_cutoff) / edge_sigma))
        mask = np.stack([mask_yx] * subtomograms.shape.z, axis=0)
        subtomograms[_j] *= mask
    return subtomograms


def _shift_coords(
    coords_px: NDArray[np.float32],
    shifts: NDArray[np.float32],
    refined_tilt_rad: NDArray[np.float32],
) -> NDArray[np.float32]:
    shifts_3d = np.stack(
        [shifts[:, 0], np.zeros(shifts.shape[0]), shifts[:, 1]], axis=1
    )
    rotvec = np.zeros(shifts_3d.shape, dtype=np.float32)
    rotvec[:, 0] = -refined_tilt_rad
    rot = Rotation.from_rotvec(rotvec)
    return coords_px + rot.apply(shifts_3d)


def _prep_loader_for_refine(
    self: CylTomogram,
    spl: CylSpline,
    ancs: NDArray[np.float32],
    binsize: int,
    twists: NDArray[np.float32] | None = None,
) -> SubtomogramLoader:
    input_img = self._get_multiscale_or_original(binsize)

    depth_px = self.nm2pixel(spl.config.fit_depth, binsize=binsize)
    width_px = self.nm2pixel(spl.config.fit_width, binsize=binsize)

    if twists is None:
        rotation = None
    else:
        rotation = -np.deg2rad(twists)
    mole = spl.anchors_to_molecules(ancs, rotation=rotation)
    if binsize > 1:
        mole = mole.translate(-self.multiscale_translation(binsize))
    scale = input_img.scale.x

    # Load subtomograms rotated by twisting. All the subtomograms should look similar.
    arr = input_img.value
    loader = SubtomogramLoader(
        arr,
        mole,
        order=1,
        scale=scale,
        output_shape=(width_px, depth_px, width_px),
        corner_safe=True,
    )
    return loader


def _get_twists(
    length: float,
    nancs: int,
    space: float,
    twist: float,
    npf: int,
):
    twist_interv = length / (nancs - 1)
    twists = np.arange(nancs) * twist_interv / space * twist
    pf_ang = 360 / npf
    twists %= pf_ang
    twists[twists > pf_ang / 2] -= pf_ang
    return twists


@delayed
def _delayed_translate(img: ip.ImgArray, shift) -> ip.ImgArray:
    return img.affine(translation=shift, mode=Mode.constant, cval=0)


@delayed
def _delayed_zncc_maximum(
    img: ip.ImgArray,
    tmp: ip.ImgArray,
    max_shifts: int,
    twist: float,
):
    shift = -ip.zncc_maximum(tmp, img, max_shifts=max_shifts)
    rad = np.deg2rad(twist)
    cos, sin = np.cos(rad), np.sin(rad)
    zxrot = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
    return shift @ zxrot


_FLIP_ZX = ip.slicer.z[::-1].x[::-1]


def _filter_by_corr(imgs_aligned: ip.ImgArray, corr_allowed: float) -> ip.ImgArray:
    if corr_allowed >= 1:
        return imgs_aligned
    corrs = np.asarray(ip.zncc(imgs_aligned, imgs_aligned[_FLIP_ZX]))
    threshold = np.quantile(corrs, 1 - corr_allowed)
    indices: np.ndarray = np.where(corrs >= threshold)[0]
    imgs_aligned = imgs_aligned[indices.tolist()]
    LOGGER.info(f" >> Correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
    return imgs_aligned


def _mask_missing_wedge(
    img: ip.ImgArray,
    tilt_model: TiltSeriesModel,
    quat: NDArray[np.float32],
) -> ip.ImgArray:
    """Mask the missing wedge of the image and return the real image."""
    shape = (img.shape[0], 1, img.shape[1])
    # central slice theorem
    mask3d = tilt_model.create_mask(Rotation(quat), shape)
    mask = mask3d[:, 0, :]
    return ip.asarray(ifft2(fft2(img.value) * mask).real, like=img)


def _get_thickness(spl: CylSpline, scale: nm) -> tuple[nm, nm]:
    thick_inner_px = spl.config.thickness_inner / scale
    thick_outer_px = spl.config.thickness_outer / scale
    return thick_inner_px, thick_outer_px


def _get_radial_prof(
    input_img: ip.ImgArray | ip.LazyImgArray,
    spl: CylSpline,
    anc: float,
    r_range: tuple[nm, nm],
    depth: nm,
) -> da.Array[np.float32]:
    coords = spl.local_cylindrical(r_range, depth, anc, scale=input_img.scale.x)
    task = map_coordinates_task(input_img, coords)
    prof = da.mean(
        da.from_delayed(task, shape=coords.shape[1:], dtype=np.float32), axis=(1, 2)
    )
    return prof


def _get_radius_offset(min_radius_px, max_radius_px) -> nm:
    n_radius = roundint(max_radius_px - min_radius_px)
    return (min_radius_px + max_radius_px - n_radius + 1) / 2


@delayed
def _lazy_rotated_auto_zncc(img, degrees, max_shifts):
    return rotated_auto_zncc(img, degrees, max_shifts=max_shifts)


def _multi_rotated_auto_zncc(
    imgs: ip.ImgArray, degrees: NDArray[np.float32], max_shift_px: int
) -> NDArray[np.float32]:
    tasks = [_lazy_rotated_auto_zncc(subimg, degrees, max_shift_px) for subimg in imgs]
    return np.stack(compute(*tasks), axis=0)


def _normalize_chunk_length(img, chunk_length: nm | None) -> nm:
    if chunk_length is None:
        if isinstance(img, ip.LazyImgArray):
            chunk_length = 72.0
        else:
            chunk_length = 999999
    return chunk_length
