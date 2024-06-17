from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import impy as ip
import numpy as np
import polars as pl
from acryo import Molecules
from numpy.typing import ArrayLike, NDArray

from cylindra._dask import Delayed, compute
from cylindra.components._ftprops import LatticeAnalyzer, LatticeParams, get_polar_image
from cylindra.components._peak import find_centroid_peak
from cylindra.components.spline import CylSpline
from cylindra.components.tomogram import _misc, _straighten
from cylindra.components.tomogram._spline_list import SplineList
from cylindra.components.tomogram._tomo_base import Tomogram
from cylindra.const import ExtrapolationMode, Mode, Ori, nm
from cylindra.const import PropertyNames as H
from cylindra.utils import (
    ceilint,
    centroid_2d,
    crop_tomograms,
    rotated_auto_zncc,
    roundint,
    set_gpu,
)

if TYPE_CHECKING:
    from typing_extensions import Literal, Self

    from cylindra.components.cylindric import CylinderModel
    from cylindra.components.spline import SplineConfig

LOGGER = logging.getLogger("cylindra")


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

    @_misc.batch_process
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

    @_misc.batch_process
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
    ) -> _misc.FitResult:
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
            subtomograms = _misc.soft_mask_edges(
                subtomograms, spl, anc, scale, edge_sigma
            )
            ds = spl.map(anc, der=1)
            yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
            degree_max = 14.0
            nrots = roundint(degree_max / degree_precision) + 1

            # Angular correlation
            out = _misc.dask_angle_corr(subtomograms, yx_tilt, nrots=nrots)
            refined_tilt_deg = np.array(out)
            refined_tilt_rad = np.deg2rad(refined_tilt_deg)

            # If subtomograms are sampled at short intervals, angles should be smoothened to
            # avoid overfitting.
            size = 2 * roundint(48.0 / interval) + 1
            if size > 1:
                # Mirror-mode padding is "a b c d | c b a".
                refined_tilt_rad = _misc.angle_uniform_filter(
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
            shifts = _misc.multi_rotated_auto_zncc(subtomo_proj, degrees, max_shift_px)

        # Update spline coordinates.
        # Because centers of subtomogram are on lattice points of pixel coordinate,
        # coordinates that will be shifted should be converted to integers.
        coords_px = self.nm2pixel(spl.map(anc), binsize=binsize).astype(np.float32)
        coords_px_new = _misc.shift_coords(coords_px, shifts, refined_tilt_rad)
        coords = coords_px_new * scale + self.multiscale_translation(binsize)

        # Update spline parameters
        self.splines[i] = spl.fit(coords, err_max=err_max)
        result = _misc.FitResult(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @_misc.batch_process
    def fit_centroid(
        self,
        i: int = None,
        *,
        max_interval: nm = 30.0,
        binsize: int = 1,
        err_max: nm = 1.0,
        max_shift: nm = 5.0,
    ) -> _misc.FitResult:
        LOGGER.info(f"Running: {self.__class__.__name__}.fit_centroid, i={i}")
        spl = self.splines[i]
        anc = spl.prep_anchor_positions(max_interval=max_interval)
        scale = self.scale * binsize

        # sample subtomograms
        loader = _misc.prep_loader_for_refine(self, spl, anc, binsize)
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
        result = _misc.FitResult(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @_misc.batch_process
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
    ) -> _misc.FitResult:
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
        twists = _misc.get_twists(spl.length(), ancs.size, space, twist, npf)
        scale = self.scale * binsize
        loader = _misc.prep_loader_for_refine(self, spl, ancs, binsize, twists)
        subtomograms = ip.asarray(loader.asnumpy(), axes="pzyx", dtype=np.float32)
        subtomograms[:] -= subtomograms.mean()  # normalize
        subtomograms.set_scale(zyx=scale)

        degrees = np.linspace(-pf_ang / 2, pf_ang / 2, n_rotations) + 180
        max_shift_px = max_shift / scale
        with set_gpu():
            inputs = subtomograms.mean(axis="y")[ip.slicer.x[::-1]]

            # Align twist-corrected images
            shifts_loc = _misc.multi_rotated_auto_zncc(inputs, degrees, max_shift_px)
            tasks = [
                _misc.delayed_translate(inputs[_j], shifts_loc[_j])
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
                _misc.delayed_zncc_maximum(
                    inputs[_j],
                    _misc.mask_missing_wedge(template, self.tilt_model, quat[_j]),
                    max_shift_px,
                    twists[_j],
                )
                for _j in range(ancs.size)
            ]
            shifts = np.stack(compute(*tasks), axis=0)

        # Update spline parameters
        self.splines[i] = spl.shift(ancs, shifts=shifts * scale, err_max=err_max)
        result = _misc.FitResult(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @_misc.batch_process
    def measure_radius(
        self,
        i: int = None,
        *,
        binsize: int = 1,
        positions: NDArray[np.float32] | Literal["auto", "anchor"] = "auto",
        min_radius: nm = 1.0,
        max_radius: nm = 100.0,
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
        max_radius : nm, default 100.0
            Maximum radius of the cylinder.
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
        max_radius = min(max_radius, spl.config.fit_width / 2)
        max_radius_px = max_radius / _scale
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        tasks = [
            _misc.get_radial_prof(
                input_img, spl_trans, anc, (min_radius, max_radius), depth
            )
            for anc in pos
        ]
        profs: list[NDArray[np.float32]] = compute(*tasks)
        prof = np.stack(profs, axis=0).mean(axis=0)
        imax_sub = find_centroid_peak(prof, *_misc.get_thickness(spl, _scale))
        offset_px = _misc.get_radius_offset(min_radius_px, max_radius_px)
        radius = (imax_sub + offset_px) * _scale
        if update:
            spl.props.update_glob(
                [pl.Series(H.radius, [radius], dtype=pl.Float32)],
                bin_size=binsize,
            )
        LOGGER.info(f" >> Radius = {radius:.3f} nm")
        return radius

    @_misc.batch_process
    def local_radii(
        self,
        *,
        i: int = None,
        depth: nm = 50.0,
        binsize: int = 1,
        min_radius: nm = 1.0,
        max_radius: nm = 100.0,
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
        max_radius : nm, default 100.0
            Maximum radius of the cylinder.
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
        thickness = _misc.get_thickness(spl, _scale)
        min_radius_px = min_radius / _scale
        max_radius = min(max_radius, spl.config.fit_width / 2)
        max_radius_px = max_radius / _scale
        offset_px = _misc.get_radius_offset(min_radius_px, max_radius_px)
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        tasks = []
        for anc in spl_trans.anchors:
            task = _misc.get_radial_prof(
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
            spl.props.update_loc([out], depth, bin_size=binsize)
        if update_glob:
            spl.props.update_glob([pl.Series(H.radius, [out.mean()])], bin_size=binsize)
        return out

    @_misc.batch_process
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
        radii = _misc.prepare_radii(spl, radius)
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
            spl.props.update_loc(lprops, depth, bin_size=binsize)
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
            spl.props.update_glob(gprops, bin_size=binsize)

        return lprops

    def iter_local_image(
        self,
        i: int,
        depth: nm = 50.0,
        pos: int | None = None,
        binsize: int = 1,
    ) -> Iterable[ip.ImgArray]:
        spl = self.splines[i]
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")

        input_img = self._get_multiscale_or_original(binsize)
        _scale = input_img.scale.x
        rmin, rmax = spl.radius_range()
        rc = (rmin + rmax) / 2
        if pos is None:
            anchors = spl.anchors
        else:
            anchors = [spl.anchors[pos]]
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        for anc in anchors:
            coords = spl_trans.local_cylindrical((rmin, rmax), depth, anc, scale=_scale)
            polar = get_polar_image(input_img, coords, rc)
            polar[:] -= np.mean(polar)
            yield polar

    @_misc.batch_process
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
        out = list[ip.ImgArray]()
        with set_gpu():
            for polar in self.iter_local_image(i, depth, pos, binsize):
                out.append(polar.fft(dims="rya"))
        return np.stack(out, axis="p")

    @_misc.batch_process
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

    @_misc.batch_process
    def local_image_with_peaks(
        self,
        *,
        i: int = None,
        binsize: int | None = None,
    ) -> list[_misc.ImageWithPeak]:
        spl = self.splines[i]
        depth = spl.props.window_size[H.twist]
        if binsize is None:
            binsize = spl.props.binsize_loc[H.twist]
        df_loc = spl.props.loc
        out = list[_misc.ImageWithPeak]()
        for j, polar_img in enumerate(self.iter_local_image(i, depth, binsize=binsize)):
            cparams = spl.cylinder_params(
                spacing=_misc.get_component(df_loc, H.spacing, j),
                pitch=_misc.get_component(df_loc, H.pitch, j),
                twist=_misc.get_component(df_loc, H.twist, j),
                skew=_misc.get_component(df_loc, H.skew, j),
                rise_angle=_misc.get_component(df_loc, H.rise, j),
                start=_misc.get_component(df_loc, H.start, j),
                npf=_misc.get_component(df_loc, H.npf, j),
            )
            analyzer = LatticeAnalyzer(spl.config)
            cft = polar_img.fft(dims="rya")
            cps_proj = (cft.real**2 + cft.imag**2).mean(axis="r")
            peakv, peakh = analyzer.params_to_peaks(cps_proj, cparams)
            peakv = peakv.shift_to_center()
            peakh = peakh.shift_to_center()
            out.append(_misc.ImageWithPeak(polar_img, cps_proj, [peakv, peakh]))
        return out

    @_misc.batch_process
    def global_cps(
        self,
        *,
        i: int = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Calculate global cylindrical power spectra.

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
        cft = self.global_cft(i=i, binsize=binsize)
        return cft.real**2 + cft.imag**2

    @_misc.batch_process
    def global_image_with_peaks(
        self,
        *,
        i: int = None,
        binsize: int | None = None,
    ) -> _misc.ImageWithPeak:
        spl = self.splines[i]
        if binsize is None:
            binsize = spl.props.binsize_glob[H.twist]
        img_st = self.straighten_cylindric(i, binsize=binsize)
        img_st -= np.mean(img_st)
        cft = img_st.fft(dims="rya")
        cps = cft.real**2 + cft.imag**2
        cps_proj = cps.mean(axis="r")
        cparams = spl.cylinder_params(
            spacing=spl.props.get_glob(H.spacing, default=None),
            pitch=spl.props.get_glob(H.pitch, default=None),
            twist=spl.props.get_glob(H.twist, default=None),
            skew=spl.props.get_glob(H.skew, default=None),
            rise_angle=spl.props.get_glob(H.rise, default=None),
            start=spl.props.get_glob(H.start, default=None),
            npf=spl.props.get_glob(H.npf, default=None),
        )
        analyzer = LatticeAnalyzer(spl.config)
        peakv, peakh = analyzer.params_to_peaks(cps_proj, cparams)
        peakv = peakv.shift_to_center()
        peakh = peakh.shift_to_center()
        return _misc.ImageWithPeak(img_st, cps_proj, [peakv, peakh])

    @_misc.batch_process
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
        lparams = analyzer.estimate_lattice_params_polar(img_st, rc, nsamples=nsamples)
        out = lparams.to_polars()
        if update:
            spl.props.update_glob(spl.props.glob.with_columns(out), bin_size=binsize)
        return out

    @_misc.batch_process
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
        img_st = self.straighten_cylindric(i, binsize=binsize)
        img_st -= np.mean(img_st)
        return img_st.fft(dims="rya")

    @_misc.batch_process
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

    @_misc.batch_process
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
        chunk_length = _misc.normalize_chunk_length(input_img, chunk_length)
        return _straighten.straighten(input_img, spl, range_, size)

    @_misc.batch_process
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
        chunk_length = _misc.normalize_chunk_length(input_img, chunk_length)
        return _straighten.straighten_cylindric(input_img, spl, range_, radii)

    @_misc.batch_process
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

    @_misc.batch_process
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

    @_misc.batch_process
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

    @_misc.batch_process
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


def _filter_by_corr(imgs_aligned: ip.ImgArray, corr_allowed: float) -> ip.ImgArray:
    if corr_allowed >= 1:
        return imgs_aligned
    _flip_zx = ip.slicer.z[::-1].x[::-1]
    corrs = np.asarray(ip.zncc(imgs_aligned, imgs_aligned[_flip_zx]))
    threshold = np.quantile(corrs, 1 - corr_allowed)
    indices: np.ndarray = np.where(corrs >= threshold)[0]
    imgs_aligned = imgs_aligned[indices.tolist()]
    LOGGER.info(f" >> Correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
    return imgs_aligned
