from __future__ import annotations
import logging
from typing_extensions import ParamSpec, Concatenate

from typing import Callable, Any, TypeVar, overload, Protocol, TYPE_CHECKING, NamedTuple
from collections.abc import Iterable
from functools import partial, wraps
import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
from scipy import ndimage as ndi
from scipy.fft import fft2, ifft2
from scipy.spatial.transform import Rotation
from dask import array as da, delayed

from acryo import Molecules, SubtomogramLoader
from acryo.alignment import ZNCCAlignment
import impy as ip

from .cyl_spline import CylSpline
from .tomogram import Tomogram
from cylindra.const import (
    nm,
    PropertyNames as H,
    Ori,
    Mode,
    GlobalVariables as GVar,
    IDName,
)
from cylindra.utils import (
    crop_tomogram,
    centroid,
    map_coordinates,
    rotated_auto_zncc,
    roundint,
    ceilint,
    set_gpu,
    mirror_zncc,
    angle_corr,
)

if TYPE_CHECKING:
    from typing_extensions import Self, Literal
    from .cylindric import CylinderModel

    Degenerative = Callable[[ArrayLike], Any]


LOCALPROPS = [
    H.splPosition,
    H.splDistance,
    H.riseAngle,
    H.yPitch,
    H.skewAngle,
    H.nPF,
    H.start,
]

LOGGER = logging.getLogger("cylindra")


def tandg(x):
    """Tangent in degree."""
    return np.tan(np.deg2rad(x))


def rmsd(shifts: ArrayLike) -> float:
    shifts = np.atleast_2d(shifts)
    return np.sqrt(np.sum(shifts**2) / shifts.shape[0])


_P = ParamSpec("_P")
_RETURN = TypeVar("_RETURN")


class BatchCallable(Protocol[_P, _RETURN]):
    """
    This protocol enables static type checking of methods decorated with ``@batch_process``.
    The parameter specifier ``_KW`` does not add any information but currently there is not
    quick solution.
    """

    @overload
    def __call__(
        self, i: Literal[None], *args: _P.args, **kwargs: _P.kwargs
    ) -> list[_RETURN]:
        ...

    @overload
    def __call__(self, i: int, *args: _P.args, **kwargs: _P.kwargs) -> _RETURN:
        ...

    @overload
    def __call__(
        self, i: Iterable[int] | None, *args: _P.args, **kwargs: _P.kwargs
    ) -> list[_RETURN]:
        ...

    def __call__(self, i, *args, **kwargs):
        ...


def batch_process(
    func: Callable[Concatenate[CylTomogram, Any, _P], _RETURN]
) -> BatchCallable[_P, _RETURN]:
    """Enable running function for every splines."""

    @wraps(func)
    def _func(self: CylTomogram, i=None, **kwargs):
        if isinstance(i, int):
            out = func(self, i=i, **kwargs)
            return out

        # Determine along which spline function will be executed.
        if i is None:
            i_list = range(self.n_splines)
        elif not isinstance(i, Iterable):
            raise TypeError("'i' must be int or iterable of int if specified")
        else:
            i_list = []
            for i_ in i:
                if -self.n_splines <= i_ < 0:
                    i_list.append(i_ + self.n_splines)
                elif 0 <= i_ < self.n_splines:
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


class FitResult(NamedTuple):
    residual: NDArray[np.float64]
    rmsd: float

    @classmethod
    def from_residual(cls, residual: ArrayLike) -> FitResult:
        return cls(residual=residual, rmsd=rmsd(residual))


class CylTomogram(Tomogram):
    """Tomogram with cylindrical splines."""

    def __init__(self):
        super().__init__()
        self._splines: list[CylSpline] = []

    @property
    def splines(self) -> list[CylSpline]:
        """List of splines."""
        return self._splines

    @property
    def n_splines(self) -> int:
        """Number of spline paths."""
        return len(self._splines)

    def export_localprops(self, file_path: str, **kwargs):
        """
        Export local properties as a csv file.

        Parameters
        ----------
        file_path : str
            File path to export.
        """
        df = self.collect_localprops()
        df.write_csv(file_path, **kwargs)
        return None

    def clear_cache(self, loc: bool = True, glob: bool = True) -> None:
        """Clear caches of registered splines."""
        for spl in self.splines:
            spl.clear_cache(loc, glob)
        return None

    def add_spline(self, coords: ArrayLike) -> None:
        """
        Add spline path to tomogram.

        Parameters
        ----------
        coords : array-like
            (N, 3) array of coordinates. A spline curve that fit it well is added.
        """
        spl = CylSpline(degree=GVar.splOrder)
        coords = np.asarray(coords)
        spl.fit_coa(coords, min_radius=GVar.minCurvatureRadius)
        interval: nm = 30.0
        length = spl.length()

        n = int(length / interval) + 1
        fit = spl(np.linspace(0, 1, n))
        if coords.shape[0] <= GVar.splOrder and coords.shape[0] < fit.shape[0]:
            return self.add_spline(fit)

        self._splines.append(spl)
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
        self._splines[i].make_anchors(interval=interval, n=n, max_interval=max_interval)
        return None

    def align_to_polarity(self, orientation: Ori | str = Ori.MinusToPlus) -> Self:
        """
        Align all the splines in the direction parallel to the given polarity.

        Parameters
        ----------
        orientation : Ori or str, default is Ori.MinusToPlus
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
        edge_sigma: nm = 2.0,
        max_shift: nm = 5.0,
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
        max_interval : nm, default is 30.0
            Maximum interval of sampling points in nm unit.
        degree_precision : float, default is 0.5
            Precision of xy-tilt degree in angular correlation.
        binsize : int, default is 1
            Multiscale bin size used for fitting.
        edge_sigma : nm, default is 2.0
            Sharpness of mask at the edges. If not None, fitting will be executed after regions
            outside the cylinder are masked. Soft mask is important for precision because sharp
            changes in intensity cause strong correlation at the edges.
        max_shift: nm, default is 5.0
            Maximum shift from the true center of the cylinder. This parameter is used in phase
            cross correlation.

        Returns
        -------
        FitResult
            Result of fitting.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.fit, i={i}")
        spl = self._splines[i]
        spl.make_anchors(max_interval=max_interval)
        npoints = spl.anchors.size
        interval = spl.length() / (npoints - 1)
        spl = self._splines[i]
        length_px = self.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = self.nm2pixel(GVar.fitWidth, binsize=binsize)

        # If subtomogram region is rotated by 45 degree, its XY-width will be
        # (length + width) / sqrt(2)
        if binsize > 1:
            centers = spl.map() - self.multiscale_translation(binsize)
        else:
            centers = spl.map()
        center_px = self.nm2pixel(centers, binsize=binsize)
        size_px = (width_px,) + (roundint((width_px + length_px) / 1.41),) * 2
        input_img = self._get_multiscale_or_original(binsize)

        subtomograms: ip.ImgArray = np.stack(
            [crop_tomogram(input_img, c, size_px) for c in center_px], axis="p"
        )

        subtomograms[:] -= subtomograms.mean()
        scale = self.scale * binsize

        with set_gpu():
            if edge_sigma is not None:
                # mask XY-region outside the cylinders with sigmoid function.
                yy, xx = np.indices(subtomograms.sizesof("yx"))
                yc, xc = np.array(subtomograms.sizesof("yx")) / 2 - 0.5
                yr = yy - yc
                xr = xx - xc
                for i, ds in enumerate(spl(der=1)):
                    _, vy, vx = ds
                    distance: NDArray[np.float64] = (
                        np.abs(-xr * vy + yr * vx) / np.sqrt(vx**2 + vy**2) * scale
                    )
                    distance_cutoff = GVar.fitWidth / 2
                    if edge_sigma == 0:
                        mask_yx = (distance > distance_cutoff).astype(np.float32)
                    else:
                        mask_yx = 1 / (
                            1 + np.exp((distance - distance_cutoff) / edge_sigma)
                        )
                    mask = np.stack([mask_yx] * subtomograms.shape.z, axis=0)
                    subtomograms[i] *= mask

            ds = spl.map(der=1)
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

            # Rotate subtomograms
            for i, img in enumerate(subtomograms):
                img: ip.ImgArray
                angle = refined_tilt_deg[i]
                img.rotate(-angle, cval=0, update=True)

            # zx-shift correction by self-PCC
            subtomo_proj = subtomograms.proj("y")

            if edge_sigma is not None:
                # Regions outside the mask don't need to be considered.
                xc = int(subtomo_proj.shape.x / 2)
                w = int(GVar.fitWidth / scale / 2)
                subtomo_proj = subtomo_proj[ip.slicer.x[xc - w : xc + w + 1]]

            shifts = np.zeros((npoints, 2))  # zx-shift
            max_shift_px = max_shift / scale * 2
            for i in range(npoints):
                img = subtomo_proj[i]
                shifts[i] = mirror_zncc(img, max_shifts=max_shift_px) / 2

        # Update spline coordinates.
        # Because centers of subtomogram are on lattice points of pixel coordinate,
        # coordinates that will be shifted should be converted to integers.
        coords_px = self.nm2pixel(spl(), binsize=binsize).astype(np.float32)

        shifts_3d = np.stack(
            [shifts[:, 0], np.zeros(shifts.shape[0]), shifts[:, 1]], axis=1
        )
        rotvec = np.zeros(shifts_3d.shape, dtype=np.float32)
        rotvec[:, 0] = -refined_tilt_rad
        rot = Rotation.from_rotvec(rotvec)
        coords_px += rot.apply(shifts_3d)

        coords = coords_px * scale + self.multiscale_translation(binsize)

        # Update spline parameters
        min_cr = GVar.minCurvatureRadius
        spl.fit_coa(coords, min_radius=min_cr, weight_ramp=(min_cr / 10, 0.5))
        result = FitResult.from_residual(residual=shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @batch_process
    def refine(
        self,
        i: int = None,
        *,
        max_interval: nm = 30.0,
        binsize: int = 1,
        corr_allowed: float = 0.9,
        max_shift: nm = 2.0,
        n_rotation: int = 7,
    ) -> FitResult:
        """
        Spline refinement using global lattice structural parameters.

        Refine spline using the result of previous fit and the global structural parameters.
        During refinement, Y-projection of XZ cross section of cylinder is rotated with the
        skew angle, thus is much more precise than the coarse fitting.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to fit.
        max_interval : nm, default is 24.0
            Maximum interval of sampling points in nm unit.
        binsize : int, default is 1
            Multiscale bin size used for refining.
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top
            90% will be used.
        max_shift: nm, default is 2.0
            Maximum shift from the true center of the cylinder. This parameter is used in
            phase cross correlation.
        n_rotation : int, default is 7
            Number of rotations to be tested during finding the cylinder center.

        Returns
        -------
        FitResult
            Result of fitting.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.refine, i={i}")
        spl = self.splines[i]
        if spl.radius is None:
            spl.make_anchors(n=3)
            self.set_radius(i=i, binsize=binsize)

        level = LOGGER.level
        LOGGER.setLevel(logging.WARNING)
        try:
            props = self.splines[i].globalprops
            if props is None:
                props = self.global_ft_params(i, binsize=binsize)
        finally:
            LOGGER.setLevel(level)
        spl.make_anchors(max_interval=max_interval)
        npoints = spl.anchors.size
        interval = spl.length() / (npoints - 1)

        # Calculate Fourier parameters by cylindrical transformation along spline.
        # Skew angles are divided by the angle of single protofilament and the residual
        # angles are used, considering missing wedge effect.
        lp = props[H.yPitch][0] * 2
        skew = props[H.skewAngle][0]
        npf = roundint(props[H.nPF][0])

        LOGGER.info(
            f" >> Parameters: spacing = {lp/2:.2f} nm, skew = {skew:.3f} deg, PF = {npf}"
        )

        # complement skewing
        skew_angles = np.arange(npoints) * interval / lp * skew
        pf_ang = 360 / npf
        skew_angles %= pf_ang
        skew_angles[skew_angles > pf_ang / 2] -= pf_ang

        input_img = self._get_multiscale_or_original(binsize)

        length_px = self.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = self.nm2pixel(GVar.fitWidth, binsize=binsize)

        mole = spl.anchors_to_molecules(rotation=-np.deg2rad(skew_angles))
        if binsize > 1:
            mole = mole.translate(-self.multiscale_translation(binsize))
        scale = input_img.scale.x

        # Load subtomograms rotated by skew angles. All the subtomograms should look similar.
        arr = input_img.value
        loader = SubtomogramLoader(
            arr,
            mole,
            order=1,
            scale=scale,
            output_shape=(width_px, length_px, width_px),
            corner_safe=True,
        )
        subtomograms = ip.asarray(loader.asnumpy(), axes="pzyx")
        bg = subtomograms.mean()
        subtomograms[:] -= bg  # normalize
        subtomograms.set_scale(input_img)

        with set_gpu():
            inputs = subtomograms.proj("y")[ip.slicer.x[::-1]]

            # Coarsely align skew-corrected images
            imgs_aligned = ip.empty(inputs.shape, dtype=np.float32, axes=inputs.axes)
            max_shift_px = max_shift / scale

            for i in range(npoints):
                img = inputs[i]
                shift = mirror_zncc(img, max_shifts=max_shift_px * 2) / 2
                imgs_aligned.value[i] = img.affine(
                    translation=shift, mode=Mode.constant, cval=0
                )

            if corr_allowed < 1:
                # remove low correlation image from calculation of template image.
                corrs = np.asarray(
                    ip.zncc(imgs_aligned, imgs_aligned[ip.slicer.z[::-1].x[::-1]])
                )
                threshold = np.quantile(corrs, 1 - corr_allowed)
                indices: np.ndarray = np.where(corrs >= threshold)[0]
                imgs_aligned = imgs_aligned[indices.tolist()]
                LOGGER.info(
                    f" >> Correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}"
                )

            # Make template using coarse aligned images.
            imgcory = imgs_aligned.proj("p")
            degrees = np.linspace(-pf_ang / 2, pf_ang / 2, n_rotation) + 180
            shift = rotated_auto_zncc(
                imgcory, degrees=degrees, max_shifts=max_shift_px * 2
            )
            template = imgcory.affine(translation=shift, mode=Mode.constant, cval=0.0)
            zncc = ZNCCAlignment(subtomograms[0].value, tilt_range=self.tilt_range)
            # Align skew-corrected images to the template
            shifts = np.zeros((npoints, 2))
            quat = mole.quaternion()
            for i in range(npoints):
                img = inputs[i]
                tmp = _mask_missing_wedge(template, zncc, quat[i])
                shift = -ip.zncc_maximum(tmp, img, max_shifts=max_shift_px)

                rad = np.deg2rad(skew_angles[i])
                cos, sin = np.cos(rad), np.sin(rad)
                zxrot = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
                shifts[i] = shift @ zxrot

        # Update spline parameters
        min_cr = GVar.minCurvatureRadius
        spl.shift_coa(
            shifts=shifts * scale, min_radius=min_cr, weight_ramp=(min_cr / 10, 0.5)
        )
        result = FitResult.from_residual(shifts * scale)
        LOGGER.info(f" >> Shift RMSD = {result.rmsd:.3f} nm")
        return result

    @batch_process
    def set_radius(
        self,
        i: int = None,
        *,
        radius: nm | None = None,
        binsize: int = 1,
    ) -> nm:
        """
        Set radius or measure radius using radial profile from the center.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to measure.
        binsize : int, default is 1
            Multiscale bin size used for radius calculation.

        Returns
        -------
        float (nm)
            Cylinder radius.
        """
        spl = self.splines[i]

        if radius is not None:
            spl.radius = float(radius)
            return spl.radius

        if spl._anchors is None:
            spl.make_anchors(n=3)

        if binsize > 1:
            input_img = self.get_multiscale(binsize)
        else:
            try:
                input_img = self.get_multiscale(1)
            except ValueError:
                input_img = self.image

        length_px = self.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = self.nm2pixel(GVar.fitWidth, binsize=binsize)
        scale = self.scale * binsize

        mole = spl.anchors_to_molecules()
        if binsize > 1:
            mole = mole.translate(-self.multiscale_translation(binsize))

        arr = input_img.value
        loader = SubtomogramLoader(
            arr,
            mole,
            order=1,
            scale=scale,
            output_shape=(width_px, length_px, width_px),
            corner_safe=True,
        )
        subtomograms = ip.asarray(loader.asnumpy(), axes="pzyx")
        subtomograms[:] -= subtomograms.mean()  # normalize
        subtomograms.set_scale(input_img)

        r_max = GVar.fitWidth / 2
        nbin = roundint(r_max / scale / 2)
        img2d = subtomograms.proj("py")
        prof = img2d.radial_profile(nbin=nbin, r_max=r_max)
        imax = np.nanargmax(prof)
        imax_sub = centroid(prof, imax - 5, imax + 5)

        # prof[0] is radial profile at r=0.5 (not r=0.0)
        r_peak_sub = (imax_sub + 0.5) / nbin * r_max
        spl.radius = r_peak_sub
        return r_peak_sub

    @batch_process
    def local_ft_params(
        self,
        *,
        i: int = None,
        ft_size: nm = 32.0,
        binsize: int = 1,
    ) -> pl.DataFrame:
        """
        Calculate local structural parameters from cylindrical Fourier space.

        To determine the peaks upsampled discrete Fourier transformation is used
        for every subtomogram.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        ft_size : nm, default is 32.0
            Length of subtomogram for calculation of local parameters.
        binsize : int, default is 1
            Multiscale bin size used for calculation.

        Returns
        -------
        polars.DataFrame
            Local properties.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.local_ft_params, i={i}")
        spl = self.splines[i]

        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")

        ylen = self.nm2pixel(ft_size)
        input_img = self._get_multiscale_or_original(binsize)
        _scale = input_img.scale.x
        rmin = spl.radius * GVar.inner / _scale
        rmax = spl.radius * GVar.outer / _scale
        tasks = []
        LOGGER.info(f" >> Rmin = {rmin * _scale:.2f} nm, Rmax = {rmax * _scale:.2f} nm")
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        for anc in spl_trans.anchors:
            coords = spl_trans.local_cylindrical((rmin, rmax), ylen, anc, scale=_scale)
            tasks.append(
                da.from_delayed(
                    lazy_ft_params(input_img, coords, spl.radius),
                    shape=(5,),
                    meta=np.array([], dtype=np.float32),
                )
            )
        with set_gpu():
            results = np.stack(
                da.compute(tasks, scheduler=ip.Const["SCHEDULER"])[0], axis=0
            )

        spl.localprops = pl.DataFrame(
            {
                H.splPosition: spl.anchors,
                H.splDistance: spl.distances(),
                H.riseAngle: results[:, 0],
                H.yPitch: results[:, 1],
                H.skewAngle: results[:, 2],
                H.nPF: np.round(results[:, 3]).astype(np.uint8),
                H.start: results[:, 4],
            }
        )

        return spl.localprops

    @batch_process
    def local_cft(
        self,
        *,
        i: int = None,
        ft_size: nm = 32.0,
        pos: int | None = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Calculate non-upsampled local cylindric Fourier transormation along spline.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        ft_size : nm, default is 32.0
            Length of subtomogram for calculation of local parameters.
        pos : int, optional
            Only calculate at ``pos``-th anchor if given.
        binsize : int, default is 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            FT images stacked along "p" axis.
        """
        spl = self.splines[i]
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")

        ylen = self.nm2pixel(ft_size, binsize=binsize)
        input_img = self._get_multiscale_or_original(binsize)
        _scale = input_img.scale.x
        rmin = spl.radius * GVar.inner / _scale
        rmax = spl.radius * GVar.outer / _scale
        out: list[ip.ImgArray] = []
        if pos is None:
            anchors = spl.anchors
        else:
            anchors = [spl.anchors[pos]]
        spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
        with set_gpu():
            for anc in anchors:
                coords = spl_trans.local_cylindrical(
                    (rmin, rmax), ylen, anc, scale=_scale
                )
                polar = map_coordinates(
                    input_img, coords, order=3, mode=Mode.constant, cval=np.mean
                )
                # "rya" = radius, y, angle
                polar = ip.asarray(polar, axes="rya", dtype=np.float32)
                polar.set_scale(r=_scale, y=_scale, a=_scale)
                polar.scale_unit = self.image.scale_unit
                polar[:] -= np.mean(polar)
                out.append(polar.fft(dims="rya"))

        return np.stack(out, axis="p")

    @batch_process
    def local_cps(
        self,
        *,
        i: int = None,
        ft_size: nm = 32.0,
        pos: int | None = None,
        binsize: int = 1,
    ) -> ip.ImgArray:
        """
        Calculate non-upsampled local cylindric power spectra along spline.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        ft_size : nm, default is 32.0
            Length of subtomogram for calculation of local parameters.
        pos : int, optional
            Only calculate at ``pos``-th anchor if given.
        binsize : int, default is 1
            Multiscale bin size used for calculation.

        Returns
        -------
        ip.ImgArray
            FT images stacked along "p" axis.
        """
        cft = self.local_cft(i=i, ft_size=ft_size, pos=pos, binsize=binsize)
        return cft.real**2 + cft.imag**2

    @batch_process
    def global_ft_params(
        self,
        *,
        i: int = None,
        binsize: int = 1,
    ) -> pl.DataFrame:
        """
        Calculate global structural parameters.

        This function transforms tomogram using cylindrical coordinate system along
        spline. This function calls ``straighten`` beforehand, so that Fourier space is
        distorted if the cylindrical structure is curved.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default is 1
            Multiscale bin size used for calculation.

        Returns
        -------
        pl.DataFrame
            Global properties.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.global_ft_params, i={i}")
        spl = self._splines[i]
        img_st = self.straighten_cylindric(i, binsize=binsize)
        df = _local_dft_params_pl(img_st, spl.radius)
        spl.globalprops = df
        return df

    @batch_process
    def global_cft(self, i: int = None, binsize: int = 1) -> ip.ImgArray:
        """
        Calculate global cylindrical fast Fourier tranformation.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default is 1
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
    def infer_polarity(self, i: int = None, binsize: int = 1, depth: nm = 40) -> Ori:
        """
        Infer spline polarities using polar 2D image.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default is 1
            Multiscale bin size used for calculation.
        depth : nm, default is 40.0
            Depth of images used to infer polarities.

        Returns
        -------
        Ori
            Orientation of corresponding splines.
        """
        current_scale = self.scale * binsize
        imgb = self.get_multiscale(binsize)

        length_px = self.nm2pixel(depth, binsize=binsize)
        width_px = self.nm2pixel(GVar.fitWidth, binsize=binsize)

        spl = self.splines[i]
        ori_clockwise = Ori(GVar.clockwise)
        ori_counterclockwise = Ori.invert(ori_clockwise, allow_none=False)
        if spl.radius is None:
            r_range = 0.5, width_px / 2
        else:
            r_px = self.nm2pixel(spl.radius, binsize=binsize)
            r_range = (GVar.inner * r_px, GVar.outer * r_px)
        point = 0.5
        coords = spl.local_cylindrical(r_range, length_px, point, scale=current_scale)
        mapped = map_coordinates(imgb, coords, order=1, mode=Mode.reflect)
        img_flat = ip.asarray(mapped, axes="rya").proj("y")

        if spl.globalprops is not None:
            # if the global properties are already calculated, use it
            npf = roundint(spl.globalprops[H.nPF][0])
        else:
            # otherwise, calculate the number of PFs from the power spectrum
            ft = img_flat.fft(shift=False, dims="ra")
            pw = ft.real**2 + ft.imag**2
            img_pw = np.mean(pw, axis=0)
            npf = np.argmax(img_pw[GVar.nPFmin : GVar.nPFmax + 1]) + GVar.nPFmin

        pw_peak = img_flat.local_power_spectra(
            key=ip.slicer.a[npf - 1 : npf + 2],
            upsample_factor=20,
            dims="ra",
        ).proj("a", method=np.max)
        r_argmax = np.argmax(pw_peak)
        clkwise = r_argmax - (pw_peak.size + 1) // 2 > 0
        return ori_clockwise if clkwise else ori_counterclockwise

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
        range_ : tuple[float, float], default is (0.0, 1.0)
            Range of spline domain.
        chunk_length : nm, optional
            If spline is longer than this, it will be first split into chunks,
            straightened respectively and all the straightened images are concatenated
            afterward, to avoid loading entire image into memory.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """
        spl = self.splines[i]

        length = self._splines[i].length(nknots=512)

        if chunk_length is None:
            if binsize == 1:
                chunk_length = 72.0
            else:
                chunk_length = 999999

        if length > chunk_length:
            transformed = self._chunked_straighten(
                i,
                length,
                range_,
                function=self.straighten,
                chunk_length=chunk_length,
                size=size,
            )

        else:
            if size is None:
                rz = rx = 1 + 2 * self.nm2pixel(
                    self._splines[i].radius * GVar.outer, binsize=binsize
                )

            else:
                if hasattr(size, "__iter__"):
                    rz, rx = self.nm2pixel(size, binsize=binsize)
                else:
                    rz = rx = self.nm2pixel(size, binsize=binsize)

            input_img = self._get_multiscale_or_original(binsize)
            _scale = input_img.scale.x
            coords = spl.cartesian(shape=(rz, rx), s_range=range_, scale=_scale)
            with set_gpu():
                transformed = map_coordinates(input_img, coords, order=1)

            axes = "zyx"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: _scale for k in axes})
            transformed.scale_unit = "nm"

        return transformed

    @batch_process
    def straighten_cylindric(
        self,
        i: int = None,
        *,
        radii: tuple[nm, nm] = None,
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
        range_ : tuple[float, float], default is (0.0, 1.0)
            Range of spline domain.
        chunk_length : nm, optional
            If spline is longer than this, it will be first split into chunks,
            straightened respectively and all the straightened images are concatenated
            afterward, to avoid loading entire image into memory.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """
        spl = self.splines[i]

        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")
        if chunk_length is None:
            if binsize == 1:
                chunk_length = 72.0
            else:
                chunk_length = 999999
        length = self._splines[i].length(nknots=512)

        if length > chunk_length:
            transformed = self._chunked_straighten(
                i,
                length,
                range_,
                function=self.straighten_cylindric,
                chunk_length=chunk_length,
                radii=radii,
                binsize=binsize,
            )

        else:
            input_img = self._get_multiscale_or_original(binsize)
            _scale = input_img.scale.x
            if radii is None:
                inner_radius = spl.radius * GVar.inner / _scale
                outer_radius = spl.radius * GVar.outer / _scale

            else:
                inner_radius, outer_radius = radii / _scale

            if outer_radius <= inner_radius:
                raise ValueError(
                    "For cylindrical straightening, 'radius' must be (rmin, rmax)"
                )
            spl_trans = spl.translate([-self.multiscale_translation(binsize)] * 3)
            coords = spl_trans.cylindrical(
                r_range=(inner_radius, outer_radius),
                s_range=range_,
                scale=_scale,
            )

            with set_gpu():
                transformed = map_coordinates(input_img, coords, order=3)

            axes = "rya"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: _scale for k in axes})
            transformed.scale_unit = "nm"

        return transformed

    @batch_process
    def map_centers(
        self,
        i: int = None,
        *,
        interval: nm | None = None,
        orientation: Ori | str | None = None,
    ) -> Molecules:
        """
        Mapping molecules along the center of a cylinder.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that mapping will be calculated.
        interval : float (nm), optional
            Interval of molecules.

        Returns
        -------
        Molecules
            Molecules object with mapped coordinates and angles.
        """
        spl = self.splines[i]
        props = self.splines[i].globalprops
        if props is None:
            props = self.global_ft_params(i)

        lp = props[H.yPitch][0] * 2
        skew = props[H.skewAngle][0]

        # Set interval to the dimer length by default.
        if interval is None:
            interval = lp

        # Check length.
        spl_length = spl.length()
        length = spl_length

        npoints = length / interval + 1
        skew_angles = np.arange(npoints) * interval / lp * skew
        u = np.arange(npoints) * interval / length
        mole = spl.anchors_to_molecules(u, rotation=np.deg2rad(skew_angles))
        if _need_rotation(spl, orientation):
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
        spl = self.splines[i]
        if not all(k in kwargs for k in [H.yPitch, H.skewAngle, H.riseAngle, H.nPF]):
            if spl.globalprops is None:
                self.global_ft_params(i)
        return spl.cylinder_model(offsets=offsets, **kwargs)

    @batch_process
    def map_monomers(
        self,
        i: int = None,
        *,
        offsets: tuple[nm, float] = None,
        orientation: Ori | str | None = None,
    ) -> Molecules:
        """
        Map coordinates of monomers in world coordinate.

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
        model = self.get_cylinder_model(i, offsets=offsets)
        spl = self.splines[i]
        mole = model.to_molecules(spl)
        if _need_rotation(spl, orientation):
            mole = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
        return mole

    @batch_process
    def map_pf_line(
        self,
        i: int = None,
        *,
        interval: nm | None = None,
        angle_offset: float = 0.0,
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
        angle_offset : float, default is 0.0
            Angle offset in degree.

        Returns
        -------
        Molecules
            Object that represents protofilament positions and angles.
        """
        spl = self.splines[i]
        props = spl.globalprops
        if props is None:
            props = self.global_ft_params(i)
        lp = props[H.yPitch][0] * 2
        skew = props[H.skewAngle][0]

        if interval is None:
            interval = lp
        ny = roundint(spl.length() / interval)
        skew_rad = np.deg2rad(skew) * interval / lp

        rcoords = np.full(ny, spl.radius)
        ycoords = np.arange(ny) * interval
        acoords = np.arange(ny) * skew_rad + np.deg2rad(angle_offset)
        coords = np.stack([rcoords, ycoords, acoords], axis=1)
        mole = spl.cylindrical_to_molecules(coords)
        if _need_rotation(spl, orientation):
            mole = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
        return mole

    #####################################################################################
    #   Utility functions
    #####################################################################################

    def collect_anchor_coords(self, i: int | Iterable[int] = None) -> NDArray:
        """
        Collect all the anchor coordinates into a single np.ndarray.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        np.ndarray
            Coordinates in shape (N, 3).
        """
        if i is None:
            i = range(self.n_splines)
        elif isinstance(i, int):
            i = [i]
        return np.concatenate([self._splines[i_]() for i_ in i], axis=0)

    def collect_localprops(
        self, i: int | Iterable[int] = None, allow_none: bool = True
    ) -> pl.DataFrame | None:
        """
        Collect all the local properties into a single polars.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        pl.DataFrame
            Concatenated data frame.
        """
        if i is None:
            i = range(self.n_splines)
        elif isinstance(i, int):
            i = [i]
        props: list[pl.DataFrame] = []
        for i_ in i:
            prop = self._splines[i_].localprops
            if prop is None:
                if not allow_none:
                    raise ValueError(f"Local properties of spline {i_} is missing.")
                continue
            props.append(
                prop.with_columns(
                    pl.repeat(i_, pl.count()).cast(pl.UInt16).alias(IDName.spline),
                    pl.arange(0, pl.count()).cast(pl.UInt16).alias(IDName.pos),
                    pl.col(H.nPF).cast(pl.UInt8),
                )
            )

        if len(props) == 0:
            return None

        return pl.concat(props, how="vertical")

    def collect_globalprops(
        self, i: int | Iterable[int] = None, allow_none: bool = True
    ) -> pl.DataFrame:
        """
        Collect all the global properties into a single polars.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        pl.DataFrame
            Concatenated data frame.
        """
        if i is None:
            i = range(self.n_splines)
        elif isinstance(i, int):
            i = [i]
        props: list[pl.DataFrame] = []
        for i_ in i:
            prop = self._splines[i_].globalprops
            if prop is None:
                if not allow_none:
                    raise ValueError(f"Local properties of spline {i_} is missing.")
                continue
            props.append(
                prop.with_columns(
                    pl.Series("radius", [self._splines[i_].radius]),
                    pl.Series("orientation", [str(self._splines[i_].orientation)]),
                    pl.Series(IDName.spline, [i_]),
                    pl.col(H.nPF).cast(pl.UInt8),
                )
            )

        if len(props) == 0:
            return None
        return pl.concat(props, how="vertical")

    def _chunked_straighten(
        self,
        i: int,
        length: nm,
        range_: tuple[float, float],
        function: Callable[..., ip.ImgArray],
        chunk_length: nm = 72.0,
        **kwargs,
    ):
        out = []
        current_distance: nm = 0.0
        start, end = range_
        spl = self.splines[i]
        while current_distance < length:
            start = current_distance / length
            stop = start + chunk_length / length

            # The last segment could be very short
            if spl.length(start=stop, stop=end) / self.scale < 3:
                stop = end

            # Sometimes divmod of floating values generates very small residuals.
            if end - start < 1e-3:
                break

            sub_range = (start, min(stop, end))
            img_st = function(i, range_=sub_range, chunk_length=np.inf, **kwargs)

            out.append(img_st)

            # We have to sum up real distance instead of start/end, to precisely deal
            # with the borders of chunks
            current_distance += img_st.shape.y * self.scale

        # concatenate all the chunks
        transformed = np.concatenate(out, axis="y")
        return transformed


def dask_angle_corr(imgs, ang_centers, drot: float = 7, nrots: int = 29):
    _angle_corr = delayed(partial(angle_corr, drot=drot, nrots=nrots))
    tasks = []
    for img, ang in zip(imgs, ang_centers):
        tasks.append(da.from_delayed(_angle_corr(img, ang), shape=(), dtype=np.float32))
    return da.compute(tasks, scheduler=ip.Const["SCHEDULER"])[0]


def _local_dft_params(img: ip.ImgArray, radius: nm):
    img = img - img.mean()
    l_circ: nm = 2 * np.pi * radius
    npfmin = GVar.nPFmin
    npfmax = GVar.nPFmax

    # First transform around the expected length of y-pitch.
    ylength_nm = img.shape.y * img.scale.y
    y0 = ceilint(ylength_nm / GVar.yPitchMax) - 1
    y1 = max(ceilint(ylength_nm / GVar.yPitchMin), y0 + 1)
    up_a = 20
    up_y = max(int(6000 / img.shape.y), 1)
    npfrange = ceilint(
        npfmax / 2
    )  # The peak of longitudinal periodicity is always in this range.

    power = img.local_power_spectra(
        key=ip.slicer.y[y0:y1].a[-npfrange : npfrange + 1],
        upsample_factor=[1, up_y, up_a],
        dims="rya",
    ).proj("r")

    ymax, amax = np.unravel_index(np.argmax(power), shape=power.shape)
    ymaxp = np.argmax(power.proj("a"))

    amax_f = amax - npfrange * up_a
    ymaxp_f = ymaxp + y0 * up_y
    ymax_f = ymax + y0 * up_y
    a_freq = np.fft.fftfreq(img.shape.a * up_a)
    y_freq = np.fft.fftfreq(img.shape.y * up_y)

    rise = np.arctan(-a_freq[amax_f] / y_freq[ymax_f])
    yspace = 1.0 / y_freq[ymaxp_f] * img.scale.y

    # Second, transform around 13 pf lateral periodicity.
    # This analysis measures skew angle and protofilament number.
    y_factor = abs(radius / yspace / img.shape.a * img.shape.y / 2)
    dy_min = ceilint(tandg(GVar.minSkew) * y_factor * npfmin) - 1
    dy_max = max(ceilint(tandg(GVar.maxSkew) * y_factor * npfmax), dy_min + 1)
    up_a = 20
    up_y = max(int(21600 / (img.shape.y)), 1)

    power = img.local_power_spectra(
        key=ip.slicer.y[dy_min:dy_max].a[npfmin:npfmax],
        upsample_factor=[1, up_y, up_a],
        dims="rya",
    ).proj("r")

    ymax, amax = np.unravel_index(np.argmax(power), shape=power.shape)
    amaxp = np.argmax(power.proj("y"))

    amax_f = amax + npfmin * up_a
    amaxp_f = amaxp + npfmin * up_a
    ymax_f = ymax + dy_min * up_y
    a_freq = np.fft.fftfreq(img.shape.a * up_a)
    y_freq = np.fft.fftfreq(img.shape.y * up_y)

    # When skew angle is positive and y-coordinate increses, a-coordinate will
    # decrese.
    skew = np.arctan(y_freq[ymax_f] / a_freq[amax_f] * 2 * yspace / radius)

    if rise == 0.0:
        start = 0.0
    else:
        start = l_circ / yspace / (np.tan(skew) + 1 / np.tan(rise))

    return np.array(
        [np.rad2deg(rise), yspace, np.rad2deg(skew), amaxp_f / up_a, abs(start)],
        dtype=np.float32,
    )


def _local_dft_params_pl(img: ip.ImgArray, radius: nm) -> pl.DataFrame:
    rise, space, skew, npf, start = _local_dft_params(img, radius)
    df = pl.DataFrame(
        {
            H.riseAngle: pl.Series([rise], dtype=pl.Float32),
            H.yPitch: pl.Series([space], dtype=pl.Float32),
            H.skewAngle: pl.Series([skew], dtype=pl.Float32),
            H.nPF: pl.Series([int(round(npf))], dtype=pl.UInt8),
            H.start: pl.Series([start], dtype=pl.Float32),
        }
    )
    return df


def ft_params(img: ip.ImgArray | ip.LazyImgArray, coords: np.ndarray, radius: nm):
    polar = map_coordinates(img, coords, order=3, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32)  # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
    polar.scale_unit = img.scale_unit
    return _local_dft_params(polar, radius)


lazy_ft_params = delayed(ft_params)


def angle_uniform_filter(input, size, mode=Mode.mirror, cval=0):
    """Uniform filter of angles."""
    phase = np.exp(1j * input)
    out = ndi.convolve1d(phase, np.ones(size), mode=mode, cval=cval)
    return np.angle(out)


def _mask_missing_wedge(
    img: ip.ImgArray,
    zncc: ZNCCAlignment,
    quat: NDArray[np.float32],
) -> ip.ImgArray:
    """Mask the missing wedge of the image and return the real image."""
    if zncc._tilt_range is None:
        return img
    mask3d = zncc._get_missing_wedge_mask(quat)
    # central slice theorem
    mask = mask3d[:, 0, :]
    return ip.asarray(ifft2(fft2(img.value) * mask).real, like=img)


def _need_rotation(spl: CylSpline, orientation: Ori | str | None) -> bool:
    if orientation is not None:
        orientation = Ori(orientation)
        if orientation is Ori.none or spl.orientation is Ori.none:
            raise ValueError(
                "Either molecules' orientation or the input orientation should "
                "not be none."
            )
        if orientation is not spl.orientation:
            return True
    return False
