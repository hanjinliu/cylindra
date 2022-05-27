from __future__ import annotations
import logging
from typing_extensions import ParamSpecKwargs
    
from typing import Callable, Iterable, Any, TypeVar, overload, Protocol, TYPE_CHECKING
from functools import partial, wraps
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.spatial.transform import Rotation
from dask import array as da, delayed

from acryo import Molecules, SubtomogramLoader
import impy as ip

from .spline import Spline
from .tomogram import Tomogram
from ..const import Mole, nm, H, K, Ori, Mode, GVar
from ..utils import (
    crop_tomogram,
    centroid,
    map_coordinates,
    roundint,
    ceilint,
    oblique_meshgrid,
    set_gpu,
    mirror_zncc, 
    angle_uniform_filter,
)

if TYPE_CHECKING:
    from typing_extensions import Self, Literal


LOCALPROPS = [H.splPosition, H.splDistance, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]

LOGGER = logging.getLogger(__name__.split(".")[0])

def tandg(x):
    """Tangent in degree."""
    return np.tan(np.deg2rad(x))

def rmsd(shifts: ArrayLike) -> float:
    shifts = np.atleast_2d(shifts)
    return np.sqrt(np.sum(shifts**2)/shifts.shape[0])

class MtSpline(Spline):
    """
    A spline object with information related to MT.
    """    
    _local_cache = (K.localprops,)
    _global_cache = (K.globalprops, K.radius, K.orientation)
    
    def __init__(self, degree: int = 3, *, lims: tuple[float, float] = (0., 1.)):
        """
        Spline object for MT.
        
        Parameters
        ----------
        k : int, default is 3
            Spline order.
        """        
        super().__init__(degree=degree, lims=lims)
        self.orientation = Ori.none
        self.radius: nm | None = None
        self.localprops: pd.DataFrame | None = None
        self.globalprops: pd.Series | None = None
    
    def invert(self) -> MtSpline:
        """
        Invert the direction of spline. Also invert orientation if exists.

        Returns
        -------
        Spline3D
            Inverted object
        """
        inverted = super().invert()
        inverted.radius = self.radius
        if self.localprops is not None:
            inverted.localprops = self.localprops[::-1]
        inverted.globalprops = self.globalprops
        return inverted
    
    def clip(self, start: float, stop: float) -> MtSpline:
        """
        Clip spline and generate a new one.
        
        This method does not convert spline bases. ``_lims`` is updated instead.
        For instance, if you want to clip spline at 20% to 80% position, call
        ``spl.clip(0.2, 0.8)``. If ``stop < start``, the orientation of spline
        will be inverted, thus the ``orientation`` attribute will also be inverted.

        Parameters
        ----------
        start : float
            New starting position.
        stop : float
            New stopping position.

        Returns
        -------
        MtSpline
            Clipped spline.
        """
        clipped = super().clip(start, stop)
        
        clipped.radius = self.radius
        if start > stop:
            clipped.orientation = Ori.invert(self.orientation)
        else:
            clipped.orientation = self.orientation
        return clipped
    
    def restore(self) -> MtSpline:
        """
        Restore the original, not-clipped spline.

        Returns
        -------
        Spline
            Copy of the original spline.
        """
        original = super().restore()
        start, stop = self._lims
        if start > stop:
            original.orientation = Ori.invert(self.orientation)
        else:
            original.orientation = self.orientation
        return original
    
        
    @property
    def orientation(self) -> Ori:
        """Orientation of spline."""
        return self._orientation
    
    @orientation.setter
    def orientation(self, value: Ori | str | None):
        if value is None:
            self._orientation = Ori.none
        else:
            self._orientation = Ori(value)


_KW = ParamSpecKwargs("_KW")
_RETURN = TypeVar("_RETURN")

class BatchCallable(Protocol[_RETURN]):
    """
    This protocol enables static type checking of methods decorated with ``@batch_process``.
    The parameter specifier ``_KW`` does not add any information but currently there is not 
    quick solution.
    """
    @overload
    def __call__(self, i: Literal[None], **kwargs: _KW) -> list[_RETURN]:
        ...
        
    @overload
    def __call__(self, i: int, **kwargs: _KW) -> _RETURN:
        ...
        
    @overload
    def __call__(self, i: Iterable[int] | None, **kwargs: _KW) -> list[_RETURN]:
        ...

    def __call__(self, i, **kwargs):
        ...


def batch_process(func: Callable[[MtTomogram, Any, _KW], _RETURN]) -> BatchCallable[_RETURN]:
    """Enable running function for every splines."""
    
    @wraps(func)
    def _func(self: MtTomogram, i=None, **kwargs):
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
                errname = type(e).__name__
                msg = str(e)
                raise RuntimeError(f"Exception at spline-{i_}.\n{errname}: {msg}") from e
            else:
                out.append(result)
            
        return out

    return _func


class MtTomogram(Tomogram):
    def __init__(self):
        super().__init__()
        self._splines: list[MtSpline] = []

    @property
    def splines(self) -> list[MtSpline]:
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
        df.to_csv(file_path, **kwargs)
        return None
    
    
    def clear_cache(self, loc: bool = True, glob: bool = True) -> None:
        """Clear caches of registered splines."""
        for spl in self.splines:
            spl.clear_cache(loc, glob)
        return None
    
    def add_spline(self, coords: ArrayLike) -> None:
        """
        Add MtSpline path to tomogram.

        Parameters
        ----------
        coords : array-like
            (N, 3) array of coordinates. A spline curve that fit it well is added.
        """        
        spl = MtSpline(degree=GVar.splOrder)
        coords = np.asarray(coords)
        spl.fit_coa(coords, min_radius=GVar.minCurvatureRadius)
        interval: nm = 30.0
        length = spl.length()
        
        n = int(length/interval) + 1
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
        Make anchors on MtSpline object(s).

        Parameters
        ----------
        interval : nm, optional
            Anchor intervals.
        n : int, optional
            Number of anchors
        max_interval : nm, optional
            Maximum interval between anchors.
        
        """        
        if interval is None and n is None:
            interval = 24.0
        self._splines[i].make_anchors(interval=interval, n=n, max_interval=max_interval)
        return None
    
    
    def align_to_polarity(self, orientation: Ori | str = Ori.MinusToPlus) -> Self:
        """
        Align all the splines in the direction parallel to microtubule polarity.

        Parameters
        ----------
        orientation : Ori or str, default is Ori.MinusToPlus
            To which direction splines will be aligned.

        Returns
        -------
        MtTomogram
            Same object with updated splines.
        """
        orientation = Ori(orientation)
        if orientation == Ori.none:
            raise ValueError("Must be PlusToMinus or MinusToPlus.")
        for i, spl in enumerate(self.splines):
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
    ) -> Self:
        """
        Roughly fit splines to MT.
        
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
            Precision of MT xy-tilt degree in angular correlation.
        binsize : int, default is 1
            Multiscale bin size used for fitting.
        edge_sigma : nm, default is 2.0
            Sharpness of mask at the edges. If not None, fitting will be executed after regions outside 
            microtubule are masked. Soft mask is important for precision because sharp changes in intensity
            cause strong correlation at the edges.
        max_shift: nm, default is 5.0
            Maximum shift from the true center of microtubule. This parameter is used in phase cross 
            correlation.

        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """
        LOGGER.info(f"Running: {self.__class__.__name__}.fit, i={i}")
        spl = self._splines[i]
        spl.make_anchors(max_interval=max_interval)
        npoints = spl.anchors.size
        interval = spl.length()/(npoints-1)
        spl = self._splines[i]
        length_px = self.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = self.nm2pixel(GVar.fitWidth, binsize=binsize)
        
        # If subtomogram region is rotated by 45 degree, its XY-width will be
        # (length + width) / sqrt(2)
        if binsize > 1:
            centers = spl() - self.multiscale_translation(binsize)
        else:
            centers = spl()
        center_px = self.nm2pixel(centers, binsize=binsize)
        size_px = (width_px,) + (roundint((width_px + length_px)/1.41),)*2
        input_img = self._get_multiscale_or_original(binsize)
            
        subtomograms: ip.ImgArray = np.stack(
            [crop_tomogram(input_img, c, size_px) for c in center_px],
            axis="p"
        )
        
        subtomograms[:] -= subtomograms.mean()
        scale = self.scale * binsize

        with set_gpu():        
            if edge_sigma is not None:
                # mask XY-region outside the microtubules with sigmoid function.
                yy, xx = np.indices(subtomograms.sizesof("yx"))
                yc, xc = np.array(subtomograms.sizesof("yx"))/2 - 0.5
                yr = yy - yc
                xr = xx - xc
                for i, ds in enumerate(spl(der=1)):
                    _, vy, vx = ds
                    distance: nm = np.abs(-xr*vy + yr*vx) / np.sqrt(vx**2 + vy**2) * scale
                    distance_cutoff = GVar.fitWidth / 2
                    if edge_sigma == 0:
                        mask_yx = (distance > distance_cutoff).astype(np.float32)
                    else:
                        mask_yx = 1 / (1 + np.exp((distance - distance_cutoff)/edge_sigma))
                    mask = np.stack([mask_yx]*subtomograms.shape.z, axis=0)
                    subtomograms[i] *= mask
            
            ds = spl(der=1)
            yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
            degree_max = 14.0
            nrots = roundint(degree_max/degree_precision) + 1
            
            # Angular correlation
            with set_gpu():
                out = dask_angle_corr(subtomograms, yx_tilt, nrots=nrots)
            refined_tilt_deg = np.array(out)
            refined_tilt_rad = np.deg2rad(refined_tilt_deg)
            
            # If subtomograms are sampled at short intervals, angles should be smoothened to 
            # avoid overfitting.
            size = 2 * roundint(48.0/interval) + 1
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
                subtomo_proj = subtomo_proj[f"x={xc-w}:{xc+w+1}"]
    
            shifts = np.zeros((npoints, 2)) # zx-shift
            max_shift_px = max_shift / scale * 2
            for i in range(npoints):
                img = subtomo_proj[i]
                shifts[i] = mirror_zncc(img, max_shifts=max_shift_px) / 2
        
        # Update spline coordinates.
        # Because centers of subtomogram are on lattice points of pixel coordinate,
        # coordinates that will be shifted should be converted to integers. 
        coords_px = self.nm2pixel(spl(), binsize=binsize).astype(np.float32)
        
        shifts_3d = np.stack([shifts[:, 0],
                              np.zeros(shifts.shape[0]), 
                              shifts[:, 1]], 
                             axis=1)
        rotvec = np.zeros(shifts_3d.shape, dtype=np.float32)
        rotvec[:, 0] = -refined_tilt_rad
        rot = Rotation.from_rotvec(rotvec)
        coords_px += rot.apply(shifts_3d)
            
        coords = coords_px * scale + self.multiscale_translation(binsize)
        
        # Update spline parameters
        spl.fit_coa(coords, min_radius=GVar.minCurvatureRadius)
        LOGGER.info(f" >> Shift RMSD = {rmsd(shifts * scale):.3f} nm")
        return self
    
    @batch_process
    def refine(
        self, 
        i: int = None,
        *, 
        max_interval: nm = 30.0,
        binsize: int = 1,
        corr_allowed: float = 0.9,
        max_shift: nm = 2.0,
    ) -> Self:
        """
        Spline refinement using global lattice structural parameters.
        
        Refine spline using the result of previous fit and the global structural parameters.
        During refinement, Y-projection of MT XZ cross section is rotated with the skew angle,
        thus is much more precise than the coarse fitting.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to fit.
        max_interval : nm, default is 24.0
            Maximum interval of sampling points in nm unit.
        binsize : int, default is 1
            Multiscale bin size used for refining.
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        max_shift: nm, default is 2.0
            Maximum shift from the true center of microtubule. This parameter is used in phase cross 
            correlation.
        
        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
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
        interval = spl.length()/(npoints-1)
            
        # Calculate Fourier parameters by cylindrical transformation along spline.
        # Skew angles are divided by the angle of single protofilament and the residual
        # angles are used, considering missing wedge effect.
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        npf = roundint(props[H.nPF])
        
        LOGGER.info(f" >> Parameters: spacing = {lp/2:.2f} nm, skew = {skew:.3f} deg, PF = {npf}")
        
        # complement skewing
        skew_angles = np.arange(npoints) * interval/lp * skew
        pf_ang = 360/npf
        skew_angles %= pf_ang
        skew_angles[skew_angles > pf_ang/2] -= pf_ang
        
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
            inputs = subtomograms.proj("y")["x=::-1"]
            
            # Coarsely align skew-corrected images                
            imgs_aligned = ip.empty(inputs.shape, dtype=np.float32, axes=inputs.axes)
            max_shift_px = max_shift / scale
            
            for i in range(npoints):
                img = inputs[i]
                shift = mirror_zncc(img, max_shifts=max_shift_px*2) / 2
                imgs_aligned.value[i] = img.affine(translation=shift, mode=Mode.constant, cval=0)
                
            if corr_allowed < 1:
                # remove low correlation image from calculation of template image.
                corrs = np.asarray(ip.zncc(imgs_aligned, imgs_aligned["z=::-1;x=::-1"]))
                threshold = np.quantile(corrs, 1 - corr_allowed)
                indices: np.ndarray = np.where(corrs >= threshold)[0]
                imgs_aligned = imgs_aligned[indices.tolist()]
                LOGGER.info(f" >> Correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
            
            # Make template using coarse aligned images.
            imgcory = imgs_aligned.proj("p")
            center_shift = mirror_zncc(imgcory, max_shifts=max_shift_px*2) / 2
            template = imgcory.affine(translation=center_shift, mode=Mode.constant, cval=0.)
            
            # Align skew-corrected images to the template
            shifts = np.zeros((npoints, 2))
            for i in range(npoints):
                img = inputs[i]
                shift = -ip.zncc_maximum(template, img, max_shifts=max_shift_px)
                    
                rad = np.deg2rad(skew_angles[i])
                cos, sin = np.cos(rad), np.sin(rad)
                zxrot = np.array([[ cos, sin],
                                  [-sin, cos]], dtype=np.float32)
                shifts[i] = shift @ zxrot

        # Update spline parameters
        spl.shift_coa(shifts=shifts*scale, min_radius=GVar.minCurvatureRadius)
        LOGGER.info(f" >> Shift RMSD = {rmsd(shifts * scale):.3f} nm")
        return self
    
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
            MT radius.
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
        nbin = roundint(r_max/scale/2)
        img2d = subtomograms.proj("py")
        prof = img2d.radial_profile(nbin=nbin, r_max=r_max)
        imax = np.nanargmax(prof)
        imax_sub = centroid(prof, imax-5, imax+5)

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
    ) -> pd.DataFrame:
        """
        Calculate MT local structural parameters from cylindrical Fourier space.
        
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
        pd.DataFrame
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
            coords = spl_trans.local_cylindrical(
                (rmin, rmax), ylen, anc, scale=_scale
            )
            tasks.append(
                da.from_delayed(
                    lazy_ft_params(input_img, coords, spl.radius), 
                    shape=(5,), 
                    meta=np.array([], dtype=np.float32)
                )
            )
        with set_gpu():
            results = np.stack(da.compute(tasks, scheduler=ip.Const["SCHEDULER"])[0], axis=0)
                
        spl.localprops = pd.DataFrame(
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
                polar = ip.asarray(polar, axes="rya", dtype=np.float32) # radius, y, angle
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
        binsize: int = 1
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
        return cft.real ** 2 + cft.imag ** 2
        
    @batch_process
    def global_ft_params(
        self,
        *,
        i: int = None,
        binsize: int = 1,
    ) -> pd.Series:
        """
        Calculate MT global structural parameters.
        
        This function transforms tomogram using cylindrical coordinate system along
        spline. This function calls ``straighten`` beforehand, so that Fourier space is 
        distorted if MT is curved.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        binsize : int, default is 1
            Multiscale bin size used for calculation.

        Returns
        -------
        pd.DataFrame
            Global properties.
        """        
        LOGGER.info(f"Running: {self.__class__.__name__}.global_ft_params, i={i}")
        spl = self._splines[i]        
        img_st = self.straighten_cylindric(i, binsize=binsize)
        series = _local_dft_params_pd(img_st, spl.radius)
        spl.globalprops = series
        return series
    
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
        MT straightening by building curved coordinate system around splines. Currently
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
                i, length, range_, function=self.straighten, 
                chunk_length=chunk_length, size=size
            )
            
        else:
            if size is None:
                rz = rx = 1 + 2 * self.nm2pixel(
                    self._splines[i].radius * GVar.outer, binsize=binsize
                )
            
            else:
                if isinstance(size, Iterable):
                    rz, rx = self.nm2pixel(size, binsize=binsize)
                else:
                    rz = rx = self.nm2pixel(size, binsize=binsize)
            
            input_img = self._get_multiscale_or_original(binsize)
            _scale = input_img.scale.x
            coords = spl.cartesian(
                shape=(rz, rx),
                s_range=range_,
                scale=_scale
            )
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
        MT straightening by building curved coordinate system around splines. Currently
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
                i, length, range_, function=self.straighten_cylindric, 
                chunk_length=chunk_length, radii=radii, binsize=binsize,
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
    
    def _chunked_straighten(
        self, i: int, length: nm, range_: tuple[float, float],
        function: Callable, chunk_length: nm = 72.0, **kwargs
    ):
        out = []
        current_distance: nm = 0.0
        start, end = range_
        spl = self.splines[i]
        while current_distance < length:
            start = current_distance/length
            stop = start + chunk_length/length
            
            # The last segment could be very short
            if spl.length(start=stop, stop=end)/self.scale < 3:
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
    
    @batch_process
    def map_centers(
        self, 
        i: int = None,
        *, 
        interval: nm | None = None,
        length: nm | None = None,
    ) -> Molecules:
        
        spl = self.splines[i]
        props = self.splines[i].globalprops
        if props is None:
            props = self.global_ft_params(i)
        
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        
        # Set interval to the dimer length by default.
        if interval is None:
            interval = lp
        
        # Check length.
        spl_length = spl.length()
        if length is None:
            length = spl_length
        else:
            length = min(length, spl_length)
            
        npoints = length / interval + 1
        skew_angles = np.arange(npoints) * interval/lp * skew
        u = np.arange(npoints) * interval / length
        return spl.anchors_to_molecules(u, rotation=np.deg2rad(skew_angles))
    
    @batch_process
    def map_monomers(
        self, 
        i: int = None,
        *, 
        length: nm | None = None,
        offsets: tuple[nm, float] = None
    ) -> Molecules:
        """
        Map coordinates of monomers in world coordinate.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that mapping will be calculated.
        length : nm, optional
            If given, only map monomer coordinates in this length of range from the starting 
            point of spline. Cannot use this if ``ranges`` is set.
        offsets : tuple of float, optional
            The offset of origin of oblique coordinate system to map monomers.

        Returns
        -------
        Molecules
            Object that represents monomer positions and angles.
        """
        spl = self._splines[i]
        
        if length is None:
            length = spl.length()
            
        # Get structural parameters
        props = self.splines[i].globalprops
        if props is None:
            props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = roundint(props[H.nPF])
        radius = spl.radius
        
        ny = roundint(length/pitch) # number of monomers in y-direction
        tan_rise = tandg(rise)
        
        # Construct meshgrid
        # a-coordinate must be radian.
        # If starting number is non-integer, we must determine the seam position to correctly
        # map monomers. Use integer here.
        shape = [ny, npf]
        tilts = [np.deg2rad(skew)/(4*np.pi)*npf,
                 roundint(-tan_rise*2*np.pi*radius/pitch)/npf]
        intervals = [pitch, 2*np.pi/npf]
        
        if offsets is None:
            offsets = [0., 0.]
        
        mesh = oblique_meshgrid(shape, tilts, intervals, offsets).reshape(-1, 2)
        radius_arr = np.full((mesh.shape[0], 1), radius, dtype=np.float32)
        mesh = np.concatenate([radius_arr, mesh], axis=1)

        mole = spl.cylindrical_to_molecules(mesh)
        mole.features = {Mole.pf: np.arange(len(mole), dtype=np.uint32) % npf}
        return mole
    
    @batch_process
    def map_pf_line(
        self,
        i: int = None,
        *,
        interval: nm | None = None,
        angle_offset: float = 0.0,
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
        props = self.splines[i].globalprops
        if props is None:
            props = self.global_ft_params(i)
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        spl = self._splines[i]
        
        if interval is None:
            interval = lp
        ny = roundint(spl.length()/interval)
        skew_rad = np.deg2rad(skew) * interval / lp
        
        rcoords = np.full(ny, spl.radius)
        ycoords = np.arange(ny) * interval
        acoords = np.arange(ny) * skew_rad + np.deg2rad(angle_offset)
        coords = np.stack([rcoords, ycoords, acoords], axis=1)
        return spl.cylindrical_to_molecules(coords)
    
    #####################################################################################
    #   Utility functions
    #####################################################################################
    
    def collect_anchor_coords(self, i: int | Iterable[int] = None) -> np.ndarray:
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
    
    
    def collect_localprops(self, i: int | Iterable[int] = None) -> pd.DataFrame | None:
        """
        Collect all the local properties into a single pd.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        pd.DataFrame
            Concatenated data frame.
        """        
        if i is None:
            i = range(self.n_splines)
        elif isinstance(i, int):
            i = [i]
        try:
            df = pd.concat(
                [self._splines[i_].localprops for i_ in i], keys=list(i)
            )
            df.index = df.index.rename(["SplineID", "PosID"])
        except ValueError:
            df = None
        
        return df
    
    def collect_globalprops(self, i: int | Iterable[int] = None) -> pd.DataFrame:
        """
        Collect all the global properties into a single pd.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        pd.DataFrame
            Concatenated data frame.
        """        
        if i is None:
            i = range(self.n_splines)
        elif isinstance(i, int):
            i = [i]
        try:
            series_list = []
            for i_ in i:
                series = self._splines[i_].globalprops
                if series is not None:
                    series["radius"] = self._splines[i_].radius
                    series["orientation"] = str(self._splines[i_].orientation)
                series_list.append(series)
            df = pd.concat(series_list, axis=1, keys=list(i)).transpose()
        except ValueError:
            df = None
        return df
    
    def plot_localprops(
        self,
        i: int | Iterable[int] = None,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        **kwargs,
    ):
        """Simple plot function for visualizing local properties."""        
        import seaborn as sns
        df = self.collect_localprops(i)
        data = df.reset_index()
        return sns.swarmplot(x=x, y=y, hue=hue, data=data, **kwargs)
    
    
    def summarize_localprops(
        self, 
        i: int | Iterable[int] = None, 
        by: str | list[str] = "SplineID", 
        functions: Callable[[ArrayLike], Any] | list[Callable[[ArrayLike], Any]] | None = None,
    ) -> pd.DataFrame:
        """Simple summary of local properties."""
        df = self.collect_localprops(i).reset_index()
        if functions is None:
            def se(x): return np.std(x)/np.sqrt(len(x))
            def n(x): return len(x)
            functions = [np.mean, np.std, se, n]
            
        return df.groupby(by=by).agg(functions)
    
    def summarize_globalprops(
        self, 
        functions: Callable[[ArrayLike], Any] | list[Callable[[ArrayLike], Any]] | None = None,
    ) -> pd.DataFrame:
        """Simple summary of global properties."""
        df = self.collect_globalprops()
        valid_colmuns = df.columns[df.columns != "orientation"]
        if functions is None:
            def se(x): return np.std(x)/np.sqrt(len(x))
            def n(x): return len(x)
            functions = [np.mean, np.std, se, n]

        return df[valid_colmuns].agg(functions)
    

def angle_corr(img: ip.ImgArray, ang_center: float = 0, drot: float = 7, nrots: int = 29):
    # img: 3D
    img_z = img.proj("z")
    mask = ip.circular_mask(img_z.shape.y/2+2, img_z.shape)
    img_mirror: ip.ImgArray = img_z["x=::-1"]
    angs = np.linspace(ang_center-drot, ang_center+drot, nrots, endpoint=True)
    corrs = []
    f0 = np.sqrt(img_z.power_spectra(dims="yx", zero_norm=True))
    cval = np.mean(img_z)
    for ang in angs:
        img_mirror_rot = img_mirror.rotate(ang*2, mode=Mode.constant, cval=cval)
        f1 = np.sqrt(img_mirror_rot.power_spectra(dims="yx", zero_norm=True))
        corr = ip.zncc(f0, f1, mask)
        corrs.append(corr)
        
    angle = angs[np.argmax(corrs)]
    return angle


def dask_angle_corr(imgs, ang_centers, drot: float = 7, nrots: int = 29):
    _angle_corr = delayed(partial(angle_corr, drot=drot, nrots=nrots))
    tasks = []
    for img, ang in zip(imgs, ang_centers):
        tasks.append(da.from_delayed(_angle_corr(img, ang), shape=(), dtype=np.float32))
    return da.compute(tasks, scheduler=ip.Const["SCHEDULER"])[0]

def _local_dft_params(img: ip.ImgArray, radius: nm):
    img = img - img.mean()
    l_circ: nm = 2*np.pi*radius
    npfmin = GVar.nPFmin
    npfmax = GVar.nPFmax
    
    # First transform around the expected length of y-pitch.
    ylength_nm = img.shape.y * img.scale.y
    y0 = ceilint(ylength_nm/GVar.yPitchMax) - 1
    y1 = max(ceilint(ylength_nm/GVar.yPitchMin), y0+1)
    up_a = 20
    up_y = max(int(6000/img.shape.y), 1)
    npfrange = ceilint(npfmax/2) # The peak of longitudinal periodicity is always in this range. 
    
    power = img.local_power_spectra(
        key=f"y={y0}:{y1};a={-npfrange}:{npfrange+1}", 
        upsample_factor=[1, up_y, up_a], 
        dims="rya",
    ).proj("r")
    
    ymax, amax = np.unravel_index(np.argmax(power), shape=power.shape)
    ymaxp = np.argmax(power.proj("a"))
    
    amax_f = amax - npfrange*up_a
    ymaxp_f = ymaxp + y0*up_y
    ymax_f = ymax + y0*up_y
    a_freq = np.fft.fftfreq(img.shape.a*up_a)
    y_freq = np.fft.fftfreq(img.shape.y*up_y)
    
    rise = np.arctan(-a_freq[amax_f]/y_freq[ymax_f])
    y_pitch = 1.0/y_freq[ymaxp_f]*img.scale.y
    
    # Second, transform around 13 pf lateral periodicity.
    # This analysis measures skew angle and protofilament number.
    y_factor = abs(radius/y_pitch/img.shape.a*img.shape.y/2)
    dy_min = ceilint(tandg(GVar.minSkew)*y_factor*npfmin) - 1
    dy_max = max(ceilint(tandg(GVar.maxSkew)*y_factor*npfmax), dy_min+1)
    up_a = 20
    up_y = max(int(21600/(img.shape.y)), 1)
    
    power = img.local_power_spectra(
        key=f"y={dy_min}:{dy_max};a={npfmin}:{npfmax}", 
        upsample_factor=[1, up_y, up_a], 
        dims="rya",
    ).proj("r")
    
    ymax, amax = np.unravel_index(np.argmax(power), shape=power.shape)
    amaxp = np.argmax(power.proj("y"))
    
    amax_f = amax + npfmin*up_a
    amaxp_f = amaxp + npfmin*up_a
    ymax_f = ymax + dy_min*up_y
    a_freq = np.fft.fftfreq(img.shape.a*up_a)
    y_freq = np.fft.fftfreq(img.shape.y*up_y)
    
    # When skew angle is positive and y-coordinate increses, a-coordinate will
    # decrese.
    skew = np.arctan(y_freq[ymax_f]/a_freq[amax_f]*2*y_pitch/radius)
    
    if rise == 0.0:
        start = 0.0
    else:
        start = l_circ/y_pitch/(np.tan(skew) + 1/np.tan(rise))
    
    return np.array([np.rad2deg(rise), 
                     y_pitch, 
                     np.rad2deg(skew), 
                     amaxp_f/up_a,
                     abs(start)], 
                    dtype=np.float32)

def _local_dft_params_pd(img: ip.ImgArray, radius: nm):
    results = _local_dft_params(img, radius)
    series = pd.Series([], dtype=np.float32)
    series[H.riseAngle] = results[0]
    series[H.yPitch] = results[1]
    series[H.skewAngle] = results[2]
    series[H.nPF] = np.round(results[3])
    series[H.start] = results[4]
    return series

def ft_params(img: ip.ImgArray | ip.LazyImgArray, coords: np.ndarray, radius: nm):
    polar = map_coordinates(img, coords, order=3, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32) # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
    polar.scale_unit = img.scale_unit
    return _local_dft_params(polar, radius)


lazy_ft_params = delayed(ft_params)


def try_all_seams(
    loader: SubtomogramLoader,
    npf: int,
    template: ip.ImgArray,
    mask: ip.ImgArray | None = None,
) -> tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]:
    """
    Try all the possible seam positions and compare correlations.

    Parameters
    ----------
    loader : SubtomogramLoader
        An aligned ``acryo.SubtomogramLoader`` object.
    npf : int
        Number of protofilament.
    template : ip.ImgArray
        Template image.
    mask : ip.ImgArray, optional
        Mask image.

    Returns
    -------
    tuple[np.ndarray, ip.ImgArray, list[np.ndarray]]
        Correlation, average and boolean array correspond to each seam position.
    """    
    corrs: list[float] = []
    labels: list[np.ndarray] = []  # list of boolean arrays
    
    if mask is None:
        mask = 1.
    
    masked_template = template * mask
    _id = np.arange(len(loader.molecules))
    
    # prepare all the labels in advance (only takes up ~0.5 MB at most)
    for pf in range(2*npf):
        res = (_id - pf) // npf
        sl = res % 2 == 0
        labels.append(sl)
    
    dask_array = loader.construct_dask(output_shape=template.shape)
    averaged_images = da.compute([da.mean(dask_array[sl], axis=0) for sl in labels])[0]
    averaged_images = ip.asarray(np.stack(averaged_images, axis=0), axes="pzyx")
    averaged_images.set_scale(zyx=loader.scale)
    
    corrs = [ip.zncc(avg*mask, masked_template) for avg in averaged_images]
    
    return np.array(corrs), averaged_images, labels