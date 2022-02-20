from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpecKwargs
else:
    from typing import ParamSpecKwargs
    
from typing import Callable, Iterable, Any, TypeVar, overload, Protocol
import json
from functools import partial, wraps
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from dask import array as da, delayed

import impy as ip

from .molecules import Molecules
from .const import nm, H, K, Ori, Mode, GVar
from .spline import Spline
from .averaging import SubtomogramLoader
from .utils import (
    load_a_subtomogram,
    centroid,
    map_coordinates,
    multi_map_coordinates,
    roundint,
    load_rot_subtomograms,
    ceilint,
    oblique_meshgrid,
    no_verbose,
    mirror_pcc,
    mirror_ft_pcc,
    angle_uniform_filter,
    )


LOCALPROPS = [H.splPosition, H.splDistance, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]

if ip.Const["RESOURCE"] == "cupy":
    SCHEDULER = "single-threaded"
else:
    SCHEDULER = "threads"


def tandg(x):
    """Tangent in degree."""
    return np.tan(np.deg2rad(x))

_KW = ParamSpecKwargs("_KW")
_RETURN = TypeVar("_RETURN")

class BatchCallable(Protocol[_RETURN]):
    """
    This protocol enables static type checking of methods decorated with ``@batch_process``.
    The parameter specifier ``_KW`` does not add any information but currently there is not 
    quick solution.
    """
    @overload
    def __call__(self, **kwargs: _KW) -> list[_RETURN]:
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
            with no_verbose():
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
        with no_verbose():
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


def json_encoder(obj: Any):
    
    """Enable Enum and pandas encoding."""
    if isinstance(obj, Ori):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")


class MtSpline(Spline):
    """
    A spline object with information related to MT.
    """    
    _local_cache = (K.localprops,)
    _global_cache = (K.globalprops, K.radius, K.orientation, K.cart_stimg, K.cyl_stimg)
    
    def __init__(self, scale: float = 1.0, k: int = 3, *, lims: tuple[float, float] = (0., 1.)):
        """
        Spline object for MT.
        
        Parameters
        ----------
        scale : float, default is 1.0
            Pixel scale
        k : int, default is 3
            Spline order.
        """        
        super().__init__(scale=scale, k=k, lims=lims)
        self.orientation = Ori.none
        self.radius: nm = None
        self.localprops: pd.DataFrame = None
        self.globalprops: pd.Series = None
        self.cart_stimg: ip.ImgArray = None
        self.cyl_stimg: ip.ImgArray = None
    
    def invert(self) -> MtSpline:
        """
        Invert the direction of spline. Also invert orientation if exists.

        Returns
        -------
        Spline3D
            Inverted object
        """
        inverted: MtSpline = super().invert()
        inverted.radius = self.radius
        inverted.globalprops = self.globalprops
        inverted.localprops = None
        inverted.cart_stimg = None
        inverted.cyl_stimg = None
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
        if start > stop:
            if self.orientation == Ori.PlusToMinus:
                clipped.orientation = Ori.MinusToPlus
            elif self.orientation == Ori.MinusToPlus:
                clipped.orientation = Ori.PlusToMinus
        else:
            clipped.orientation = self.orientation
        return clipped
            
        
    @property
    def orientation(self) -> Ori:
        return self._orientation
    
    @orientation.setter
    def orientation(self, value: Ori | str):
        try:
            self._orientation = Ori(value)
        except ValueError:
            self._orientation = Ori.none
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d[K.radius] = self.radius
        d[K.orientation] = self.orientation.name
        if self.localprops is not None:
            cols = [l for l in LOCALPROPS if l in self.localprops.columns]
            d[K.localprops] = self.localprops[cols]
        if self.globalprops is not None:
            d[K.globalprops] = self.globalprops
        return d
        
    @classmethod
    def from_dict(cls, d: dict):
        self = super().from_dict(d)
        localprops = d.get(K.localprops, None)
        if localprops is not None and H.splPosition in localprops:
            self.anchors = localprops[H.splPosition]
        self.radius = d.get(K.radius, None)
        self.orientation = d.get(K.orientation, Ori.none)
        if localprops is None:
            self.localprops = None
        else:
            self.localprops = pd.DataFrame(localprops)
        globalprops = d.get(K.globalprops, None)
        if globalprops is None:
            self.globalprops = globalprops
        else:
            self.globalprops = pd.Series(globalprops)
        return self
        

class MtTomogram:
    """
    Tomogram object. It always connected to a 3D image but processed lazily. Thus
    you can create a lot of MtTomogram objects without MemoryError. Subtomograms
    are temporarily loaded into memory via cache map. Once memory usage exceed
    certain amount, the subtomogram cache will automatically deleted from the old
    ones.
    """
    _image: ip.LazyImgArray
    def __init__(self, 
                 *,
                 subtomogram_length: nm = 48.0,
                 subtomogram_width: nm = 40.0,
                 light_background: bool = False,
                 ):
        self.subtomo_length = subtomogram_length
        self.subtomo_width = subtomogram_width
        self._splines: list[MtSpline] = []
        self.light_background = light_background
        self.metadata: dict[str, Any] = {}
        
    
    def __hash__(self) -> int:
        return id(self)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} of '{self._image.name}'"
            
    @property
    def splines(self) -> list[MtSpline]:
        """List of splines."""
        return self._splines
    
    @property
    def n_splines(self) -> int:
        """Number of spline paths."""
        return len(self._splines)
    
    @classmethod
    def imread(
        cls, 
        path: str,
        *, 
        scale: float = None,
        subtomogram_length: nm = 48.0,
        subtomogram_width: nm = 40.0,
        light_background: bool = True
    ) -> MtTomogram:
        
        self = cls(subtomogram_length=subtomogram_length,
                   subtomogram_width=subtomogram_width,
                   light_background=light_background)
        img = ip.lazy_imread(path, chunks=GVar.daskChunk).as_float()
        if scale is not None:
            img.set_scale(xyz=scale)
        else:
            if (abs(img.scale.z - img.scale.x) > 1e-4
                or abs(img.scale.z - img.scale.y) > 1e-4):
                raise ValueError("Uneven scale.")
        
        self._set_image(img)
        return self
        
    @property
    def image(self) -> ip.LazyImgArray:
        """Tomogram image data."""
        return self._image
    
    def _set_image(self, img: ip.LazyImgArray | np.ndarray) -> None:
        if isinstance(img, ip.LazyImgArray):
            pass
        elif isinstance(img, np.ndarray):
            if img.ndim != 3:
                raise ValueError("Can only set 3-D image.")
            img = ip.aslazy(img, dtype=np.float32, axes="zyx", chunks=GVar.daskChunk)
        else:
            raise TypeError(f"Cannot set type {type(img)} as an image.")
        if (abs(img.scale.z - img.scale.x) > 1e-4
            or abs(img.scale.z - img.scale.y) > 1e-4):
            raise ValueError("Uneven scale.")
        self.scale = img.scale.x
        self._image = img
        return None
    
    
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
    
    def save_json(self, path: str) -> None:
        """
        Save splines with its local properties as a json file.

        Parameters
        ----------
        path : str
            File path to save file.
        """        
        path = str(path)
        
        all_results = {}
        for i, spl in enumerate(self._splines):
            spl_dict = spl.to_dict()
            all_results[i] = spl_dict
        
        from .__init__ import __version__
        metadata = self.metadata.copy()
        metadata["light_background"] = self.light_background
        metadata["version"] = __version__
        all_results["metadata"] = metadata

        with open(path, mode="w") as f:
            json.dump(all_results, f, indent=4, separators=(",", ": "), default=json_encoder)
        return None
    
    
    def load_json(self, file_path: str) -> MtTomogram:
        """
        Load splines from a json file.

        Parameters
        ----------
        file_path : str
            File path to the json file.

        Returns
        -------
        MtTomogram
            Same object with spline curves loaded.
        """        
        file_path = str(file_path)
        
        with open(file_path, mode="r") as f:
            js: dict = json.load(f)
        
        for i, d in js.items():
            try:
                # integer key means it's spline info
                int(i)
            except:
                setattr(self, i, d)
            else:
                self._splines.append(
                    MtSpline.from_dict(d)
                )
            
        return self
    
    
    def add_spline(self, coords: ArrayLike) -> None:
        """
        Add MtSpline path to tomogram.

        Parameters
        ----------
        coords : array-like
            (N, 3) array of coordinates. A spline curve that fit it well is added.
        """        
        spl = MtSpline(self.scale, k=GVar.splOrder)
        coords = np.asarray(coords)
        sqsum = GVar.splError**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        interval: nm = 30.0
        length = spl.length()
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        if coords.shape[0] <= GVar.splOrder and coords.shape[0] < fit.shape[0]:
            return self.add_spline(fit)
        
        self._splines.append(spl)
        return None
    
    @overload
    def nm2pixel(self, value: nm) -> int:
        ...
        
    @overload
    def nm2pixel(self, value: Iterable[nm]) -> np.ndarray:
        ...
    
    def nm2pixel(self, value):
        """
        Convert nm float value into pixel value. Useful for conversion from 
        coordinate to pixel position.

        Returns
        -------
        np.ndarray or int
            Pixel position.
        """        
        pix = np.round(np.asarray(value)/self.scale).astype(np.int16)
        if np.isscalar(value):
            pix = int(pix)
        return pix
    
    @batch_process
    def make_anchors(self, i = None, *, interval: nm = None, n: int = None, max_interval: nm = None):
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
    
    
    def align_to_polarity(self, orientation: Ori | str = Ori.MinusToPlus) -> MtTomogram:
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
        
    
    def collect_anchor_coords(self, i: int|Iterable[int] = None) -> np.ndarray:
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
    
    
    def collect_localprops(self, i: int | Iterable[int] = None) -> pd.DataFrame:
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
        df = pd.concat([self._splines[i_].localprops for i_ in i], 
                        keys=list(i)
                       )
        
        df.index = df.index.rename(["SplineID", "PosID"])
        return df
    
    
    def plot_localprops(self, i: int | Iterable[int] = None,
                        x=None, y=None, hue=None, **kwargs):
        """
        Simple plot function for visualizing local properties.
        """        
        import seaborn as sns
        df = self.collect_localprops(i)
        data = df.reset_index()
        return sns.swarmplot(x=x, y=y, hue=hue, data=data, **kwargs)
    
    
    def summerize_localprops(
        self, 
        i: int | Iterable[int] = None, 
        by: str | list[str] = "SplineID", 
        functions: Callable[[ArrayLike], Any] | list[Callable[[ArrayLike], Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Simple summerize of local properties.
        """
        df = self.collect_localprops(i).reset_index()
        if functions is None:
            def se(x): return np.std(x)/np.sqrt(len(x))
            def n(x): return len(x)
            functions = [np.mean, np.std, se, n]
            
        return df.groupby(by=by).agg(functions)
    
    
    def collect_radii(self, i: int | Iterable[int] = None) -> np.ndarray:
        """
        Collect all the radius into a single array.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        np.ndarray
            Radius of each spline
        """        
        if i is None:
            i = range(self.n_splines)
        elif isinstance(i, int):
            i = [i]
        return np.array([self._splines[i_].radius for i_ in i])
    
    
    def _sample_subtomograms(
        self, 
        i: int,
        rotate: bool = True
    ) -> ip.ImgArray:
        spl = self._splines[i]
        length_px = self.nm2pixel(self.subtomo_length)
        width_px = self.nm2pixel(self.subtomo_width)
        
        if rotate:
            out = load_rot_subtomograms(self.image, length_px, width_px, spl)
            
        else:
            # If subtomogram region is rotated by 45 degree, its XY-width will be
            # sqrt(2) * (length + width)
            center_px = self.nm2pixel(spl())
            size_px = (width_px,) + (roundint((width_px+length_px)/1.41),)*2
            
            out = np.stack([load_a_subtomogram(self._image, c, size_px) 
                            for c in center_px],
                            axis="p")
                
        return out
    
    @batch_process
    def fit(
        self, 
        i = None,
        *, 
        max_interval: nm = 30.0,
        degree_precision: float = 0.5,
        cutoff: float = 0.2,
        dense_mode: bool = False,
        dense_mode_sigma: nm = 2.0,
    ) -> MtTomogram:
        """
        Roughly fit i-th spline to MT.
        
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
        degree_precision : float, default is 0.2
            Precision of MT xy-tilt degree in angular correlation.
        cutoff : float, default is 0.35
            The cutoff frequency of lowpass filter that will applied to subtomogram 
            before alignment-based fitting. 
        dense_mode : bool, default is False
            If True, fitting will be executed in the dense-microtubule mode.
        dense_mode_sigma : nm, default is 2.0
            Sharpness of mask at the edges. Soft mask is important for precision
            because sharp changes in intensity cause strong correlation at the edges.

        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """        
        spl = self._splines[i]
        spl.make_anchors(max_interval=max_interval)
        npoints = spl.anchors.size
        interval = spl.length()/(npoints-1)
        subtomograms = self._sample_subtomograms(i, rotate=False)
        subtomograms -= subtomograms.mean()
        subtomograms: ip.ImgArray
        if 0 < cutoff < 0.866:
            subtomograms = subtomograms.lowpass_filter(cutoff)
        
        if dense_mode:
            # mask XY-region outside the microtubules with sigmoid function.
            yy, xx = np.indices(subtomograms.sizesof("yx"))
            yc, xc = np.array(subtomograms.sizesof("yx"))/2 - 0.5
            yr = yy - yc
            xr = xx - xc
            for i, ds in enumerate(spl(der=1)):
                _, vy, vx = ds
                distance: nm = np.abs(-xr*vy + yr*vx) / np.sqrt(vx**2 + vy**2) * self.scale
                distance_cutoff = self.subtomo_width / 2
                if dense_mode_sigma == 0:
                    mask_yx = (distance > distance_cutoff).astype(np.float32)
                else:
                    mask_yx = 1 / (1 + np.exp((distance - distance_cutoff)/dense_mode_sigma))
                mask = np.stack([mask_yx]*subtomograms.shape.z, axis=0)
                subtomograms[i] *= mask
        
        ds = spl(der=1)
        yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
        degree_max = 14.0
        nrots = roundint(degree_max/degree_precision) + 1

        # Angular correlation
        out = dask_angle_corr(subtomograms, yx_tilt, nrots=nrots)
        refined_tilt_deg = np.array(out)
        refined_tilt_rad = np.deg2rad(refined_tilt_deg)
        
        # If subtomograms are sampled at short intervals, angles should be smoothened to 
        # avoid overfitting.
        size = 2*roundint(48.0/interval) + 1
        if size > 1:
            # Mirror-mode padding is "a b c d | c b a".
            refined_tilt_rad = angle_uniform_filter(
                refined_tilt_rad, size=size, mode=Mode.mirror
                )
            refined_tilt_deg = np.rad2deg(refined_tilt_rad)
        
        # Rotate subtomograms            
        for i, img in enumerate(subtomograms):
            angle = refined_tilt_deg[i]
            img.rotate(-angle, cval=0, update=True)
            
        # zx-shift correction by self-PCC
        subtomo_proj = subtomograms.proj("y")
        
        if dense_mode:
            # Regions outside the mask don't need to be considered.
            xc = int(subtomo_proj.shape.x/2)
            w = int((self.subtomo_width/self.scale)/2)
            subtomo_proj = subtomo_proj[f"x={xc-w}:{xc+w+1}"]

        shape = subtomo_proj[0].shape
        shifts = np.zeros((npoints, 2)) # zx-shift
        mask_yx = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        for i in range(npoints):
            img = subtomo_proj[i]
            shifts[i] = mirror_pcc(img, mask=mask_yx) / 2
        
        # Update spline coordinates.
        # Because centers of subtomogram are on lattice points of pixel coordinate,
        # coordinates that will be shifted should be converted to integers. 
        coords_px = self.nm2pixel(spl()).astype(np.float32)
        
        shifts_3d = np.stack([shifts[:, 0],
                              np.zeros(shifts.shape[0]), 
                              shifts[:, 1]], 
                             axis=1)
        rotvec = np.zeros(shifts_3d.shape, dtype=np.float32)
        rotvec[:, 0] = -refined_tilt_rad
        rot = Rotation.from_rotvec(rotvec)
        coords_px += rot.apply(shifts_3d)
            
        coords = coords_px * self.scale
        
        # Update spline parameters
        sqsum = GVar.splError**2 * coords.shape[0]  # unit: nm^2
        spl.fit(coords, s=sqsum)
        
        return self
    
    @batch_process
    def refine(
        self, 
        i = None,
        *, 
        max_interval: nm = 30.0,
        cutoff: float = 0.35,
        projection: bool = True,
        corr_allowed: float = 0.9,
    ) -> MtTomogram:
        """
        Refine spline using the result of previous fit and the global structural parameters.
        During refinement, Y-projection of MT XZ cross section is rotated with the skew angle,
        thus is much more precise than the coarse fitting.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to fit.
        max_interval : nm, default is 24.0
            Maximum interval of sampling points in nm unit.
        cutoff : float, default is 0.35
            The cutoff frequency of lowpass filter that will applied to subtomogram 
            before alignment-based fitting. 
        projection: bool, default is True
            If true, 2-D images of projection along the longitudinal axis are used for boosting
            correlation calculation. Otherwise 3-D images will be used.
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        
        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """
        spl = self.splines[i]
        if spl.radius is None:
            spl.make_anchors(n=3)
            self.measure_radius(i=i)
            
        props = self.global_ft_params(i)
        spl.make_anchors(max_interval=max_interval)
        npoints = spl.anchors.size
        interval = spl.length()/(npoints-1)
            
        # Calculate Fourier parameters by cylindrical transformation along spline.
        # Skew angles are divided by the angle of single protofilament and the residual
        # angles are used, considering missing wedge effect.
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        npf = roundint(props[H.nPF])
        
        skew_angles = np.arange(npoints) * interval/lp * skew
        pf_ang = 360/npf
        skew_angles %= pf_ang
        skew_angles[skew_angles > pf_ang/2] -= pf_ang
        
        length_px = self.nm2pixel(self.subtomo_length)
        width_px = self.nm2pixel(self.subtomo_width)
        
        images: list[ip.ImgArray] = []
        mole = spl.anchors_to_molecules(rotation=np.deg2rad(skew_angles))
        
        # Load subtomograms rotated by skew angles. All the subtomograms should look similar.
        chunksize = max(int(self.subtomo_length*2/interval), 1)
        for coords in mole.iter_cartesian((width_px, length_px, width_px), 
                                          self.scale, chunksize=chunksize):
            _subtomo = multi_map_coordinates(self.image, coords, order=1, cval=np.mean)
            images.extend(_subtomo)
        
        subtomograms = ip.asarray(np.stack(images, axis=0), axes="pzyx")
        subtomograms: ip.ImgArray = subtomograms - subtomograms.mean()  # normalize
        if 0 < cutoff < 0.866:
            subtomograms = subtomograms.lowpass_filter(cutoff)

        # prepare input images according to the options.
        if projection:
            inputs = subtomograms.proj("y")["x=::-1"]
        else:
            inputs = subtomograms["x=::-1"]
        
        inputs_ft = inputs.fft(dims=inputs["p=0"].axes)
        
        # Coarsely align skew-corrected images
        shape = inputs["p=0"].shape
        
        # prepare a mask image for PCC calculation
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
            
        imgs_aligned: ip.ImgArray = ip.empty(inputs.shape, dtype=np.float32, axes=inputs.axes)
        
        for i in range(npoints):
            img: ip.ImgArray = inputs[i]
            ft = inputs_ft[i]
            shift = mirror_ft_pcc(ft, mask=mask) / 2
            if not projection:
                shift[1] = 0
            imgs_aligned.value[i] = img.affine(translation=shift, mode=Mode.constant, cval=0)
            
        if corr_allowed < 1:
            # remove low correlation image from calculation of template image.
            corrs = ip.zncc(imgs_aligned, imgs_aligned["z=::-1;x=::-1"])
            threshold = np.quantile(corrs, 1 - corr_allowed)
            indices: np.ndarray = np.where(corrs >= threshold)[0]
            imgs_aligned = imgs_aligned[indices.tolist()]
        
        # Make template using coarse aligned images.
        imgcory: ip.ImgArray = imgs_aligned.proj("p")
        center_shift = mirror_pcc(imgcory, mask=mask) / 2
        template = imgcory.affine(translation=center_shift, mode=Mode.constant, cval=0)
        template_ft = template.fft(dims=template.axes)
        
        # Align skew-corrected images to the template
        shifts = np.zeros((npoints, 2))
        for i in range(npoints):
            ft = inputs_ft[i]
            shift = -ip.ft_pcc_maximum(template_ft, ft, mask=mask)
            
            if not projection:
                shift = shift[[0, 2]]
                
            rad = np.deg2rad(skew_angles[i])
            cos, sin = np.cos(rad), np.sin(rad)
            zxrot = np.array([[ cos, sin],
                              [-sin, cos]], dtype=np.float32)
            shifts[i] = shift @ zxrot

        # Update spline parameters
        sqsum = GVar.splError**2 * npoints # unit: nm^2
        spl.shift_fit(shifts=shifts*self.scale, s=sqsum)
        return self
                
    @batch_process
    def get_subtomograms(self, i = None) -> ip.ImgArray:
        """
        Get subtomograms at anchors. All the subtomograms are rotated to oriented
        to the spline.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to load samples.

        Returns
        -------
        ip.ImgArray
            Subtomograms stacked along "p" axis.
        """        
        subtomograms = self._sample_subtomograms(i)
        return subtomograms
    
    @batch_process
    def measure_radius(self, i = None) -> nm:
        """
        Measure MT radius using radial profile from the center.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to measure.

        Returns
        -------
        float (nm)
            MT radius.
        """        
        if self.splines[i]._anchors is None:
            self.splines[i].make_anchors(n=3)
            
        subtomograms = self._sample_subtomograms(i)
        r_max = self.subtomo_width / 2
        nbin = roundint(r_max/self.scale/2)
        img2d = subtomograms.proj("py")
        prof = img2d.radial_profile(nbin=nbin, r_max=r_max)

        # determine precise radius using centroid    
        if self.light_background:
            prof = -prof
        
        imax = np.argmax(prof)
        imax_sub = centroid(prof, imax-5, imax+5)
        
        # prof[0] is radial profile at r=0.5 (not r=0.0)
        r_peak_sub = (imax_sub + 0.5) / nbin * r_max
        
        self._splines[i].radius = r_peak_sub
        return r_peak_sub
    
    @batch_process
    def local_ft_params(self, i = None, ft_size: nm = 32.0) -> pd.DataFrame:
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

        Returns
        -------
        pd.DataFrame
            Local properties.
        """        
        spl = self.splines[i]
        if spl.localprops is not None:
            return spl.localprops
        
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")
        
        ylen = self.nm2pixel(ft_size)
        rmin = spl.radius*GVar.inner/self.scale
        rmax = spl.radius*GVar.outer/self.scale
        tasks = []
        for anc in spl.anchors:
            coords = spl.local_cylindrical((rmin, rmax), ylen, anc)
            coords = np.moveaxis(coords, -1, 0)
            tasks.append(
                da.from_delayed(lazy_ft_params(self.image, coords, spl.radius), 
                                shape=(5,), 
                                meta=np.array([], dtype=np.float32)
                                )
                )
        results = np.stack(da.compute(tasks, scheduler=SCHEDULER)[0], axis=0)
                
        spl.localprops = pd.DataFrame([])
        spl.localprops[H.splPosition] = spl.anchors
        spl.localprops[H.splDistance] = spl.distances()
        spl.localprops[H.riseAngle] = results[:, 0]
        spl.localprops[H.yPitch] = results[:, 1]
        spl.localprops[H.skewAngle] = results[:, 2]
        spl.localprops[H.nPF] = np.round(results[:, 3]).astype(np.uint8)
        spl.localprops[H.start] = results[:, 4]
        
        return spl.localprops
    
    @batch_process
    def local_cft(self, i = None, ft_size: nm = 32.0) -> ip.ImgArray:
        """
        Calculate non-upsampled local cylindric Fourier transormation along spline. 

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        ft_size : nm, default is 32.0
            Length of subtomogram for calculation of local parameters.

        Returns
        -------
        ip.ImgArray
            FT images stacked along "p" axis.
        """
        spl = self.splines[i]
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")
        
        ylen = self.nm2pixel(ft_size)
        rmin = spl.radius*GVar.inner/self.scale
        rmax = spl.radius*GVar.outer/self.scale
        out: list[ip.ImgArray] = []
        with no_verbose():
            for anc in spl.anchors:
                coords = spl.local_cylindrical((rmin, rmax), ylen, anc)
                coords = np.moveaxis(coords, -1, 0)
                polar = map_coordinates(self.image, coords, order=3, mode=Mode.constant, cval=np.mean)
                polar = ip.asarray(polar, axes="rya", dtype=np.float32) # radius, y, angle
                polar.set_scale(r=self.scale, y=self.scale, a=self.scale)
                polar.scale_unit = self.image.scale_unit
                polar -= np.mean(polar)
                out.append(polar.fft(dims="rya"))
        
        return np.stack(out, axis="p")
        
    @batch_process
    def global_ft_params(self, i = None) -> pd.Series:
        """
        Calculate MT global structural parameters from cylindrical Fourier space along 
        spline. This function calls ``straighten`` beforehand, so that Fourier space is 
        distorted if MT is curved.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.

        Returns
        -------
        pd.DataFrame
            Global properties.
        """        
        spl = self._splines[i]
        if spl.globalprops is not None:
            return spl.globalprops
        
        img_st = self.straighten_cylindric(i)
        series = _local_dft_params_pd(img_st, spl.radius)
        spl.globalprops = series
        return series
    
    @batch_process
    def global_cft(self, i = None) -> ip.ImgArray:
        """
        Calculate global cylindrical fast Fourier tranformation.
        
        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to analyze.
        
        
        Returns
        -------
        ip.ImgArray
            Complex image.
        """
        img_st: ip.ImgArray = self.straighten_cylindric(i)
        img_st -= np.mean(img_st)
        return img_st.fft(dims="rya")

    @batch_process
    def straighten(self, 
                   i = None, 
                   *,
                   size: nm | tuple[nm, nm] = None,
                   range_: tuple[float, float] = (0.0, 1.0), 
                   chunk_length: nm = 72.0) -> ip.ImgArray:
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
        chunk_length : nm, default is 72.0
            If spline is longer than this, it will be first split into chunks, straightened respectively and
            all the straightened images are concatenated afterward, to avoid loading entire image into memory.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """        
        try_cache = size is None and range_ == (0.0, 1.0)
        spl = self.splines[i]
        if try_cache and spl.cart_stimg is not None:
            return spl.cart_stimg
        
        length = self._splines[i].length(nknots=512)
        
        if length > chunk_length:
            transformed = self._chunked_straighten(
                i, length, range_, function=self.straighten, 
                chunk_length=chunk_length, size=size
            )
            
        else:
            if size is None:
                rz = rx = self.nm2pixel(self._splines[i].radius * GVar.outer) * 2 + 1
            
            else:
                if isinstance(size, Iterable):
                    rz, rx = self.nm2pixel(size)
                else:
                    rz = rx = self.nm2pixel(size)
                    
            coords = spl.cartesian((rz, rx), s_range=range_)
            coords = np.moveaxis(coords, -1, 0)
            
            transformed = map_coordinates(self.image, coords, order=1)
            
            axes = "zyx"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: self.scale for k in axes})
            transformed.scale_unit = "nm"
        
        if try_cache:
            spl.cart_stimg = transformed
        
        return transformed

    @batch_process
    def straighten_cylindric(self, 
                             i = None, 
                             *,
                             radii: tuple[nm, nm] = None,
                             range_: tuple[float, float] = (0.0, 1.0), 
                             chunk_length: nm = 72.0) -> ip.ImgArray:
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
        chunk_length : nm, default is 72.0
            If spline is longer than this, it will be first split into chunks, straightened respectively and
            all the straightened images are concatenated afterward, to avoid loading entire image into memory.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """        
        try_cache = radii is None and range_ == (0.0, 1.0)
        spl = self.splines[i]
        
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")
        
        if try_cache and spl.cyl_stimg is not None:
            return spl.cyl_stimg
        
        length = self._splines[i].length(nknots=512)
        
        if length > chunk_length:
            transformed = self._chunked_straighten(
                i, length, range_, function=self.straighten_cylindric, 
                chunk_length=chunk_length, radii=radii
                )
            
        else:
            if radii is None:
                inner_radius = spl.radius * GVar.inner / self.scale
                outer_radius = spl.radius * GVar.outer / self.scale
                
            else:
                inner_radius, outer_radius = radii / self.scale

            if outer_radius <= inner_radius:
                raise ValueError("For cylindrical straightening, 'radius' must be (rmin, rmax)")
            
            coords = spl.cylindrical((inner_radius, outer_radius), s_range=range_)
            coords = np.moveaxis(coords, -1, 0)
            
            transformed = map_coordinates(self.image, coords, order=3)
            
            axes = "rya"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: self.scale for k in axes})
            transformed.scale_unit = "nm"
        
        if try_cache:
            spl.cyl_stimg = transformed
        
        return transformed
    
    def _chunked_straighten(self, i: int, length: nm, range_: tuple[float, float],
                            function: Callable, chunk_length: nm = 72.0, **kwargs):
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
    def reconstruct(self, 
                    i = None,
                    *, 
                    rot_ave: bool = False,
                    erase_corner: bool = True,
                    niter: int = 1,
                    y_length: nm = 50.0) -> ip.ImgArray:
        """
        3D reconstruction of MT.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to reconstruct.
        rot_ave : bool, default is False
            If true, rotational averaging is applied to the reconstruction to remove missing wedge.
        erase_corner : bool, default is True
            Substitute four corners with median of non-corner domain. This option is useful for iso
            threshold visualization.
        y_length : nm, default is 100.0
            Longitudinal length of reconstruction.

        Returns
        -------
        ip.ImgArray
            Reconstructed image.
        """                
        # Cartesian transformation along spline.
        img_st = self.straighten(i)
        scale = img_st.scale.y
        total_length: nm = img_st.shape.y*scale
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
        props = self.global_ft_params(i)
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = int(props[H.nPF])
        radius = self.splines[i].radius
        
        # Determine how to split image into tubulin dimer fragments
        dl, resl = divmod(total_length, lp)
        borders = np.linspace(0, total_length - resl, int((total_length - resl)/lp)+1)
        skew_angles = np.arange(borders.size - 1) * skew
        
        # Rotate fragment with skew angle
        imgs: list[ip.ImgArray] = []
        ylen = 99999
    
        # Split image into dimers along y-direction
        for start, stop, ang in zip(borders[:-1], borders[1:], skew_angles):
            start = self.nm2pixel(start)
            stop = self.nm2pixel(stop)
            imgs.append(img_st[:, start:stop].rotate(ang, dims="zx", mode=Mode.reflect))
            ylen = min(ylen, stop-start)
        
        # Make image sizes same and prepare FT images.
        imgs = [img[f"y=:{ylen}"] for img in imgs]
        ft_imgs = [img.fft() for img in imgs]

        # align each fragment
        ref = imgs[0]
        shape = ref.shape
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        imgs_aligned: list[ip.ImgArray] = []
        for _ in range(niter):
            ft_ref = ref.fft()
            imgs_aligned.clear()
            for k in range(len(imgs)):
                img = imgs[k]
                ft_img = ft_imgs[k]
                shift = ip.ft_pcc_maximum(ft_img, ft_ref, mask=mask)
                imgs_aligned.append(
                    img.affine(translation=shift, mode=Mode.grid_wrap)
                    )

            out: ip.ImgArray = np.stack(imgs_aligned, axis="p").proj("p")
            ref = out
                    
        # rotational averaging
        center = np.array(out.shape)/2 - 0.5
        
        if rot_ave:
            input_ = out.copy()
            trs0 = np.eye(4, dtype=np.float32)
            trs1 = np.eye(4, dtype=np.float32)
            trs0[:3, 3] = -center
            trs1[:3, 3] = center
            slope = tandg(rise)
            for pf in range(1, npf):
                ang = -2*np.pi*pf/npf
                dy = 2*np.pi*pf/npf*radius*slope/self.scale
                cos = np.cos(ang)
                sin = np.sin(ang)
                rot = np.array([[cos, 0.,-sin, 0.],
                                [ 0., 1.,  0., dy],
                                [sin, 0., cos, 0.],
                                [ 0., 0.,  0., 1.]],
                                dtype=np.float32)
                mtx = trs1 @ rot @ trs0
                out.value[:] += input_.affine(mtx, mode=Mode.grid_wrap)
            
        # stack images for better visualization
        dup = ceilint(y_length/lp)
        outlist = [out]
        if dup > 0:
            for ang in skew_angles[:min(dup, len(skew_angles))-1]:
                outlist.append(out.rotate(-ang, dims="zx", mode=Mode.reflect))
        
        out = np.concatenate(outlist, axis="y")
        
        if erase_corner:
            # This option is needed because padding mode is "grid-wrap".
            z, x = np.indices(out.sizesof("zx"))
            r = min(out.sizesof("zx")) / 2 - 0.5
            corner = (z-center[0])**2 + (x-center[2])**2 > r**2
            sl = np.stack([corner]*out.shape.y, axis=1)
            out.value[sl] = np.median(out.value[~sl])
            
        return out
    
    @batch_process
    def reconstruct_cylindric(
        self, 
        i = None,
        *,
        rot_ave: bool = False, 
        radii: tuple[nm, nm] = None,
        niter: int = 1,
        y_length: nm = 50.0
    ) -> ip.ImgArray:
        """
        3D reconstruction of MT in cylindric coordinate system.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to reconstruct.
        rot_ave : bool, default is False
            If true, rotational averaging is applied to the reconstruction to remove missing wedge.
        y_length : nm, default is 100.0
            Longitudinal length of reconstruction.

        Returns
        -------
        ip.ImgArray
            Reconstructed image.
        """        
        # Cartesian transformation along spline.
        img_open = self.straighten_cylindric(i, radii=radii)
        scale = img_open.scale.y
        total_length: nm = img_open.shape.y*scale
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        lp = pitch * 2
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = roundint(props[H.nPF])
        radius = self.splines[i].radius
        
        # Determine how to split image into tubulin dimer fragments
        dl, resl = divmod(total_length, lp)
        borders = np.linspace(0, total_length - resl, int((total_length - resl)/lp)+1)
        skew_angles = np.arange(borders.size - 1) * skew
        
        # Rotate fragment with skew angle
        imgs: list[ip.ImgArray] = []
        ylen = 99999
    
        # Split image into dimers along y-direction
        for start, stop, ang in zip(borders[:-1], borders[1:], skew_angles):
            start = self.nm2pixel(start)
            stop = self.nm2pixel(stop)
            shift = -ang/360*img_open.shape.a
            imgs.append(
                img_open[:, start:stop].affine(translation=[0, shift],
                                               dims="ya", mode=Mode.grid_wrap,
                                               order=3)
                )
            ylen = min(ylen, stop-start)
        
        # Make image sizes same and prepare FT images.
        imgs = [img[f"y=:{ylen}"] for img in imgs]
        ft_imgs = [img.fft(dims="rya") for img in imgs]
        
        # align each fragment
        ref = imgs[0]
        shape = ref.shape
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        imgs_aligned: list[ip.ImgArray] = []
        for _ in range(niter):
            ft_ref = ref.fft(dims="rya")
            imgs_aligned.clear()
            for k in range(len(imgs)):
                img = imgs[k]
                ft_img = ft_imgs[k]
                shift = ip.ft_pcc_maximum(ft_img, ft_ref, mask=mask)
                imgs_aligned.append(
                    img.affine(translation=shift, 
                               dims="rya", mode=Mode.grid_wrap,
                               order=3)
                    )

            out: ip.ImgArray = np.stack(imgs_aligned, axis="p").proj("p")
            ref = out
        
        # rotational averaging
        if rot_ave:
            input_ = out.copy()
            a_size = out.shape.a
            slope = tandg(rise)
            for pf in range(1, npf):
                dy = 2*np.pi*pf/npf*radius*slope/self.scale
                shift_a = a_size/npf*pf
                shift = [dy, shift_a]
                
                rot_input = input_.affine(translation=shift, 
                                          dims="ya", mode=Mode.grid_wrap)
                out.value[:] += rot_input
            
        # stack images for better visualization
        dup = ceilint(y_length/lp)
        outlist = [out]
        if dup > 0:
            for ang in skew_angles[:min(dup, len(skew_angles))-1]:
                shift = ang/360*img_open.shape.a
                outlist.append(out.affine(translation=[0, -shift], 
                                          dims="ya", mode=Mode.grid_wrap))
    
        return np.concatenate(outlist, axis="y")
    
    @batch_process
    def map_centers(
        self, 
        i = None,
        *, 
        interval: nm | None = None,
        length: nm | None = None,
    ) -> Molecules:
        
        spl = self.splines[i]
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
        i = None,
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
            The offset of origin of oblique coordinate system to map monomers. If not given
            this parameter will be determined by cylindric reconstruction.

        Returns
        -------
        Molecules
            Object that represents monomer positions and angles.
        """
        spl = self._splines[i]
        
        if length is None:
            length = spl.length()
            
        # Get structural parameters
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = int(props[H.nPF])
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
            # Calculate reconstruction in cylindric coodinate system
            rec_cyl_3d: ip.ImgArray = self.reconstruct_cylindric(i, rot_ave=True, y_length=0)
            sigma = rec_cyl_3d.shape.y/8
            rec_cyl_3d.value[:] = ndi.gaussian_filter(rec_cyl_3d.value, sigma=sigma, mode=Mode.grid_wrap)
            rec_cyl = rec_cyl_3d.proj("r")
            
            # Find monomer peak
            argpeak = np.argmin if self.light_background else np.argmax
            ymax, amax = np.unravel_index(argpeak(rec_cyl), rec_cyl.shape)
            offsets = [ymax*self.scale, amax/rec_cyl.shape.a*2*np.pi]
        
        mesh = oblique_meshgrid(shape, tilts, intervals, offsets).reshape(-1, 2)
        radius_arr = np.full((mesh.shape[0], 1), radius, dtype=np.float32)
        mesh = np.concatenate([radius_arr, mesh], axis=1)

        return spl.cylindrical_to_molecules(mesh)
    
    @batch_process
    def map_pf_line(self, i = None, *, angle_offset: float = 0) -> Molecules:
        """
        Calculate mapping of protofilament line at an angle offset.
        This function is useful for visualizing seam or protofilament.

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
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        spl = self._splines[i]
        ny = roundint(spl.length()/pitch)
        mono_skew_rad = np.deg2rad(skew) / 2
        
        rcoords = np.full(ny, spl.radius)
        ycoords = np.arange(ny) * pitch
        acoords = np.arange(ny) * mono_skew_rad + np.deg2rad(angle_offset)
        coords = np.stack([rcoords, ycoords, acoords], axis=1)
        return spl.cylindrical_to_molecules(coords)
    
    def get_subtomogram_loader(
        self,
        mole: Molecules,
        shape: tuple[nm, nm, nm], 
        chunksize: int = 128,
    ) -> SubtomogramLoader:
        """Create a subtomogram loader from molecules."""
        output_shape = tuple(self.nm2pixel(shape))
        return SubtomogramLoader(self.image, mole, output_shape=output_shape, chunksize=chunksize)
    
    # @batch_process
    # def fine_reconstruction(
    #     self, 
    #     i = None, 
    #     *, 
    #     mole: Molecules,
    #     template: ip.ImgArray = None, 
    #     mask: ip.ImgArray = None
    # ) -> tuple[ip.ImgArray, Molecules]:
    #     from .averaging import SubtomogramSampler
    #     spl = self.splines[i]
    #     mole = spl.cylindrical_to_molecules()
        
    #     aligned_mole = averaging.align_subtomograms(
    #         self.image, mole, template=template, mask=mask, scale=self.scale
    #     )
        
    #     subtomo = averaging.get_subtomograms(
    #         self.image, aligned_mole, template.shape, self.scale
    #     )
        
    #     averaged_image = np.mean(subtomo, axis="p")
        
    #     return averaged_image, aligned_mole
        


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
    return da.compute(tasks, scheduler=SCHEDULER)[0]


def _local_dft_params(img: ip.ImgArray, radius: nm):
    img = img - img.mean()
    l_circ: nm = 2*np.pi*radius
    npfmin = GVar.nPFmin
    npfmax = GVar.nPFmax
    ylength_nm = img.shape.y * img.scale.y
    y0 = ceilint(ylength_nm/GVar.yPitchMax) - 1
    y1 = max(ceilint(ylength_nm/GVar.yPitchMin), y0+1)
    up_a = 20
    up_y = max(int(1500/ylength_nm), 1)
    npfrange = ceilint(npfmax/2) # The peak of longitudinal periodicity is always in this range. 
    
    power = img.local_power_spectra(key=f"y={y0}:{y1};a={-npfrange}:{npfrange+1}", 
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
    y_factor = abs(radius/y_pitch/npfmin*img.shape.y/4)
    dy_min = ceilint(tandg(GVar.minSkew)*y_factor) - 1
    dy_max = max(ceilint(tandg(GVar.maxSkew)*y_factor), dy_min+1)
    up_a = 20
    up_y = max(int(5400/(img.shape.y*img.scale.y)), 1)
    
    power = img.local_power_spectra(key=f"y={dy_min}:{dy_max};a={npfmin}:{npfmax}", 
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

def ft_params(img: ip.LazyImgArray, coords: np.ndarray, radius: nm):
    polar = map_coordinates(img, coords, order=3, mode=Mode.constant, cval=np.mean)
    polar = ip.asarray(polar, axes="rya", dtype=np.float32) # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
    polar.scale_unit = img.scale_unit
    return _local_dft_params(polar, radius)


lazy_ft_params = delayed(ft_params)
lazy_ft_pcc = delayed(ip.ft_pcc_maximum)


def _affine(img, matrix, mode: str, cval: float, order):
    out = ndi.affine_transform(img[0], matrix[0,:,:,0], mode=mode, cval=cval, 
                               order=order, prefilter=order>1)
    return out[np.newaxis]


def dask_affine(images, matrices, mode: str = Mode.constant, cval: float = 0, order=1):
    imgs = da.from_array(images, chunks=(1,)+images.shape[1:])
    mtxs = da.from_array(matrices[..., np.newaxis], chunks=(1,)+matrices.shape[1:]+(1,))
    return imgs.map_blocks(_affine,
                           mtxs,
                           mode=mode,
                           cval=cval,
                           order=order,
                           meta=np.array([], dtype=np.float32),
                           ).compute()
