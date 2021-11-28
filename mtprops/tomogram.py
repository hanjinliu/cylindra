from __future__ import annotations
from typing import Callable, Iterable
import json
from collections import namedtuple
from functools import partial, wraps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi
from dask import array as da, delayed

import impy as ip
from .const import nm, H, K, Ori, Mode, CacheKey, GVar
from .spline import Spline3D
from .cache import ArrayCacheMap
from .utils import (load_a_subtomogram, centroid, map_coordinates, roundint, load_rot_subtomograms,
                    ceilint, oblique_meshgrid, no_verbose, mirror_pcc)

cachemap = ArrayCacheMap(maxgb=ip.Const["MAX_GB"])
LOCALPROPS = [H.splPosition, H.splDistance, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]
Coordinates = namedtuple("Coordinates", ["world", "spline"])

if ip.Const["RESOURCE"] == "cupy":
    SCHEDULER = "single-threaded"
else:
    SCHEDULER = "threads"
    

def batch_process(func):
    @wraps(func)
    def _func(self: MtTomogram, i=None, **kwargs):
        if isinstance(i, int):
            with no_verbose:
                out = func(self, i=i, **kwargs)
            return out
        
        # Determine along which spline function will be executed.
        if i is None:
            i_list = range(self.n_paths)
        elif not hasattr(i, "__iter__"):
            raise TypeError("'i' must be int or iterable of int if specified")
        else:
            i_list = []
            for i_ in i:
                if -self.n_paths <= i_ < 0:
                    i_list.append(i_ + self.n_paths)
                elif 0 <= i_ < self.n_paths:
                    i_list.append(i_)
                else:
                    raise ValueError(f"Index {i_} is out of bound")
            
            if len(i_list) > len(set(i_list)):
                raise ValueError("Indices cannot contain duplicated values.")
        
        # Run function along each spline
        out = []
        with no_verbose:
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

def json_encoder(obj):
    """
    Enable Enum and DataFrame encoding.
    """    
    if isinstance(obj, Ori):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")

class MtSpline(Spline3D):
    """
    A spline object with info related to MT.
    """    
    _local_properties = ["localprops"]
    _global_properties = ["globalprops", "radius"]
    
    def __init__(self, scale: float = 1.0, k: int = 3):
        """
        Spline object for MT.
        
        Parameters
        ----------
        scale : float, default is 1.0
            Pixel scale
        k : int, default is 3
            Spline order.
        """        
        super().__init__(scale=scale, k=k)
        self.orientation = Ori.none
        self.radius: nm = None
        self.localprops: pd.DataFrame = None
        self.globalprops: pd.Series = None
        
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
        d[K.localprops] = self.localprops[[l for l in LOCALPROPS if l in self.localprops.columns]]
        d[K.globalprops] = self.globalprops
        return d
        
    @classmethod
    def from_dict(cls, d: dict):
        self = super().from_dict(d)
        localprops = d.get(K.localprops, None)
        if localprops is not None and H.splPosition in localprops:
            self.anchors = localprops[H.splPosition]
        self.globalprops = pd.Series(d.get(K.globalprops, None))
        self.radius = d.get(K.radius, None)
        self.orientation = d.get(K.orientation, Ori.none)
        self.localprops = pd.DataFrame(localprops)
        return self

class MtTomogram:
    """
    Tomogram object. It always connected to a 3D image but processed lazily. Thus
    you can create a lot of MtTomogram objects without MemoryError. Subtomograms
    are temporarily loaded into memory via cache map. Once memory usage exceed
    certain amount, the subtomogram cache will automatically deleted from the old
    ones.
    """    
    def __init__(self, *,
                 subtomogram_length: nm = 48.0,
                 subtomogram_width: nm = 40.0,
                 ft_size: nm = 33.4,
                 light_background: bool = True,
                 ):
        self.subtomo_length = subtomogram_length
        self.subtomo_width = subtomogram_width
        self._paths: list[MtSpline] = []
        self._ft_size = None
        self.ft_size = ft_size
        self.light_background = light_background
        self.metadata = {}
        
    
    def __hash__(self) -> int:
        return id(self)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} of '{self._image.name}'"
            
    @property
    def paths(self) -> list[MtSpline]:
        return self._paths
    
    @property
    def ft_size(self) -> nm:
        """
        Length of local Fourier transformation window size in nm.
        """        
        return self._ft_size
    
    @ft_size.setter
    def ft_size(self, value: nm):
        if not np.isscalar(value):
            raise ValueError("'ft_size' must be a scalar.")
        
        is_same = self._ft_size and (self._ft_size == value)
        self._ft_size = value
        
        # Delete outdated local properties
        if not is_same:
            for spl in self._paths:
                spl.localprops = None
    
    @property
    def n_paths(self) -> int:
        """
        Number of spline paths.
        """        
        return len(self._paths)
    
    @property
    def image(self) -> ip.LazyImgArray:
        """
        Tomogram image data.
        """        
        return self._image
    
    @image.setter
    def image(self, image):
        if not isinstance(image, ip.LazyImgArray):
            raise TypeError(f"Type {type(image)} not supported.")
        self._image = image
        
        if (abs(image.scale.z - image.scale.x) > 1e-3 
            or abs(image.scale.z - image.scale.y) > 1e-3):
            raise ValueError("Uneven scale.")
        self.scale = image.scale.x
        return None
    
    def export_results(self, file_path: str, **kwargs):
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
    
    def save(self, path: str):
        """
        Save splines with its local properties as a json file.

        Parameters
        ----------
        path : str
            File path to save file.
        """        
        path = str(path)
        
        all_results = {}
        for i, spl in enumerate(self._paths):
            spl_dict = spl.to_dict()
            all_results[i] = spl_dict
        
        from .__init__ import __version__
        metadata = self.metadata.copy()
        metadata["light_background"] = self.light_background
        metadata["ft_size"] = self.ft_size
        metadata["version"] = __version__
        all_results["metadata"] = metadata

        with open(path, mode="w") as f:
            json.dump(all_results, f, indent=4, separators=(",", ": "), default=json_encoder)
        return None
    
    def load(self, file_path: str) -> MtTomogram:
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
                int(i)
            except:
                setattr(self, i, d)
            else:
                self._paths.append(
                    MtSpline.from_dict(d)
                )
            
        return self

    def argpeak(self, x: np.ndarray) -> Callable:
        """
        Dispatch argmin and argmax according to background.
        """        
        if self.light_background:
            return np.argmin(x)
        else:
            return np.argmax(x)
    
    def add_spline(self, coords: np.ndarray):
        """
        Add MtSpline path to tomogram.

        Parameters
        ----------
        coords : np.ndarray
            (N, 3) array of coordinates. A spline curve that fit it well is added.
        """        
        spl = MtSpline(self.scale, k=GVar.splOrder)
        sqsum = GVar.splError**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        interval: nm = 30.0
        length = spl.length()
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        if coords.shape[0] <= GVar.splOrder and coords.shape[0] < fit.shape[0]:
            return self.add_spline(fit)
        
        self._paths.append(spl)
        return None
    
    def nm2pixel(self, value: Iterable[nm]|nm) -> np.ndarray|int:
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
    def make_anchors(self, i = None, interval: nm = None, n: int = None, max_interval: nm = None):
        """
        Make anchors on MtSpline object(s).

        Parameters
        ----------
        interval : nm
            Anchor intervals.
        """        
        if interval is None and n is None:
            interval = 24.0
        self._paths[i].make_anchors(interval=interval, n=n, max_interval=max_interval)
        return None
    
    def collect_anchor_coords(self, i: int|Iterable[int] = None) -> np.ndarray:
        """
        Collect all the anchor coordinates into a single np.ndarray.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to collect.

        Returns
        -------
        np.ndarray
            Coordinates in shape (N, 3).
        """        
        if i is None:
            i = range(self.n_paths)
        elif isinstance(i, int):
            i = [i]
        return np.concatenate([self._paths[i_]() for i_ in i], axis=0)
    
    def collect_localprops(self, i: int|Iterable[int] = None) -> pd.DataFrame:
        """
        Collect all the local properties into a single pd.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to collect.

        Returns
        -------
        pd.DataFrame
            Concatenated data frame.
        """        
        if i is None:
            i = range(self.n_paths)
        elif isinstance(i, int):
            i = [i]
        df = pd.concat([self._paths[i_].localprops for i_ in i], 
                        keys=list(i)
                       )
        
        df.index = df.index.rename(["SplineID", "PosID"])
        return df
    
    def plot_localprops(self, i: int|Iterable[int] = None,
                        x=None, y=None, hue=None, **kwargs):
        """
        Simple plot function for visualizing local properties.
        """        
        import seaborn as sns
        df = self.collect_localprops(i)
        data = df.reset_index()
        return sns.swarmplot(x=x, y=y, hue=hue, data=data, **kwargs)
    
    def summerize_localprops(self, i: int|Iterable[int] = None, 
                             by: str | list[str] = "SplineID", 
                             functions: Callable|list[Callable] = None) -> pd.DataFrame:
        """
        Simple summerize of local properties.
        """
        df = self.collect_localprops(i).reset_index()
        if functions is None:
            def se(x): return np.std(x)/np.sqrt(len(x))
            def n(x): return len(x)
            functions = [np.mean, np.std, se, n]
            
        return df.groupby(by=by).agg(functions)
    
    def collect_radii(self, i: int|Iterable[int] = None) -> np.ndarray:
        """
        Collect all the radius into a single array.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to collect.

        Returns
        -------
        np.ndarray
            Radius of each spline
        """        
        if i is None:
            i = range(self.n_paths)
        elif isinstance(i, int):
            i = [i]
        return np.array([self._paths[i_].radius for i_ in i])
    
    def _sample_subtomograms(self, 
                             i: int,
                             rotate: bool = True) -> ip.ImgArray:
        spl = self._paths[i]
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
    def fit(self, 
            i = None,
            *, 
            max_interval: nm = 30.0,
            degree_precision: float = 0.2,
            cutoff_freq: float = 0.0,
            dense_mode: bool = False,
            ) -> MtTomogram:
        """
        Fit i-th path to MT.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to fit.
        max_interval : nm, default is 24.0
            Maximum interval of sampling points in nm unit.
        degree_precision : float, default is 0.2
            Precision of MT xy-tilt degree in angular correlation.
        cutoff_freq : float, default is 0.0
            Cutoff frequency of Butterworth low-pass prefilter.
        dense_mode : bool, default is False
            If True, fitting will be executed in the dense-mocrotubule mode.

        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """        
        spl = self._paths[i]
        spl.make_anchors(max_interval=max_interval)
        npoints = len(spl)
        interval = spl.length()/(npoints-1)
        subtomograms = self._sample_subtomograms(i, rotate=False)
        
        # Apply low-pass filter if needed
        if cutoff_freq > 0:
            subtomograms = subtomograms.lowpass_filter(cutoff_freq)
        
        if dense_mode:
            # roughly crop MT and fill outside with a constant value
            yy, xx = np.indices(subtomograms.sizesof("yx"))
            yc, xc = np.array(subtomograms.sizesof("yx"))/2 - 0.5
            cval = np.mean(subtomograms)
            for i, ds in enumerate(spl(der=1)):
                _, vy, vx = ds
                const = xc*vy - yc*vx
                distance = np.abs(-xx*vy + yy*vx + const)/np.sqrt(vx**2 + vy**2) * self.scale
                sl = np.stack([distance > self.subtomo_width/2]*subtomograms.shape.z, 
                              axis=0)
                subtomograms[i][sl] = cval
        
        ds = spl(der=1)
        yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
        nrots = roundint(14/degree_precision) + 1

        # Angular correlation
        out = dask_angle_corr(subtomograms[1:-1], yx_tilt, nrots=nrots)
        refined_tilt = np.array([0] + list(out) + [0])
        size = 2*roundint(48.0/interval) + 1
        if size > 1:
            # Mirror-mode padding is "a b c d | c b a", thus edge values will be substituted
            # with the adjucent values respectively.
            refined_tilt = ndi.median_filter(refined_tilt, size=size, mode=Mode.mirror)
        
        # Rotate subtomograms            
        for i, img in enumerate(subtomograms):
            angle = refined_tilt[i]
            img.rotate(-angle, cval=np.mean(img), update=True)
            
        # zx-shift correction by self-PCC
        subtomo_proj = subtomograms.proj("y")
        
        if dense_mode:
            xc = int(subtomo_proj.shape.x//2)
            w = int((self.subtomo_width/self.scale)//2)
            subtomo_proj = subtomo_proj[f"x={xc-w}:{xc+w+1}"]

        shape = subtomo_proj[0].shape
        shifts = np.zeros((npoints, 2)) # zx-shift
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        for i in range(npoints):
            img = subtomo_proj[i]
            shifts[i] = mirror_pcc(img, mask=mask)/2
        
        # Update spline coordinates.
        # Because centers of subtomogram are on lattice points of pixel coordinate,
        # coordinates that will be shifted should be converted to integers. 
        coords_px = self.nm2pixel(spl()).astype(np.float32)
        for i in range(npoints):
            shiftz, shiftx = shifts[i]
            shift = np.array([shiftz, 0, shiftx], dtype=np.float32)
            rad = -np.deg2rad(refined_tilt[i])
            cos, sin = np.cos(rad), np.sin(rad)
            shift = shift @ [[1.,   0.,  0.],
                             [0.,  cos, sin],
                             [0., -sin, cos]]
            coords_px[i] += shift
            
        coords = coords_px * self.scale
        
        # Update spline parameters
        sqsum = GVar.splError**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        
        return self
    
    @batch_process
    def refine(self, 
               i = None,
               *, 
               max_interval: nm = 30.0,
               cutoff_freq: float = 0.0
               ) -> MtTomogram:
        """
        Refine spline using the result of previous fit and the global structural parameters.
        During refinement, Y-projection of MT XZ cross section is rotated with the skew angle,
        thus is much more precise than the coarse fitting.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to fit.
        max_interval : nm, default is 24.0
            Maximum interval of sampling points in nm unit.
        cutoff_freq : float, default is 0.0
            Cutoff frequency of butterworth low-pass prefilter.
        
        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """        
        
        spl = self.paths[i]
        if spl.radius is None:
            spl.make_anchors(n=3)
            self.measure_radius(i=i)
        props = self.global_ft_params(i)
        spl.make_anchors(max_interval=max_interval)
        npoints = len(spl)
        interval = spl.length()/(npoints-1)
        subtomograms = self._sample_subtomograms(i)
        if cutoff_freq > 0:
            subtomograms = subtomograms.lowpass_filter(cutoff_freq)
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
        # Skew angles are divided by the angle of single protofilament and the residual
        # angles are used, considering missing wedge effect.
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        npf = roundint(props[H.nPF])
        skew_angles = np.arange(npoints) * interval/lp * skew
        skew_angles %= 360/npf
        skew_angles[skew_angles > 360/npf/2] -= 360/npf

        # Rotate subtomograms at skew angles. All the subtomograms should look "similar"
        # after this rotation.
        subtomo_proj = subtomograms.proj("y")["x=::-1"]
        cval = np.median(subtomo_proj)
        
        imgs_rot_list: list[ip.ImgArray] = []
        for i, ang in enumerate(skew_angles):
            rotimg = subtomo_proj[i].rotate(ang, dims="zx", mode=Mode.constant, cval=cval)
            imgs_rot_list.append(rotimg)
            
        imgs_rot: ip.ImgArray = np.stack(imgs_rot_list, axis="p")
        imgs_rot_ft = imgs_rot.fft(dims="zx")

        # Coarsely align skew-corrected images
        iref = npoints//2
        ft_ref = imgs_rot_ft[iref]
        shape = ft_ref.shape
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape) # mask for PCC
        imgs_aligned = []
        for i in range(npoints):
            img: ip.ImgArray = imgs_rot[i]
            ft = imgs_rot_ft[i]
            shift = ip.ft_pcc_maximum(ft_ref, ft, mask=mask)
            imgs_aligned.append(img.affine(translation=-shift, mode=Mode.constant, cval=cval))
            
        # Make template using coarse aligned images.
        imgcory: ip.ImgArray = np.stack(imgs_aligned, axis="y").proj("y")
        center_shift = mirror_pcc(imgcory, mask=mask)/2
        template = imgcory.affine(translation=center_shift, mode=Mode.constant, cval=cval)
        template_ft = template.fft(dims="zx")
                
        # Align skew-corrected images to the template
        shifts = np.zeros((npoints, 2))
        for i in range(npoints):
            ft = imgs_rot_ft[i]
            shift = -ip.ft_pcc_maximum(template_ft, ft, mask=mask)
            rad = np.deg2rad(skew_angles[i])
            cos, sin = np.cos(rad), np.sin(rad)
            zxrot = np.array([[ cos, sin],
                              [-sin, cos]], dtype=np.float32)
            shifts[i] = shift @ zxrot
        
        # Update spline parameters
        sqsum = GVar.splError**2 * npoints # unit: nm^2
        spl.shift_fit(shifts=shifts, s=sqsum)
        return self
    
    @batch_process
    def fine_fit(self, i = None, *, max_interval: nm = 30.0) -> MtTomogram:
        """
        Fit i-th path to MT, using y=0, theta=1 of Fourier space.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to fit.
        max_interval : nm, default is 24.0
            Maximum interval of sampling points in nm unit.
            
        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """        
        spl = self.paths[i]
        if spl.radius is None:
            spl.make_anchors(n=3)
            self.measure_radius(i=i)
        spl.make_anchors(max_interval=max_interval)
        npoints = len(spl)
        subtomograms = self._sample_subtomograms(i)
        subproj = subtomograms.proj("y")["x=::-1"] # axes = pzx
        offset = np.array(subproj.sizesof("zx"))/2 - 0.5
        
        from .spline import _polar_coords_2d
        from scipy.optimize import minimize
        
        def _func(center, img):
            center = center + offset
            center = center[::-1] # (z, x) -> (x, z)
            map_ = _polar_coords_2d(spl.radius*0.8, spl.radius*1.2, center)
            map_ = np.moveaxis(map_, -1, 0)
            
            out = ndi.map_coordinates(img, map_, prefilter=True, mode=Mode.reflect)
            out = ip.asarray(out, axes="ra")
            pw = out.local_power_spectra("a=0:2", dims="ra").proj("r")
            return pw[1]/pw[0]
        
        shifts = np.zeros((npoints, 2))
        for i in range(npoints):
            img = subproj[i]
            res = minimize(_func, np.array([0., 0.]), args=(img,))
            shift = res.x
            shifts[i] = shift
        self._shifts = shifts
        # Update spline parameters
        sqsum = GVar.splError**2 * npoints # unit: nm^2
        spl.shift_fit(shifts=shifts, s=sqsum)
        return self
                
    @batch_process
    def get_subtomograms(self, i = None) -> ip.ImgArray:
        """
        Get subtomograms at anchors. All the subtomograms are rotated to oriented
        to the spline.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to load samples.

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
            Path ID that you want to measure.

        Returns
        -------
        float (nm)
            MT radius.
        """        
        if self.paths[i]._anchors is None:
            self.paths[i].make_anchors(n=3)
            
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
        r_peak_sub = (imax_sub+0.5)/nbin*r_max
        
        self._paths[i].radius = r_peak_sub
        return r_peak_sub
    
    @batch_process
    def ft_params(self, i = None) -> pd.DataFrame:
        """
        Calculate MT local structural parameters from cylindrical Fourier space.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to analyze.

        Returns
        -------
        pd.DataFrame
            Local properties.
        """        
        spl = self.paths[i]
        if spl.localprops is not None:
            return spl.localprops
        
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")
        
        ylen = self.nm2pixel(self.ft_size)
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
    def global_ft_params(self, i = None) -> pd.Series:
        """
        Calculate MT global structural parameters from cylindrical Fourier space along 
        spline. This function calls ``straighten`` beforehand, so that Fourier space is 
        distorted if MT is curved.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to analyze.

        Returns
        -------
        pd.DataFrame
            Global properties.
        """        
        spl = self._paths[i]
        if spl.globalprops is not None:
            return spl.globalprops
        
        img_st = self.cylindric_straighten(i)
        results = _local_dft_params(img_st, spl.radius)
        series = pd.Series([], dtype=np.float32)
        series[H.riseAngle] = results[0]
        series[H.yPitch] = results[1]
        series[H.skewAngle] = results[2]
        series[H.nPF] = np.round(results[3])
        series[H.start] = results[4]
        
        spl.globalprops = series
        return series

    @batch_process
    def straighten(self, 
                   i = None, 
                   *,
                   size: nm | tuple[nm, nm] = None,
                   range_: tuple[float, float] = (0.0, 1.0), 
                   chunkwise: bool = True) -> ip.ImgArray:
        """
        MT straightening by building curved coordinate system around splines. Currently
        Cartesian coordinate system and cylindrical coordinate system are supported.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to straighten.
        size : float (nm), optional
            Vertical/horizontal box size.
        range_ : tuple[float, float], default is (0.0, 1.0)
            Range of spline domain.
        chunkwise : bool, default is True
            If True, spline is first split into chunks and all the straightened images are
            concatenated afterward, in order to avoid loading entire image into memory.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """        
        try_cache = size is None and range_ == (0.0, 1.0)
        cache_key = CacheKey.cart_straight
        spl = self.paths[i]
        if try_cache:
            try:
                transformed = cachemap[(self, spl, cache_key)]
            except KeyError:
                pass
            else:
                return transformed
            
        chunk_length: nm = 72.0
        length = self._paths[i].length(nknots=512)
        
        if chunkwise and length > chunk_length:
            transformed = self._chunked_straighten(
                i, length, range_,
                self.straighten, size=size
            )
            
        else:
            if size is None:
                rz = rx = self.nm2pixel(self._paths[i].radius * GVar.outer) * 2 + 1
            
            else:
                rz, rx = self.nm2pixel(size)

            coords = spl.cartesian((rz, rx), s_range=range_)
            coords = np.moveaxis(coords, -1, 0)
            
            transformed = map_coordinates(self.image, coords, order=1)
            
            axes = "zyx"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: self.scale for k in axes})
            transformed.scale_unit = "nm"
        
        if try_cache:
            cachemap[(self, spl, cache_key)] = transformed
        
        return transformed

    @batch_process
    def cylindric_straighten(self, 
                             i = None, 
                             *,
                             radii: tuple[nm, nm] = None,
                             range_: tuple[float, float] = (0.0, 1.0), 
                             chunkwise: bool = True) -> ip.ImgArray:
        """
        MT straightening by building curved coordinate system around splines. Currently
        Cartesian coordinate system and cylindrical coordinate system are supported.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to straighten.
        radii : tuple of float (nm), optional
            Lower/upper limit of radius.
        range_ : tuple[float, float], default is (0.0, 1.0)
            Range of spline domain.
        chunkwise : bool, default is True
            If True, spline is first split into chunks and all the straightened images are
            concatenated afterward, in order to avoid loading entire image into memory.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """        
        try_cache = radii is None and range_ == (0.0, 1.0)
        cache_key = CacheKey.cyl_straight
        spl = self.paths[i]
        
        if spl.radius is None:
            raise ValueError("Radius has not been determined yet.")
        
        if try_cache:
            try:
                transformed = cachemap[(self, spl, cache_key)]
            except KeyError:
                pass
            else:
                return transformed
            
        chunk_length: nm = 72.0
        length = self._paths[i].length(nknots=512)
        
        if chunkwise and length > chunk_length:
            transformed = self._chunked_straighten(
                i, length, range_, 
                function=self.cylindric_straighten,
                radii=radii
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
            cachemap[(self, spl, cache_key)] = transformed
        
        return transformed
    
    def _chunked_straighten(self, i: int, length: nm, range_: tuple[float, float],
                            function: Callable, **kwargs):
        out = []
        current_distance: nm = 0.0
        chunk_length: nm = 72.0
        start, end = range_
        spl = self.paths[i]
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
            img_st = function(i, range_=sub_range, chunkwise=False, **kwargs)
            
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
                    seam_offset: float | str = None,
                    erase_corner: bool = True,
                    niter: int = 1,
                    y_length: nm = 50.0) -> ip.ImgArray:
        """
        3D reconstruction of MT.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to reconstruct.
        rot_ave : bool, default is False
            If true, rotational averaging is applied to the reconstruction to remove missing wedge.
        seam_offset : float or "find", optional
            Angle offset of seam position in degree. If given and rot_ave is True, reconstruction 
            will be executed considering seam.
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
        radius = self.paths[i].radius
        
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
            imgs.append(img_st[:, start:stop].rotate(-ang, dims="zx", mode=Mode.reflect))
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
            slope = -np.tan(np.deg2rad(rise))
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
            
            if seam_offset == "find":
                seam_offset = self.seam_offset(i)
                
            if seam_offset is not None:
                if self.light_background:
                    missing = out.max()
                    extrema = np.minimum
                else:
                    missing = out.min()
                    extrema = np.maximum
                
                z, x = np.indices(out.sizesof("zx"))
                angle = np.rad2deg(np.arctan2(z-center[0], x-center[2]))
                eps = 2.0
                # TODO: do not work when seam_offset ~ 0??
                sl_zx = (seam_offset - 360/npf - eps <= angle) & (angle <= seam_offset + eps)
                sl = np.stack([sl_zx]*out.shape.y, axis=1)
                
                # image only contains one protofilament just before seam.
                base: ip.ImgArray = ip.empty(out.shape, dtype=np.float32, axes="zyx")
                base[:] = missing
                base[sl] = out[sl]
                out[:] = base[:]
                
                for pf in range(1, npf):
                    ang = 2*np.pi*pf/npf
                    dy = -2*np.pi*pf/npf*radius*slope/self.scale
                    cos = np.cos(ang)
                    sin = np.sin(ang)
                    rot = np.array([[cos, 0.,-sin, 0.],
                                    [ 0., 1.,  0., dy],
                                    [sin, 0., cos, 0.],
                                    [ 0., 0.,  0., 1.]],
                                    dtype=np.float32)
                    mtx = trs1 @ rot @ trs0
                    rot_img = base.affine(mtx, mode=Mode.grid_wrap)
                    out.value[:] = extrema(out, rot_img)
                
        # stack images for better visualization
        dup = ceilint(y_length/lp)
        outlist = [out]
        if dup > 0:
            for ang in skew_angles[:min(dup, len(skew_angles))-1]:
                outlist.append(out.rotate(ang, dims="zx", mode=Mode.reflect))
        
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
    def cylindric_reconstruct(self, 
                              i = None,
                              *,
                              rot_ave: bool = False, 
                              seam_offset: float | str = None,
                              radii: tuple[nm, nm] = None,
                              niter: int = 1,
                              y_length: nm = 50.0) -> ip.ImgArray:
        """
        3D reconstruction of MT in cylindric coordinate system.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to reconstruct.
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
        img_open = self.cylindric_straighten(i, radii=radii)
        scale = img_open.scale.y
        total_length: nm = img_open.shape.y*scale
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        lp = pitch * 2
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = roundint(props[H.nPF])
        radius = self.paths[i].radius
        
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
                                               dims="ya", mode=Mode.grid_wrap)
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
                               dims="rya", mode=Mode.grid_wrap)
                    )

            out: ip.ImgArray = np.stack(imgs_aligned, axis="p").proj("p")
            ref = out
        
        # rotational averaging
        if rot_ave:
            input_ = out.copy()
            a_size = out.shape.a
            slope = np.tan(np.deg2rad(rise))
            for pf in range(1, npf):
                dy = 2*np.pi*pf/npf*radius*slope/self.scale
                shift_a = a_size/npf*pf
                shift = [dy, shift_a]
                
                rot_input = input_.affine(translation=shift, 
                                        dims="ya", mode=Mode.grid_wrap)
                out.value[:] += rot_input
            
            if seam_offset == "find":
                seam_offset = self._find_seam_offset(i, input_)
                
            if seam_offset is not None:
                if self.light_background:
                    missing = out.max()
                    extrema = np.minimum
                else:
                    missing = out.min()
                    extrema = np.maximum
                    
                len_pf = out.shape.a/npf
                seam_px = seam_offset/360*a_size
                sl_temp = np.arange(int(seam_px - len_pf), ceilint(seam_px)) % a_size
                
                # image only contains one protofilament just before seam.
                base: ip.ImgArray = ip.empty(out.shape, dtype=np.float32, axes="rya")
                base[:] = missing
                base[:,:,sl_temp] = out[:,:,sl_temp]
                out[:] = base[:]
                
                slope = np.tan(np.deg2rad(rise))
                da = a_size/npf * np.arange(npf)
                dy = 2*np.pi/npf*radius*slope/self.scale * np.arange(npf)
                for pf in range(1, npf):
                    shift = [-dy[pf], -da[pf]]
                    rot_img = base.affine(translation=shift, dims="ya", mode=Mode.grid_wrap)
                    out.value[:] = extrema(out, rot_img)
                    
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
    def map_monomer(self, i = None) -> Coordinates:
        """
        Map coordinates of tubulin monomers in world coordinate.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that mapping will be calculated.

        Returns
        -------
        Coordinates
            Named tuple with monomer positions in world coordinates and spline coordinates.
        """        
        spl = self._paths[i]
        rec_cyl = self.cylindric_reconstruct(i, rot_ave=True, y_length=0)
        rec2d = rec_cyl.proj("r")
        ymax, amax = np.unravel_index(self.argpeak(rec2d), rec2d.shape)
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = int(props[H.nPF])
        radius = spl.radius
        ny = roundint(spl.length()/pitch)
        tan_rise = np.tan(np.deg2rad(rise))
        mesh = oblique_meshgrid((ny, npf), 
                                rise = tan_rise*2*np.pi*radius/npf/pitch,
                                tilt = -np.deg2rad(skew)*npf/(4*np.pi), 
                                offset = (ymax/pitch*self.scale, amax/rec_cyl.shape.a*2*np.pi)
                                ).reshape(-1, 2)
        
        dtheta = 2*np.pi/npf
        mesh = np.concatenate([np.full((mesh.shape[0], 1), radius, dtype=np.float32), mesh], axis=1)
        mesh[:, 1] *= pitch
        mesh[:, 2] *= dtheta
        
        crds = Coordinates(world = spl.inv_cylindrical(coords=mesh),
                           spline = mesh)
        return crds
    
    @batch_process
    def pf_offsets(self, i = None) -> np.ndarray:
        """
        Calculate pixel offsets of protofilament origins.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that offsets will be calculated.

        Returns
        -------
        np.ndarray
            Float angle offsets in degree. If MT has N protofilaments, this array will be (N,).
        """
        # WIP!
        spl = self._paths[i]
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        npf = int(props[H.nPF])
        imgst = self.cylindric_straighten(i).proj("r")
        tilt = -np.deg2rad(skew) * spl.radius / pitch / 2
        with no_verbose:
            img_shear = imgst.affine(np.array([[1, 0, 0],[tilt, 1, 0],[0, 0, 1]]), 
                                     mode=Mode.grid_wrap, dims="ya")
            line = img_shear.proj("y")
        
        if self.light_background:
            line = -line
            
        offset = np.rad2deg(np.angle(line.fft(dims="a", shift=False)[npf]))
        return (np.arange(npf)*360/npf + offset) % 360
    
    def _find_seam_offset(self, i, rec: ip.ImgArray):
        spl = self._paths[i]
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = roundint(props[H.nPF])
        a_size = rec.shape.a
        tilt = -np.deg2rad(skew) * spl.radius / pitch / 2
        with no_verbose:
            mtx = np.array([[1.,   0., 0., 0.],
                            [0.,   1., 0., 0.],
                            [0., tilt, 1., 0.],
                            [0.,   0., 0., 1.]],
                           dtype=np.float32)
            img_shear = rec.affine(mtx, mode=Mode.grid_wrap, dims="rya")
            line = img_shear.proj("ry")
        
        # "offset" means the peak of monomer     
        offset_rad = np.angle(np.fft.fft(line.value)[npf]) % (2 * np.pi)
        offset_px: float = offset_rad / 2 / np.pi * a_size
                
        l = roundint(a_size/npf)
        l_dimer = pitch*2
        slope = np.tan(np.deg2rad(-rise))
        opt_y_mat = np.zeros((npf, npf), dtype=np.float32)
        with no_verbose:
            # TODO: This is not efficient. At least, fft is calculated many times in pcc_maximum
            for i in range(npf):
                for j in range(i, npf):
                    si = roundint(a_size/npf*i - offset_px - l/2)
                    sj = roundint(a_size/npf*j - offset_px - l/2)
                    sl_i = np.arange(si, si+l) % a_size
                    sl_j = np.arange(sj, sj+l) % a_size
                    img0 = img_shear[:, :, sl_i]
                    img1 = img_shear[:, :, sl_j]
                    shift = ip.pcc_maximum(img0, img1)
                    opt_y_mat[i, j] = opt_y_mat[j, i] = (shift[1] - slope*a_size/npf*(j-i)) % l_dimer
        
        std_list: list[float] = []
        for seam in range(npf):
            cl0 = np.cos(opt_y_mat[:seam, :seam]/l_dimer*2*np.pi)
            cl1 = np.cos(opt_y_mat[seam:, seam:]/l_dimer*2*np.pi)
            std_list.append(np.std(np.concatenate([cl0.ravel(), cl1.ravel()])))
        
        seampos = np.argmin(std_list)
        return np.rad2deg(seampos / len(std_list) * 2 * np.pi + offset_rad) % 360
    
    @batch_process
    def seam_offset(self, i = None) -> float:
        """
        Find seam position using cylindrical recunstruction.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that seam offset will be calculated.

        Returns
        -------
        float
            Angle offset of seam in degree.
        """        
        # WIP!
        rec = self.cylindric_reconstruct(i, y_length=0)
        return self._find_seam_offset(i, rec)

    
    @batch_process
    def map_pf_line(self, i = None, angle_offset: float = 0) -> Coordinates:
        """
        Calculate mapping of protofilament line at an angle offset.
        This function is useful for visualizing seam or protofilament.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that mapping will be calculated.
        angle_offset : float, default is 0.0
            Angle offset in degree.

        Returns
        -------
        Coordinates
            World coordinates and spline coordinates of protofilament.
        """        
        props = self.global_ft_params(i)
        pitch = props[H.yPitch]
        skew = props[H.skewAngle]
        spl = self._paths[i]
        ny = roundint(spl.length()/pitch)
        mono_skew_rad = np.deg2rad(skew) / 2
        
        rcoords = np.full(ny, spl.radius)
        ycoords = np.arange(ny) * pitch
        acoords = np.arange(ny) * mono_skew_rad + np.deg2rad(angle_offset)
        coords = np.stack([rcoords, ycoords, acoords], axis=1)
        crds = Coordinates(world = spl.inv_cylindrical(coords=coords),
                           spline = coords)
        return crds
        
def angle_corr(img: ip.ImgArray, ang_center: float = 0, drot: float = 7, nrots: int = 29):
    # img: 3D
    img_z = img.proj("z")
    mask = ip.circular_mask(img_z.shape.y/2+2, img_z.shape)
    img_mirror = img_z["x=::-1"]
    angs = np.linspace(ang_center-drot, ang_center+drot, nrots, endpoint=True)
    corrs = []
    f0 = np.sqrt(img_z.power_spectra(dims="yx", zero_norm=True))
    for ang in angs:
        f1 = np.sqrt(img_mirror.rotate(ang*2).power_spectra(dims="yx", zero_norm=True))
        corr = ip.zncc(f0, f1, mask)
        corrs.append(corr)
        
    angle = angs[np.argmax(corrs)]
    return angle

def dask_angle_corr(imgs, ang_centers, drot: float=7, nrots: int = 29):
    _angle_corr = delayed(partial(angle_corr, drot=drot, nrots=nrots))
    tasks = []
    for img, ang in zip(imgs, ang_centers):
        tasks.append(da.from_delayed(_angle_corr(img, ang), shape=(), dtype=np.float32))
    return da.compute(tasks, scheduler=SCHEDULER)[0]
    
def _local_dft_params(img: ip.ImgArray, radius: nm):
    l_circ: nm = 2*np.pi*radius
    npfmin = GVar.nPFmin
    npfmax = GVar.nPFmax
    ylength_nm = img.shape.y*img.scale.y
    y0 = ceilint(ylength_nm/GVar.yPitchMax) - 1
    y1 = max(ceilint(ylength_nm/GVar.yPitchMin), y0+1)
    up_a = 20
    up_y = max(int(1500/ylength_nm), 1)
    npfrange = ceilint(npfmax/2) # The peak of longitudinal periodicity is always in this range. 
    
    power = img.local_power_spectra(key=f"y={y0}:{y1};a={-npfrange}:{npfrange+1}", 
                                    upsample_factor=[1, up_y, up_a], 
                                    dims="rya"
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
    dy0 = ceilint(np.tan(np.deg2rad(-GVar.maxSkew))*y_factor) - 1
    dy1 = max(ceilint(np.tan(np.deg2rad(-GVar.minSkew))*y_factor), dy0+1)
    up_a = 20
    up_y = max(int(5400/(img.shape.y*img.scale.y)), 1)
    
    power = img.local_power_spectra(key=f"y={dy0}:{dy1};a={npfmin}:{npfmax}", 
                                    upsample_factor=[1, up_y, up_a], 
                                    dims="rya"
                                    ).proj("r")
    
    ymax, amax = np.unravel_index(np.argmax(power), shape=power.shape)
    amaxp = np.argmax(power.proj("y"))
    
    amax_f = amax + npfmin*up_a
    amaxp_f = amaxp + npfmin*up_a
    ymax_f = ymax + dy0*up_y
    a_freq = np.fft.fftfreq(img.shape.a*up_a)
    y_freq = np.fft.fftfreq(img.shape.y*up_y)
    
    skew = np.arctan(y_freq[ymax_f]/a_freq[amax_f]*2*y_pitch/radius)
    start = l_circ/y_pitch/(np.tan(skew) + 1/np.tan(rise))
    
    return np.array([np.rad2deg(rise), 
                     y_pitch, 
                     np.rad2deg(skew), 
                     amaxp_f/up_a,
                     abs(start)], 
                    dtype=np.float32)
    

def ft_params(img: ip.ImgArray, coords: np.ndarray, radius: nm):
    polar = map_coordinates(img, coords, order=3, mode=Mode.grid_wrap)
    polar = ip.asarray(polar, axes="rya") # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
    polar.scale_unit = img.scale_unit
    return _local_dft_params(polar, radius)

lazy_ft_params = delayed(ft_params)
lazy_ft_pcc = delayed(ip.ft_pcc_maximum)

def _affine(img, matrix, order=1):
    out = ndi.affine_transform(img[0], matrix[0,:,:,0], order=order, prefilter=order>1)
    return out[np.newaxis]

def dask_affine(images, matrices, order=1):
    imgs = da.from_array(images, chunks=(1,)+images.shape[1:])
    mtxs = da.from_array(matrices[..., np.newaxis], chunks=(1,)+matrices.shape[1:]+(1,))
    return imgs.map_blocks(_affine, 
                           mtxs,
                           order=order,
                           meta=np.array([], dtype=np.float32),
                           ).compute()
