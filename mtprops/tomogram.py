from __future__ import annotations
from typing import Iterable
import json
import matplotlib.pyplot as plt
from collections import namedtuple
from functools import partial, wraps
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from dask import array as da, delayed

from ._dependencies import impy as ip
from .const import nm, H, Ori, CacheKey, GVar
from .spline import Spline3D
from .cache import ArrayCacheMap
from .utils import (load_a_subtomogram, centroid, rotational_average, map_coordinates, roundint, 
                    ceilint, oblique_meshgrid)

cachemap = ArrayCacheMap(maxgb=ip.Const["MAX_GB"])
LOCALPROPS = [H.splPosition, H.splDistance, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]
Coordinates = namedtuple("Coordinates", ["world", "spline"])

if ip.Const["RESOURCE"] == "cupy":
    SCHEDULER = "single-threaded"
else:
    SCHEDULER = "threads"
    

def batch_process(func):
    # TODO: error handling
    @wraps(func)
    def _func(self: MtTomogram, i=None, **kwargs):
        if isinstance(i, int):
            with ip.SetConst("SHOW_PROGRESS", False):
                out = func(self, i=i, **kwargs)
            return out
        
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
        
        with ip.SetConst("SHOW_PROGRESS", False):
            out = [func(self, i=i_, **kwargs) for i_ in i_list]
        
        return out
    return _func  

def json_encoder(obj):
    if isinstance(obj, Ori):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")

class MtSpline(Spline3D):
    """
    A spline object with info related to MT.
    """    
    def __init__(self, scale:float=1, k=3):
        super().__init__(scale=scale, k=k)
        self.orientation = Ori.none
        self._init_properties()
    
    def _init_properties(self):
        self._radius: nm = None
        self.localprops: pd.DataFrame = pd.DataFrame([])
        
    @property
    def orientation(self) -> Ori:
        return self._orientation
    
    @orientation.setter
    def orientation(self, value: Ori | str):
        try:
            self._orientation = Ori(value)
        except ValueError:
            self._orientation = Ori.none
    
    @property
    def radius(self) -> nm:
        return self._radius
    
    @radius.setter
    def radius(self, value: nm):
        try:
            self._radius = float(value)
        except ValueError:
            raise ValueError(f"Cannot set {type(value)} to radius.")
        else:
            self._updates += 1
    
    def fit(self, coords:np.ndarray, w=None, s=None):
        super().fit(coords, w=w, s=s)
        self._init_properties()
        return self
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["radius"] = self.radius
        d["orientation"] = self.orientation.name
        d["localprops"] = self.localprops[LOCALPROPS]
        return d
        
    @classmethod
    def from_dict(cls, d: dict):
        self = super().from_dict(d)
        self.radius = d.get("radius", None)
        self.orientation = d.get("orientation", Ori.none)
        self.localprops = pd.DataFrame(d.get("localprops", None))
        if H.splPosition in self.localprops.columns:
            self.anchors = self.localprops[H.splPosition]
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
                 box_size: tuple[nm, nm, nm] = (44.0, 56.0, 56.0), 
                 ft_size: nm = 33.4,
                 light_background: bool = True,
                 ):
        
        self.box_size = box_size
        self.ft_size = ft_size
        self.light_background = light_background
        self.metadata = {}
        
        self._paths: list[MtSpline] = []
    
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
    def box_size(self) -> tuple[nm, nm, nm]:
        return self._box_radius
    
    @box_size.setter
    def box_size(self, value: nm|Iterable[nm]):
        if np.isscalar(value):
            value = (value,)*3
        elif len(value) != 3:
            raise ValueError("'box_radius' must be composed of 1 or 3 scalars.")
        self._box_radius = tuple(value)
    
    @property
    def ft_size(self) -> nm:
        return self._ft_size
    
    @ft_size.setter
    def ft_size(self, value: nm):
        if not np.isscalar(value):
            raise ValueError("'ft_size' must be a scalar.")
        self._ft_size = value
    
    @property
    def n_paths(self) -> int:
        return len(self._paths)
    
    @property
    def image(self) -> ip.arrays.LazyImgArray:
        return self._image
    
    @image.setter
    def image(self, image):
        if not isinstance(image, ip.arrays.LazyImgArray):
            raise TypeError(f"Type {type(image)} not supported.")
        self._image = image
        
        if abs(image.scale.z - image.scale.x) > 1e-3 or abs(image.scale.z - image.scale.y) > 1e-3:
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
    
    def load(self, file_path :str) -> MtTomogram:
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
    def make_anchors(self, i: int = None, interval: nm = None, n: int = None):
        """
        Make anchors on MtSpline object(s).

        Parameters
        ----------
        interval : nm
            Anchor intervals.
        """        
        if interval is None and n is None:
            interval = 24.0
        self._paths[i].make_anchors(interval=interval, n=n)
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
        df.index.name = ("path", "position")
        return df
    
    def _sample_subtomograms(self, 
                             i: int,
                             rotate: bool = True):
        spl = self._paths[i]
        anchors = spl.anchors
        if rotate:
            try:
                out = cachemap[(self, spl, CacheKey.subtomograms)]
            except KeyError:
                size_px = self.nm2pixel(self.box_size)
                plane_shape = (size_px[0], size_px[2])
                axial_size = size_px[1]
                
                out = []
                for u in spl.anchors:
                    coords = spl.local_cartesian(plane_shape, axial_size, u)
                    coords = np.moveaxis(coords, -1, 0)
                    out.append(map_coordinates(self.image, coords, order=3))
                out = ip.asarray(np.stack(out, axis=0), axes="pzyx")
                out.set_scale(xyz=self.scale)
                
                cachemap[(self, spl, CacheKey.subtomograms)] = out
        else:
            center_px = self.nm2pixel(spl(anchors))
            size_px = self.nm2pixel(self.box_size)
            
            out = np.stack([load_a_subtomogram(self._image, c, size_px, True) 
                            for c in center_px],
                            axis="p")
                
        return out
    
    @batch_process
    def fit(self, 
            i: int = None,
            *, 
            max_interval: nm = 30.0,
            degree_precision: float = 0.2,
            cutoff_freq: float = 0.0,
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
            Cutoff frequency of butterworth low-pass prefilter.

        Returns
        -------
        MtTomogram
            Same object with updated MtSpline objects.
        """        
        
        spl = self._paths[i]
        length = spl.length()
        npoints = max(ceilint(length/max_interval) + 1, 4)
        interval = length/(npoints-1)
        spl.make_anchors(n=npoints)
        subtomograms = self._sample_subtomograms(i, rotate=False)
        if cutoff_freq > 0:
            subtomograms = subtomograms.lowpass_filter(cutoff_freq)
        
        ds = spl(der=1)
        yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
        nrots = roundint(14/degree_precision) + 1

        # Angular correlation
        out = delayed_angle_corr(subtomograms[1:-1], yx_tilt, nrots=nrots)
        refined_tilt = np.array([0] + list(out) + [0])
        size = 2*roundint(48.0/interval) + 1
        if size > 1:
            # Mirror-mode padding is "a b c d | c b a", thus edge values will be substituted
            # with the adjucent values respectively.
            refined_tilt = ndi.median_filter(refined_tilt, size=size, mode="mirror")
        
        # Rotate subtomograms            
        for i, img in enumerate(subtomograms):
            angle = refined_tilt[i]
            img.rotate(-angle, cval=np.mean(img), update=True)
            
        # zx-shift correction by self-PCC
        subtomo_proj = subtomograms.proj("y")
        shape = subtomo_proj[0].shape
        shifts = np.zeros((npoints, 2)) # zx-shift
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        for i in range(npoints):
            img = subtomo_proj[i]
            shifts[i] = ip.pcc_maximum(img[::-1, ::-1], img, mask=mask)/2
        
        # Update spline coordinates.
        coords = spl()
        for i in range(npoints):
            shiftz, shiftx = -shifts[i]
            shift = np.array([shiftz, 0, shiftx])
            rad = -np.deg2rad(refined_tilt[i])
            cos, sin = np.cos(rad), np.sin(rad)
            shift = shift @ [[1.,   0.,  0.],
                             [0.,  cos, sin],
                             [0., -sin, cos]]
            coords[i] += shift * self.scale
        
        # Update spline parameters
        sqsum = GVar.splError**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        
        return self
    
    @batch_process
    def refine(self, 
               i: int = None,
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
        spl = self._paths[i]
        length = spl.length()
        npoints = max(ceilint(length/max_interval) + 1, 4)
        interval = length/(npoints-1)
        spl.make_anchors(n=npoints)
        subtomograms = self._sample_subtomograms(i)
        if cutoff_freq > 0:
            subtomograms = subtomograms.lowpass_filter(cutoff_freq)
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
        props = self.global_ft_params(i)
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        npf = roundint(props[H.nPF])
        skew_angles = np.arange(npoints) * interval/lp * skew
        skew_angles %= 360/npf
        skew_angles[skew_angles > 360/npf/2] -= 360/npf
        
        subtomo_proj = subtomograms.proj("y")
        imgs_rot = []
        for i, ang in enumerate(skew_angles):     
            # rotate 'ang' counter-clockwise           
            rotimg = subtomo_proj[i].rotate(-ang, dims="zx", mode="reflect")
            imgs_rot.append(rotimg)

        # Align skew-corrected images
        iref = npoints//2
        imgref = imgs_rot[iref]
        shifts = np.zeros((npoints, 2)) # zx-shift
        shape = imgref.shape
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape) # mask for PCC
        imgs_aligned = []
        for i in range(npoints):
            img = imgs_rot[i]
            if i != iref:
                shifts[i] = ip.pcc_maximum(imgref, img, mask=mask)
            else:
                shifts[i] = np.array([0, 0])
                
            imgs_aligned.append(img.affine(translation=-shifts[i]))
        
        # Make template
        imgcory = np.stack(imgs_aligned, axis="y").proj("y")
        center_shift = ip.pcc_maximum(imgcory, imgcory[::-1, ::-1], mask=mask)
        template = imgcory.affine(translation=center_shift/2, mode="reflect")
        
        # Align skew-corrected images to the template
        for i in range(npoints):
            img = imgs_rot[i]
            shifts[i] = ip.pcc_maximum(template, img, mask=mask)

        coords = spl()
        for i in range(npoints):
            shiftz, shiftx = -shifts[i]
            shift = np.array([shiftz, 0, shiftx])
            rad = np.deg2rad(skew_angles[i])
            cos, sin = np.cos(rad), np.sin(rad)
            
            # rotate by 'rad' clockwise
            zxrot = np.array([[cos,  0., sin],
                              [0.,   1.,  0.],
                              [-sin, 0., cos]])
                        
            mtx = spl.rotation_matrix(spl.anchors[i])[:3, :3]
            coords[i] += shift @ zxrot @ mtx * self.scale
        
        # Update spline parameters
        sqsum = GVar.splError**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        return self
                
                
    @batch_process
    def get_subtomograms(self, i: int = None) -> ip.arrays.ImgArray:
        """
        Get subtomograms at anchors. All the subtomograms are rotated to oriented
        to the spline.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to load samples.

        Returns
        -------
        ip.arrays.ImgArray
            Subtomograms stacked along "p" axis.
        """        
        subtomograms = self._sample_subtomograms(i)
        return subtomograms
    
    @batch_process
    def measure_radius(self, i: int = None) -> nm:
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
        subtomograms = self._sample_subtomograms(i)
        nbin = 17
        r_max: nm = GVar.rMax
        img2d = subtomograms.proj("py")
        prof = img2d.radial_profile(nbin=nbin, r_max=r_max)
        if self.light_background:
            prof = -prof
            
        imax = np.argmax(prof)
        r_peak_sub = centroid(prof, imax-5, imax+5)/nbin*r_max
        self._paths[i].radius = r_peak_sub
        return r_peak_sub
    
    @batch_process
    def ft_params(self, i: int = None) -> pd.DataFrame:
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
        ylen = self.nm2pixel(self.ft_size)
        spl = self._paths[i]
        spl.localprops = pd.DataFrame([])
        rmin = self.nm2pixel(spl.radius*GVar.inner)
        rmax = self.nm2pixel(spl.radius*GVar.outer)
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
                
        spl.localprops[H.splPosition] = spl.anchors
        spl.localprops[H.splDistance] = spl.distances()
        spl.localprops[H.riseAngle] = results[:, 0]
        spl.localprops[H.yPitch] = results[:, 1]
        spl.localprops[H.skewAngle] = results[:, 2]
        spl.localprops[H.nPF] = np.round(results[:, 3]).astype(np.uint8)
        spl.localprops[H.start] = np.round(results[:, 4]).astype(np.uint8)
        
        return spl.localprops
    
    @batch_process
    def global_ft_params(self, i: int = None):
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
        img_st = self.straighten(i, radius=(spl.radius*GVar.inner, spl.radius*GVar.outer), cylindrical=True)
        results = _local_dft_params(img_st, spl.radius)
        series = pd.Series([], dtype=np.float32)
        series[H.riseAngle] = results[0]
        series[H.yPitch] = results[1]
        series[H.skewAngle] = results[2]
        series[H.nPF] = np.round(results[3])
        series[H.start] = np.round(results[4])
        
        return series

    @batch_process
    def straighten(self, 
                   i: int = None, 
                   *,
                   radius: nm | tuple[nm, nm] = None,
                   range_: tuple[float, float] = (0.0, 1.0), 
                   chunkwise: bool = True,
                   cylindrical: bool = False):
        """
        MT straightening by building curved coordinate system around splines. Currently
        Cartesian coordinate system and cylindrical coordinate system are supported.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to straighten.
        radius : float (nm), optional
            Parameter that specify the size of coordinate system. For Cartesian coordinate
            system, this parameter is interpreted as box size. For cylindrical coordinate
            system, this parameter is interpreted as the range of radius.
        range_ : tuple[float, float], default is (0.0, 1.0)
            Range of spline domain.
        chunkwise : bool, default is True
            If True, spline is first split into chunks and all the straightened images are
            concatenated afterward, in order to avoid loading entire image into memory.
        cylindrical : bool, default is False
            If True, cylindrical coordinate system will be used.

        Returns
        -------
        ip.array.ImgArray
            Straightened image. If Cartesian coordinate system is used, it will have "zyx".
        """        
        try_cache = radius is None and range_ == (0.0, 1.0)
        cache_key = CacheKey.cyl_straight if cylindrical else CacheKey.cart_straight
        spl = self.paths[i]
        if try_cache:
            try:
                transformed = cachemap[(self, spl, cache_key)]
            except KeyError:
                pass
            else:
                return transformed
            
        start, end = range_
        chunk_length: nm = 72.0
        length = self._paths[i].length(nknots=512)
        
        if chunkwise and length > chunk_length:
            # split image into chunks and call straightening separately.
            out = []
            current_distance: nm = 0.0
            while current_distance < length:
                start = current_distance/length
                stop = start + chunk_length/length
                if end - start < 1e-3:
                    # Sometimes divmod of floating values generates very small residuals.
                    break
                sub_range = (start, min(stop, end))
                img_st = self.straighten(i, range_=sub_range, radius=radius, 
                                         chunkwise=False, cylindrical=cylindrical)
                
                out.append(img_st)
                
                # We have to sum up real distance instead of start/end, to precisely deal
                # with the borders of chunks
                current_distance += img_st.shape.y * self.scale
            
            # concatenate all the chunks
            transformed = np.concatenate(out, axis="y")
            
        else:
            if radius is None:
                if cylindrical:
                    rz, rx = self.nm2pixel(self._paths[i].radius * np.array([GVar.inner, GVar.outer]))
                else:
                    rz = rx = self.nm2pixel(self._paths[i].radius * GVar.outer) * 2 + 1
            
            else:
                rz, rx = self.nm2pixel(radius)

            if cylindrical:
                if rx <= rz:
                    raise ValueError("For cylindrical straightening, 'radius' must be (rmin, rmax)")
                coords = spl.cylindrical((rz, rx), s_range=range_)
            else:
                coords = spl.cartesian((rz, rx), s_range=range_)
                
            coords = np.moveaxis(coords, -1, 0)
            
            transformed = map_coordinates(self.image, 
                                          coords,
                                          order=1
                                          )
            
            axes = "rya" if cylindrical else "zyx"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: self.scale for k in axes})
            transformed.scale_unit = "nm"
        
        if try_cache:
            cachemap[(self, spl, cache_key)] = transformed
        
        return transformed
    
    @batch_process
    def rotational_average(self, i: int = None) -> ip.arrays.ImgArray:
        """
        2D rotational averaging of MT at XZ-section.

        Parameters
        ----------
        i : int or iterable of int, optional
            Path ID that you want to execute averaging.

        Returns
        -------
        ip.arrays.ImgArray
            Averaged images. They will have "pzx" axes.
        """        
        subtomograms = self._sample_subtomograms(i)
        projs = subtomograms.proj("y") # pzx
        spl = self._paths[i]
        out = np.empty_like(projs)
        for p, (img, npf) in enumerate(zip(projs, spl.localprops[H.nPF])):
            out.value[p] = rotational_average(img, fold=npf)
        
        return out
    
    @batch_process
    def reconstruct(self, 
                    i: int = None,
                    *, 
                    rot_ave: bool = False,
                    y_length: nm = 50.0):
        """
        3D reconstruction of MT.

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
        ip.arrays.ImgArray
            Reconstructed image.
        """        
        # TODO: dask parallelize, iteration
        
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
        imgs = []
        ylen = 99999
    
        # Split image into dimers along y-direction
        for start, stop, ang in zip(borders[:-1], borders[1:], skew_angles):
            start = self.nm2pixel(start)
            stop = self.nm2pixel(stop)
            imgs.append(img_st[:, start:stop].rotate(-ang, dims="zx", mode="reflect"))
            ylen = min(ylen, stop-start)

        # align each fragment
        imgs[0] = imgs[0][:, :ylen]
        ref = imgs[0]
        shape = ref.shape
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        for i in range(1, len(imgs)):
            img = imgs[i][:, :ylen]
            shift = ip.pcc_maximum(img, ref, mask=mask)
            imgs[i] = img.affine(translation=shift, mode="grid-wrap")

        out = np.stack(imgs, axis="p").proj("p")
                    
        # rotational averaging
        # TODO: don't Affine twice
        if rot_ave:
            input_ = out.copy()
            center = np.array(out.shape)/2 - 0.5
            trs0 = np.eye(4, dtype=np.float32)
            trs1 = np.eye(4, dtype=np.float32)
            trs0[:3, 3] = -center
            trs1[:3, 3] = center
            for i in range(1, npf):
                ang = -2*np.pi*i/npf
                slope = np.tan(np.deg2rad(rise))
                dy = 2*np.pi*i/npf*radius*slope/self.scale
                cos = np.cos(ang)
                sin = np.sin(ang)
                rot = np.array([[cos, 0.,-sin, 0.],
                                [ 0., 1.,  0., dy],
                                [sin, 0., cos, 0.],
                                [ 0., 0.,  0., 1.]],
                                dtype=np.float32)
                mtx = trs1 @ rot @ trs0
                out.value[:] += input_.affine(mtx, mode="grid-wrap")
        
        # stack images for better visualization
        dup = ceilint(y_length/lp)
        outlist = [out]
        if dup > 0:
            for ang in skew_angles[:min(dup, len(skew_angles))-1]:
                outlist.append(out.rotate(ang, dims="zx", mode="reflect"))
    
        return np.concatenate(outlist, axis="y")
    
    @batch_process
    def cylindric_reconstruct(self, 
                              i: int = None,
                              rot_ave: bool = False, 
                              y_length: nm = 50.0):
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
        ip.arrays.ImgArray
            Reconstructed image.
        """        
        # TODO: dask parallelize
        
        # Cartesian transformation along spline.
        img_open = self.straighten(i, cylindrical=True)
        scale = img_open.scale.y
        total_length: nm = img_open.shape.y*scale
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
        props = self.global_ft_params(i)
        lp = props[H.yPitch] * 2
        skew = props[H.skewAngle]
        rise = props[H.riseAngle]
        npf = roundint(props[H.nPF])
        radius = self.paths[i].radius
        
        # Determine how to split image into tubulin dimer fragments
        dl, resl = divmod(total_length, lp)
        borders = np.linspace(0, total_length - resl, int((total_length - resl)/lp)+1)
        skew_angles = np.arange(borders.size - 1) * skew
        
        # Rotate fragment with skew angle
        imgs = []
        ylen = 99999
    
        # Split image into dimers along y-direction
        for start, stop, ang in zip(borders[:-1], borders[1:], skew_angles):
            start = self.nm2pixel(start)
            stop = self.nm2pixel(stop)
            shift = ang/360*img_open.shape.a
            imgs.append(
                img_open[:, start:stop].affine(translation=[0, shift],
                                               dims="ya", mode="grid-wrap")
                )
            ylen = min(ylen, stop-start)
        
        # align each fragment
        imgs[0] = imgs[0][f"y=:{ylen}"]
        ref = imgs[0]
        shape = ref.shape
        mask = ip.circular_mask(radius=[s//4 for s in shape], shape=shape)
        for i in range(1, len(imgs)):
            img = imgs[i][f"y=:{ylen}"]
            shift = ip.pcc_maximum(img, ref, mask=mask)
            imgs[i] = img.affine(translation=shift, 
                                 dims="rya", mode="grid-wrap")

        out = np.stack(imgs, axis="p").proj("p")
        
        # rotational averaging
        # TODO: don't Affine twice
        if rot_ave:
            input_ = out.copy()
            for i in range(1, npf):
                slope = np.tan(np.deg2rad(rise))
                dy = 2*np.pi*i/npf*radius*slope/self.scale
                shift_a = out.shape.a/npf*i
                shift = [dy, shift_a]
                out.value[:] += input_.affine(translation=shift, 
                                              dims="ya", mode="grid-wrap")
        
        # stack images for better visualization
        dup = ceilint(y_length/lp)
        outlist = [out]
        if dup > 0:
            for ang in skew_angles[:min(dup, len(skew_angles))-1]:
                shift = ang/360*img_open.shape.a
                outlist.append(out.affine(translation=[0, -shift], 
                                          dims="ya", mode="grid-wrap"))
    
        return np.concatenate(outlist, axis="y")
    
    @batch_process
    def map_monomer(self, i: int = None) -> Coordinates:
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
        # TODO: Sometimes skew angles seems to be not accurate enough for correct tubulin mapping.
        # ... and sometimes offset is wrong.
        
        spl = self._paths[i]
        rec_cyl = self.cylindric_reconstruct(i, rot_ave=True, y_length=0)
        r0 = self.nm2pixel(spl.radius)
        rec2d = rec_cyl[f"r={r0}:"].proj("r")
        if self.light_background:
            ymax, amax = np.unravel_index(np.argmin(rec2d), rec2d.shape)
        else:
            ymax, amax = np.unravel_index(np.argmax(rec2d), rec2d.shape)
        
        # Calculate Fourier parameters by cylindrical transformation along spline.
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
        
        crds = Coordinates(world = spl.inv_cylindrical(coords=mesh, ylength=ny*pitch),
                           spline = mesh)
        return crds
        
def angle_corr(img, ang_center: float = 0, drot: float = 7, nrots: int = 29):
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

def delayed_angle_corr(imgs, ang_centers, drot: float=7, nrots: int = 29):
    _angle_corr = delayed(partial(angle_corr, drot=drot, nrots=nrots))
    tasks = []
    for img, ang in zip(imgs, ang_centers):
        tasks.append(da.from_delayed(_angle_corr(img, ang), shape=(), dtype=np.float32))
    return da.compute(tasks, scheduler=SCHEDULER)[0]
    
def _local_dft_params(img, radius: nm):
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
    
    skew = np.arctan(-y_freq[ymax_f]/a_freq[amax_f]*2*y_pitch/radius)
    start = l_circ/y_pitch/(-np.tan(skew) + 1/np.tan(rise))
    
    return np.array([np.rad2deg(rise), 
                     y_pitch, 
                     np.rad2deg(skew), 
                     amaxp_f/up_a,
                     abs(start)], 
                    dtype=np.float32)
    

def ft_params(img, coords, radius):
    polar = map_coordinates(img, coords, order=3, mode="grid-wrap")
    polar = ip.asarray(polar, axes="rya") # radius, y, angle
    polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
    polar.scale_unit = img.scale_unit
    return _local_dft_params(polar, radius)

lazy_ft_params = delayed(ft_params)

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
