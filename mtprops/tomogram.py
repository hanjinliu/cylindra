from __future__ import annotations
import statistics
from typing import Iterable
import numpy as np
import json
from functools import wraps
from skimage.transform import warp_polar
from scipy import ndimage as ndi
from ._dependencies import impy as ip
import pandas as pd
from dask import array as da

from .const import nm, H
from .spline import Spline3D
from .cache import ArrayCacheMap
from .utils import load_a_subtomogram


INNER = 0.7
OUTER = 1.6

cachemap = ArrayCacheMap(maxgb=ip.Const["MAX_GB"])

def batch_process(func):
    @wraps(func)
    def _func(self: MtTomogram, i=None, **kwargs):
        if isinstance(i, int):
            return func(self, i=i, **kwargs)
        
        if i is None:
            i_list = list(range(self.n_paths))
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
        out = [func(self, i=i_, **kwargs) for i_ in i_list]
        if len(out) == 1:
            out = out[0]
        return out
    return _func            

class MtSpline(Spline3D):
    def __init__(self, scale:float=1, k=3):
        super().__init__(scale=scale, k=k)
        self.radius: nm = 0.0
        self.orientation: str = None
        self.localprops: pd.DataFrame = pd.DataFrame([])
    
    @classmethod
    def from_dict(cls, d: dict):
        self = super().from_dict(d)
        self.radius = d.get("radius", None)
        self.orientation = d.get("orientation", None)
        return self
    
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["radius"] = self.radius
        d["orientation"] = self.orientation
        d["localprops"] = self.localprops
        return d
        

class MtTomogram:
    def __init__(self, 
                 box_radius_pre: tuple[nm, nm, nm] = (22.0, 28.0, 28.0), 
                 box_radius: tuple[nm, nm, nm] = (16.7, 16.7, 16.7),
                 light_background: bool = True,
                 name: str = None,
                 ):
        self.name = name
        self.box_radius_pre = box_radius_pre
        self.box_radius = box_radius
        self.light_background = light_background
        
        self._paths: list[MtSpline] = []
    
    def __hash__(self) -> int:
        return id(self)
    
    @property
    def paths(self):
        return self._paths
    
    @property
    def n_paths(self):
        return len(self._paths)
    
    def save_results(self, path: str, **kwargs):
        df = pd.concat([spl.localprops for spl in self._paths], 
                        keys=np.arange(len(self._paths))
                       )
        df.to_csv(path, **kwargs)
        return None
    
    def save(self, path: str, contain_results: bool=True):
        path = str(path)
        
        all_results = {}
        for i, spl in enumerate(self._paths):
            spl_dict = spl.to_dict()
            if contain_results:
                spl_dict.update({H.splPosition: spl.localprops[H.splPosition],
                                 H.splDistance: spl.localprops[H.splDistance],
                                 H.skew: spl.localprops[H.skew],
                                 H.yPitch: spl.localprops[H.yPitch],
                                 H.nPF: spl.localprops[H.nPF],
                                 })

            all_results[i] = spl_dict
        
        with open(path, mode="w") as f:
            json.dump(all_results, f)
        return None
    
    def load(self, path :str):
        path = str(path)
        
        with open(path, mode="r") as f:
            js: dict = json.load(f)
        
        for i, d in js.items():
            self._paths.append(
                MtSpline.from_dict(d)
            )
            
        return self
    
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
    
    def add_path(self, coords: np.ndarray):
        spl = MtSpline(self.scale)
        error_nm = 1.0
        sqsum = error_nm**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        self._paths.append(spl)
        return None
    
    def nm2pixel(self, value: Iterable[nm]|nm):
        pix = (np.asarray(value)/self.scale).astype(np.int16)
        if np.isscalar(value):
            pix = int(pix)
        return pix
    
    def make_anchors(self, interval: nm):
        """
        Make anchors on every MtSpline objects

        Parameters
        ----------
        interval : nm
            Anchor intervals.
        """        
        for spl in self._paths:
            spl.make_anchors(interval)
        return None
    
    def _sample_subtomograms(self, 
                            i: int,
                            cache: bool = True,
                            rotate: bool = True):
        try:
            out = cachemap[(self, self._paths[i])]
        except KeyError:
            spl = self._paths[i]
            
            anchors = spl.anchors
            center_px = self.nm2pixel(spl(anchors))
            radius_px = self.nm2pixel(self.box_radius_pre)
            
            with ip.SetConst("SHOW_PROGRESS", False):
            
                out = np.stack([load_a_subtomogram(self._image, c, radius_px, True) 
                                for c in center_px],
                                axis="p")
                if rotate:
                    matrices = spl.rotation_matrix(anchors, center=radius_px, inverse=True)
                    out.value[:] = dask_affine(out.value, matrices)
            if cache:
                cachemap[(self, self._paths[i])] = out
            
        return out
    
    @batch_process
    def fit(self, i: int = None, max_interval: nm = 24.0):
        spl = self._paths[i]
        length = spl.length()
        npoints = int(np.ceil(length/max_interval)) + 1
        interval = length/(npoints-1)
        spl.make_anchors(n=npoints)
        subtomograms = self._sample_subtomograms(i, rotate=False, cache=False)
        img = da.from_array(subtomograms[1:-1], chunks=(1,)+subtomograms.shape[1:])
        ds = spl(der=1)
        yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
        
        with ip.SetConst("SHOW_PROGRESS", False):
            # Angular correlation
            out = dask_angle_corr(subtomograms[1:-1], yx_tilt)
            refined_tilt = np.array([0] + list(out) + [0])
            size = 2*int(round(48/interval)) + 1
            if size > 1:
                # Mirror-mode padding is "a b c d | c b a", thus edge values will be substituted
                # with the adjucent values respectively.
                refined_tilt = ndi.median_filter(refined_tilt, size=size, mode="mirror")
            
            # Rotate subtomograms            
            for i, img in enumerate(subtomograms):
                angle = refined_tilt[i]
                img.rotate(-angle, cval=np.median(img), update=True)
            
            subtomo_proj = subtomograms.proj("y")
            iref = npoints//2
            imgref = subtomo_proj[iref]
            shape = np.array(imgref.shape)
            shifts = np.zeros((npoints, 2)) # zx-shift
            imgs = []
            bg = np.median(imgref)
            for i in range(npoints):
                # NCC based template matching is robust for lattice defect etc.
                if i != iref:
                    corr = imgref.ncc_filter(subtomo_proj[i], bg=bg)
                    shifts[i] = np.unravel_index(np.argmax(corr), shape) - shape/2
                else:
                    shifts[i] = np.array([0, 0])
                    
                img = subtomo_proj[i]
                imgs.append(img.affine(translation=-shifts[i]))
                
            imgs = np.stack(imgs, axis="y")
            imgcory = imgs.proj("y")
            center_shift = ip.pcc_maximum(imgcory, imgcory[::-1,::-1])
            shifts -= center_shift/2
        
        coords = spl()
        for i in range(npoints):
            shiftz, shiftx = -shifts[i]
            shift = np.array([shiftz, 0, shiftx])
            deg = refined_tilt[i]
            rad = -np.deg2rad(deg)
            cos = np.cos(rad)
            sin = np.sin(rad)
            shift = shift @ [[1.,   0.,  0.],
                             [0.,  cos, sin],
                             [0., -sin, cos]]
            coords[i] += shift * self.scale
        
        # update spline
        error_nm = 1.0
        sqsum = error_nm**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        
        return self
    
    @batch_process
    def get_subtomograms(self, i: int = None):
        with ip.SetConst("SHOW_PROGRESS", False):
            subtomograms = self._sample_subtomograms(i, rotate=True)
            nbin = 17
            r_max: nm = 17.0
            img2d = subtomograms.proj("py")
            prof = img2d.radial_profile(nbin=nbin, r_max=r_max)
            if self.light_background:
                prof = -prof
                
            imax = np.argmax(prof)
            r_peak_sub = centroid(prof, imax-5, imax+5)/nbin*r_max
            self._paths[i].radius = r_peak_sub
        return subtomograms
    
    @batch_process
    def calc_ft_params(self, i: int = None, *, upsample_factor: int = 20):

        subtomograms = self._sample_subtomograms(i, rotate=True)
        ylen = self.nm2pixel(self.box_radius[1])
        ylen0 = self.nm2pixel(self.box_radius_pre[1])
        subtomograms = subtomograms[f"y={ylen0 - ylen}:{ylen0 + ylen + 1}"]
        spl = self._paths[i]
        rmin = self.nm2pixel(spl.radius*INNER)
        rmax = self.nm2pixel(spl.radius*OUTER)
        with ip.SetConst("SHOW_PROGRESS", False):
            
            img = da.from_array(subtomograms, chunks=(1,)+subtomograms.shape[1:])
            results: np.ndarray
            results = img.map_blocks(_calc_ft_params, 
                                     int(rmin),
                                     int(rmax),
                                     drop_axis=[1, 2, 3],
                                     up=upsample_factor,
                                     meta=np.array([], dtype=np.float32),
                                     ).compute().reshape(-1, 3)
        
        spl.localprops[H.splPosition] = spl.anchors
        spl.localprops[H.splDistance] = spl.distances()
        spl.localprops[H.skew] = results[:, 0]
        spl.localprops[H.yPitch] = results[:, 1]
        spl.localprops[H.nPF] = results[:, 2].astype(np.uint8)
        
        return spl.localprops

    @batch_process
    def straighten(self, 
                   i: int = None, 
                   range_: tuple[float, float] = (0.0, 1.0), 
                   radius: nm = None,
                   split: bool = True,
                   polar: bool = False):
        start, end = range_
        step = 0.1
        if split and end - start > step:
            out = []
            while start < end:
                if end - start < 1e-3:
                    break
                sub_range = (start, min(start+step, end))
                out.append(
                    self.straighten(i, range_=sub_range, radius=radius, 
                                    split=False, polar=polar)
                )
                start += step
                
            transformed = np.concatenate(out, axis="y")
            
        else:
            if radius is None:
                rz, ry, rx = self.nm2pixel(self.box_radius)
            elif np.isscalar(radius):
                rz = rx = self.nm2pixel(radius)
            else:
                rz, rx = self.nm2pixel(radius)
            spl = self._paths[i]
            if polar:
                if rx <= rz:
                    raise ValueError("For polar straightening, 'radius' must be (rmin, rmax)")
                coords = spl.cylindrical_coords((rz, rx), s_range=range_, scale=self.scale)
            else:
                coords = spl.cartesian_coords((2*rz+1, 2*rx+1), s_range=range_, scale=self.scale)
            coords = np.moveaxis(coords, -1, 0)
            
            # crop image and shift coordinates
            sl = []
            for i in range(3):
                imin = int(np.min(coords[i]))
                imax = int(np.max(coords[i])) + 2
                sl.append(slice(imin, imax))
                coords[i] -= imin
            sl = tuple(sl)
            transformed = ndi.map_coordinates(self.image[sl], 
                                            coords,
                                            order=1,
                                            prefilter=False
                                            )
            
            axes = "rya" if polar else "zyx"
            transformed = ip.asarray(transformed, axes=axes)
            transformed.set_scale({k: self.scale for k in axes})
            transformed.scale_unit = "nm"
        
        return transformed
    
    def average(self, i: int = None, l0: nm = 24, r:nm=19):
        # TODO
        ...
        
    @batch_process
    def _mt_mask(self, i: int = None, shape=None, scale=None):
        scale = scale or self.scale
        mask = np.zeros(shape, dtype=np.bool_)
        z, x = np.indices(shape)
        cz, cx = np.array(shape)/2 - 0.5
        _sq = (z-cz)**2 + (x-cx)**2
        r = self._paths[i].radius/scale
        mask[_sq < (r*INNER)**2] = True
        mask[_sq > (r*OUTER)**2] = True
        return mask

def angle_corr(img, ang_center:float=0, drot:float=7, nrots:int=29):
    # img: 3D
    img_z = img.proj("z")
    mask = ip.circular_mask(img_z.sizeof("y")/2+2, img_z.shape)
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


def dask_angle_corr(imgs, ang_centers):
    img = da.from_array(imgs, chunks=(1,)+imgs.shape[1:])
    ang_centers = da.from_array(ang_centers, chunks=(1,))
    
    def _angle_corr(imgs, angle_center):
        return np.array([angle_corr(imgs[0], angle_center[0])], dtype=np.float32)

    # Angular correlation
    out = img.map_blocks(_angle_corr, 
                         ang_centers, 
                         drop_axis=range(1, imgs.ndim),
                         meta=np.array([], dtype=np.float32)
                         ).compute()
    return out
    
def warp_polar_3d(img3d, center=None, radius=None, angle_freq=360):
    input_img = np.moveaxis(img3d.value, 1, 2)
    out = warp_polar(input_img, center=center, radius=radius, 
                     output_shape=(angle_freq, radius), multichannel=True, clip=False)
    out = ip.asarray(out, axes="ary") # angle, radius, y
    return out

def _calc_ft_params(img, rmin, rmax, up=20):
    img = img[0]
    peak_est = img.sizeof("y")/(4.16/img.scale.y) # estimated peak
    y0 = int(peak_est*0.8)
    y1 = int(peak_est*1.3)
    
    polar = warp_polar_3d(img, radius=rmax, angle_freq=int((rmin+rmax)*np.pi))[:, rmin:]
    da = 20
    power_1 = polar.local_power_spectra(key=f"y={y0}:{y1};a={-da}:", 
                                        upsample_factor=[up, 1, up], 
                                        dims="ary"
                                        ).proj("r")
    power_2 = polar.local_power_spectra(key=f"y={y0}:{y1};a=:{da+1}", 
                                        upsample_factor=[up, 1, up], 
                                        dims="ary"
                                        ).proj("r")
    power = np.concatenate([power_1, power_2], axis="a")
    amax, ymax = np.unravel_index(np.argmax(power), shape=power.shape)
    amax_f = amax - da*up
    ymax_f = ymax + y0*up
    a_freq = np.fft.fftfreq(polar.sizeof("a")*up)
    y_freq = np.fft.fftfreq(polar.sizeof("y")*up)
    
    skew = a_freq[amax_f]*360
    y_pitch = 1.0/y_freq[ymax_f]*img.scale.y
    
    power_a = polar.local_power_spectra(key="a=12:17;y=0:1", dims="ary")
    proj_along_a = power_a.proj("ry")
    npf = np.argmax(proj_along_a) + 12
    
    return np.array([skew, y_pitch, npf], dtype=np.float32)


def rotational_average(img, fold:int=13):
    angles = np.arange(fold)*360/fold
    average_img = img.copy()
    with ip.SetConst("SHOW_PROGRESS", False):
        for angle in angles[1:]:
            average_img.value[:] += img.rotate(angle, dims="zx")
    average_img /= fold
    return average_img

def _affine(img, matrix=None):
    out = ndi.affine_transform(img[0], matrix[0], order=1, prefilter=False)
    return out[np.newaxis]
    
def dask_affine(images, matrices):
    imgs = da.from_array(images, chunks=(1,)+images.shape[1:])
    mtxs = da.from_array(matrices, chunks=(1,)+matrices.shape[1:])
    return imgs.map_blocks(_affine, 
                           matrix=mtxs, 
                           meta=np.array([], dtype=np.float32)
                           ).compute()

def centroid(arr: np.ndarray, xmin: int, xmax: int) -> float:
    xmin = max(xmin, 0)
    xmax = min(xmax, arr.size)
    x = np.arange(xmin, xmax)
    input_arr = arr[xmin:xmax] - np.min(arr[xmin:xmax])
    return np.sum(input_arr*x)/np.sum(input_arr)