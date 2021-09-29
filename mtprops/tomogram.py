from __future__ import annotations
import statistics
from typing import Iterable
import numpy as np
from skimage.transform import warp_polar
from scipy import ndimage as ndi
from dataclasses import dataclass
from ._dependencies import impy as ip
import pandas as pd
from dask import delayed, array as da

from .const import nm
from .spline import Spline3D
from .cache import ArrayCacheMap
from .utils import load_a_subtomogram


_INNER = 0.7
_OUTER = 1.6

cachemap = ArrayCacheMap(maxgb=ip.Const["MAX_GB"])

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

@dataclass
class MtPathProps:
    spl: Spline3D
    radius: nm = None
    orientation: str = None
    localprops: pd.DataFrame = None
    
    def __post_init__(self):
        self.localprops = pd.DataFrame([])

class MtTomogram:
    def __init__(self, 
                 name=None,
                 radius_pre: tuple[nm, nm, nm] = (22.0, 28.0, 28.0), 
                 radius: tuple[nm, nm, nm] = (16.7, 16.7, 16.7),
                 light_background:bool=True
                 ):
        self.name = name
        self.radius_pre = radius_pre
        self.radius = radius
        self.light_background = light_background
        
        self.results: list[MtPathProps] = []
    
    def __hash__(self) -> int:
        return id(self)
    
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
        spl = Spline3D(self.scale)
        error_nm = 1.0
        sqsum = error_nm**2 * coords.shape[0] # unit: nm^2
        spl.fit(coords, s=sqsum)
        self.results.append(MtPathProps(spl))
        return None
    
    def nm2pixel(self, value: Iterable[nm]|nm):
        pix = (np.asarray(value)/self.scale).astype(np.int16)
        if np.isscalar(value):
            pix = int(pix)
        return pix
    
    def sample_subtomograms(self, 
                            i: int,
                            positions: tuple[float,...],
                            rotate: bool = True):
        if not (isinstance(positions, tuple) and rotate):
            return self._sample_subtomograms(i, np.asarray(positions, dtype=np.float32), rotate)
        try:
            out = cachemap[(self, self.results[i].spl, positions)]
        except KeyError:
            out = self._sample_subtomograms(i, np.asarray(positions, dtype=np.float32), rotate)
            cachemap[(self, self.results[i].spl, positions)] = out
        return out
        
    def _sample_subtomograms(self, 
                            i: int,
                            positions: tuple[float,...],
                            rotate: bool = True):
        spl = self.results[i].spl
        
        positions = np.asarray(positions, dtype=np.float32)
        center_px = self.nm2pixel(spl(positions))
        radius_px = self.nm2pixel(self.radius_pre)
        
        with ip.SetConst("SHOW_PROGRESS", False):
            if center_px.ndim == 2:
                out = np.stack([load_a_subtomogram(self._image, c, radius_px, True) 
                                for c in center_px],
                               axis="p")
                if rotate:
                    matrices = spl.rotation_matrix(positions, center=radius_px, inverse=True)
                    out.value[:] = dask_affine(out.value, matrices)
            else:
                out = load_a_subtomogram(self._image, center_px, radius_px, True)
                if rotate:
                    radius_px = np.array(out.shape)/2 - 0.5
                    mtx = spl.rotation_matrix(positions, center=radius_px, inverse=True)
                    out.value[:] = ndi.affine_transform(out.value, mtx, order=1, prefilter=False)
        
        return out
    
    def fit(self, i: int = 0, max_interval: nm = 24.0):
        spl = self.results[i].spl
        length = spl.length()
        nseg = int(length/max_interval) + 1
        interval: nm = length/nseg
        npoints = nseg + 1
        positions = np.linspace(0, 1, npoints)
        
        subtomograms = self.sample_subtomograms(i, positions=positions, rotate=False)
        ds = spl(positions, 1)
        yx_tilt = np.rad2deg(np.arctan2(-ds[:, 2], ds[:, 1]))
        refined_tilt = np.zeros_like(yx_tilt)
        with ip.SetConst("SHOW_PROGRESS", False):
            # Angular correlation
            tasks = []
            for i in range(1, npoints - 1):
                task = lazy_angle_corr(subtomograms[i], yx_tilt[i])
                tasks.append(da.from_delayed(task, shape=(), dtype=np.float32))
            refined_tilt = np.array([0] + da.compute(tasks)[0] + [0])
            size = 2*int(round(48/interval)) + 1
            if size > 1:
                refined_tilt = ndi.median_filter(refined_tilt, size=size, mode="mirror") # a b c d | c b a
            
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
        
        coords = spl(positions)
        for i in range(npoints):
            shiftz, shiftx = -shifts[i]
            shift = np.array([shiftz, 0, shiftx])
            deg = yx_tilt[i]
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
    
    def determine_radius(self, 
                         i: int, 
                         positions:Iterable[float]
                         ):
        with ip.SetConst("SHOW_PROGRESS", False):
            subtomograms = self.sample_subtomograms(i, positions, rotate=True)
            nbin = 17
            r_max: nm = 17.0
            img2d = subtomograms.proj("py")
            prof = img2d.radial_profile(nbin=nbin, r_max=r_max)
            if self.light_background:
                prof = -prof
                
            imax = np.argmax(prof)
            r_peak_sub = centroid(prof, imax-5, imax+5)/nbin*r_max
            self.results[i].radius = r_peak_sub
        return self
    
    def calc_ft_params(self,
                       i: int, 
                       positions:Iterable[float], 
                       *, 
                       upsample_factor: int = 20):

        subtomograms = self.sample_subtomograms(i, positions, rotate=True)
        ylen = self.nm2pixel(self.radius[1])
        ylen0 = self.nm2pixel(self.radius_pre[1])
        subtomograms = subtomograms[f"y={ylen0 - ylen}:{ylen0 + ylen + 1}"]
        spl = self.results[i].spl
        rmin = self.nm2pixel(self.results[i].radius*_INNER)
        rmax = self.nm2pixel(self.results[i].radius*_OUTER)
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
            
        self.results[i].localprops["splPosition"] = spl.distances(positions)
        self.results[i].localprops["skew"] = results[:, 0]
        self.results[i].localprops["y_pitch"] = results[:, 1]
        self.results[i].localprops["nPF"] = results[:, 2].astype(np.uint8)
        
        return results

    def straighten(self, 
                   i:int, 
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
                rz, ry, rx = self.nm2pixel(self.radius)
            elif np.isscalar(radius):
                rz = rx = self.nm2pixel(radius)
            else:
                rz, rx = self.nm2pixel(radius)
            spl = self.results[i].spl
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
    
    def average(self, i: int, l0: nm = 24, r:nm=19):
        spl = self.paths[i]
        l = spl.length()
        lmax = l//l0*l0
        kernel_box = self.nm2pixel(self.radius/self.scale)
        straight_image = self.straighten(i, radius=r)
        mean_pitch = np.mean(self.pitch_lengths[spl[i]])
        
        
    
    def _mt_mask(self, i: int, shape, scale=None):
        scale = scale or self.scale
        mask = np.zeros(shape, dtype=np.bool_)
        z, x = np.indices(shape)
        cz, cx = np.array(shape)/2 - 0.5
        _sq = (z-cz)**2 + (x-cx)**2
        r = self.mt_radius[self.paths[i]]/scale
        mask[_sq < (r*_INNER)**2] = True
        mask[_sq > (r*_OUTER)**2] = True
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

lazy_angle_corr = delayed(angle_corr)
    
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
    
    skew = a_freq[amax_f]*2*np.pi
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

