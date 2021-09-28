from __future__ import annotations
import statistics
from dask.array.chunk import astype
import numpy as np
from numba import jit
from skimage.transform import warp_polar
from scipy import ndimage as ndi
from ._dependencies import impy as ip
import pandas as pd
from dask import delayed, array as da

from .const import Header
from .spline import Spline3D

def _range(start:int, stop:int, step:int):
    i = start
    stop_ = stop - 1
    while i < stop_:
        yield i
        i += step
    yield stop_

def _extend_with_nan(a, length):
    a = list(a)
    x = length - len(a)
    if x < 0:
        raise ValueError("Cannot extend. Input is longer than output.")
    return a + [np.nan] * x

def make_slice_and_pad(center:int, radius:int, size:int):
    z0 = center - radius
    z1 = center + radius + 1
    z0_pad = z1_pad = 0
    if z0 < 0:
        z0_pad = -z0
        z0 = 0
    elif size < z1:
        z1_pad = z1 - size
        z1 = size

    return slice(z0, z1), (z0_pad, z1_pad)

def calc_total_length(path):
    each_length = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    total_length = np.sum(each_length)
    return total_length

def vector_to_grad(dr):
    # Here, it is important that gradients in yx-plane are in range of -2pi:2pi while those in zy-plane
    # are restricted in range of -pi:pi. Otherwise, MT polarity will be reversed more than once. This
    # causes a problem that MT polarity appears the same no matter in which direction you see.
    yx = np.rad2deg(np.arctan2(-dr[:,2], dr[:,1]))
    zy = np.rad2deg(np.arctan(np.sign(dr[:,1])*dr[:,0]/np.abs(dr[:,1])))
    return zy, yx

def load_a_subtomogram(img, pos, radius:tuple[int, int, int], dask:bool=True):
    """
    From large image ``img``, crop out small region centered at ``pos``.
    Image will be padded if needed.
    """
    z, y, x = pos.astype(np.int32)
    rz, ry, rx = (np.array(radius)/img.scale).astype(np.int32)
    sizez, sizey, sizex = img.sizesof("zyx")

    sl_z, pad_z = make_slice_and_pad(z, rz, sizez)
    sl_y, pad_y = make_slice_and_pad(y, ry, sizey)
    sl_x, pad_x = make_slice_and_pad(x, rx, sizex)
    reg = img[sl_z, sl_y, sl_x]
    if dask:
        reg = reg.data
    with ip.SetConst("SHOW_PROGRESS", False):
        pads = [pad_z, pad_y, pad_x]
        if np.any(np.array(pads) > 0):
            reg = reg.pad(pads, dims="zyx", constant_values=np.median(reg))
    return reg

@jit("(f4[:,:],f4[:,:],f4)", nopython=True, cache=True)
def _get_coordinate(path:np.ndarray, coords:np.ndarray, interval:float=1.0):
    npoints, ndim = path.shape
    inc = 0.01*interval
    dist_unit = interval - inc/2.0
    r2 = path[0]
    r1 = np.zeros(r2.size)
    r = np.zeros(r2.size)
    r_last = np.zeros(r2.size)
    dr = np.zeros(r2.size)

    nout = 0
    for i in range(1, npoints):
        r1[:] = r2
        r[:] = r1
        r2[:] = path[i]
        dr[:] = r2 - r1
        distance = np.sqrt(np.sum(dr**2))
        r_inc = dr * inc / distance
        n2 = int(distance/inc)
        if npoints == 2:
            n2 += 1
        while n2 >= 0:
            dr[:] = r - r_last
            distance = np.sqrt(np.sum(dr**2))
            if distance >= dist_unit:
                coords[:, nout] = r
                r_last[:] = r
                nout += 1
            r += r_inc
            n2 -= 1

def get_coordinates(path, interval):
    npoints, ndim = path.shape
    total_length = calc_total_length(path)
    coords = np.zeros((ndim, int(total_length/interval)+1), dtype=np.float32) - 1
    _get_coordinate(path.copy(), coords, interval)
    while coords[0, -1] < 0:
        coords = coords[:,:-1]
    return coords.T

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
    out = []
    input_img = np.moveaxis(img3d.value, 1, 2)
    out = warp_polar(input_img, center=center, radius=radius, 
                     output_shape=(angle_freq, radius), multichannel=True)
    out = ip.asarray(out, axes="<ry")
    return out

def make_rotate_mat(deg_yx, deg_zy, shape):
    compose_affine_matrix = ip.arrays.utils._transform.compose_affine_matrix
    center = np.array(shape)/2. - 0.5
    translation_0 = compose_affine_matrix(translation=center, ndim=3)
    rotation_yx = compose_affine_matrix(rotation=[0, 0, -np.deg2rad(deg_yx + 180)], ndim=3)
    rotation_zy = compose_affine_matrix(rotation=[-np.deg2rad(deg_zy), 0, 0], ndim=3)
    translation_1 = compose_affine_matrix(translation=-center, ndim=3)
    
    mx = translation_0 @ rotation_zy @ rotation_yx @ translation_1
    mx[-1, :] = [0, 0, 0, 1]
    return mx

def rot3d(img, yx, zy):
    mat = make_rotate_mat(yx, zy, img.shape)
    return img.affine(matrix=mat)

lazy_rot3d = delayed(rot3d)

@delayed
def rot3dinv(img, yx, zy):
    mat = np.linalg.inv(make_rotate_mat(yx, zy, img.shape))
    return img.affine(matrix=mat)

def _calc_pitch_length_xyz(img3d):
    up = 20 # upsample factor along y-axis
    peak_est = img3d.sizeof("y")/(4.16/img3d.scale.y) # estimated peak
    y0 = int(peak_est*0.8)
    y1 = int(peak_est*1.3)
    projection = img3d.local_power_spectra(key=f"y={y0}:{y1}", upsample_factor=[1, up, 1], dims="zyx").proj("zx")
    imax = np.argmax(projection)
    imax_f = imax + y0*up
    freq = np.fft.fftfreq(img3d.sizeof("y")*up)
    pitch = 1/freq[imax_f]*img3d.scale.y
    return pitch

@delayed
def _calc_pitch_length(img3d, rmin, rmax, up=20):
    peak_est = img3d.sizeof("y")/(4.16/img3d.scale.y) # estimated peak
    y0 = int(peak_est*0.8)
    y1 = int(peak_est*1.3)
    
    polar3d = warp_polar_3d(img3d, radius=rmax, angle_freq=int((rmin+rmax)*np.pi))[:, rmin:]
    power_spectra = polar3d.local_power_spectra(key=f"y={y0}:{y1}", upsample_factor=[1, 1, up], dims="<ry")
    
    proj_along_y = power_spectra.proj("<r")
    imax = np.argmax(proj_along_y)
    imax_f = imax + y0*up
    freq = np.fft.fftfreq(img3d.sizeof("y")*up)
    pitch = 1/freq[imax_f]*img3d.scale.y
    
    return pitch


def rotational_average(img, fold:int=13):
    angles = np.arange(fold)*360/fold
    average_img = img.copy()
    with ip.SetConst("SHOW_PROGRESS", False):
        for angle in angles[1:]:
            average_img.value[:] += img.rotate(angle, dims="zx")
    average_img /= fold
    return average_img

@delayed
def _calc_pf_number(img2d, mask):
    corrs = []
    for pf in [12, 13, 14, 15, 16]:
        av = rotational_average(img2d, pf)
        corrs.append(ip.zncc(img2d, av, mask))
    
    return np.argmax(corrs) + 12


class MTPath:
    inner = 0.7
    outer = 1.6
    def __init__(self, scale:float, interval_nm:float=24, radius_pre_nm=(22, 28, 28), radius_nm=(16.7, 16.7, 16.7),
                 light_background:bool=True, label:int=-1):
        self.scale = scale
        self.interval = interval_nm
        self.radius_pre = np.array(radius_pre_nm)
        self.radius = np.array(radius_nm)
        self.light_background = light_background
        
        self._step: int = 1
        
        self.points: np.ndarray = None
        self.spl: Spline3D = None
        
        self.subtomograms: list[np.ndarray] = []
        
        self.grad_angles_yx: np.ndarray = None
        self.grad_angles_zy: np.ndarray = None
        
        self._pf_numbers = None
        self._average_images = None
        
        self.label = int(label)
    
    @property
    def npoints(self) -> int:
        return self.points.shape[0]
    
    @property
    def ntomograms(self) -> int:
        return len(self.subtomograms) # TODO: fix after using da.stack
    
    @property
    def pf_numbers(self):
        if self._pf_numbers is None:
            self.calc_pf_numbers()
        return self._pf_numbers
    
    @pf_numbers.setter
    def pf_numbers(self, value:list):
        self._pf_numbers = np.array(value, dtype=np.uint8)
        
    @property
    def average_images(self):
        if self._average_images is None:
            self.rot_ave_zx()
        return self._average_images
    
    @average_images.setter
    def average_images(self, value):
        self._average_images = value
    
    @property
    def curvature(self):
        # https://en.wikipedia.org/wiki/Curvature#Space_curves
        dz, dy, dx = self.spl.partition(self.npoints, 1).T
        ddz, ddy, ddx = self.spl.partition(self.npoints, 2).T
        a = (ddz*dy-ddy*dz)**2 + (ddx*dz-ddz*dx)**2 + (ddy*dx-ddx*dy)**2
        return np.sqrt(a)/(dx**2+dy**2+dz**2)**1.5/self.interval
    
    
    def __add__(self, other:MTPath):
        # TODO: test
        if not isinstance(other, MTPath):
            raise TypeError("Only MTPath can be added to MTPath")
        new = self.__class__(self, self.scale, interval_nm=self.interval, radius_pre_nm=self.radius_pre, 
                             radius_nm=self.radius, light_background=self.light_background, label=-1)
        for attr in ["_sub_images", "_even_interval_points", "grad_angles_yx", "grad_angles_zy", "_pf_numbers"]:
            setattr(new, attr, getattr(self, attr) + getattr(other, attr))
        return new
        
    def set_path(self, coordinates, step:int=None):
        # coordinates: nm
        original_points = np.asarray(coordinates, dtype=np.float32)
        self.points = get_coordinates(original_points, self.interval)
        
        # when interval is very short, we don't have to run angle/shift correction for all the subtomograms
        self._step = step or max(int(30/self.interval), 1)
        if np.ceil(self.npoints/self._step) < 4:
            self._step = self.npoints//4
        return self
    
    def load_subtomograms(self, img):
        """
        From a large image crop out sub-images around each point of the path.
        """        
        subtomograms = []
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in _range(0, self.npoints, self._step):
                subtomograms.append(
                    load_a_subtomogram(img, self.points[i]/self.scale, self.radius_pre)
                    )
        self.subtomograms = np.stack(subtomograms, axis="p")
        return self
        
    def grad_path(self):
        """
        Calculate gradient at each point of the path.
        """        
        dr  = np.gradient(self.points, axis=0)
        self.grad_angles_zy, self.grad_angles_yx = vector_to_grad(dr)
        return self
    
    def smooth_path(self):
        """
        Correct/smoothen calculated gradients angular correlation and median filter.
        Gradient at either tip is set to the same value as the adjacent one respectively.
        """        
        new_angles = np.zeros_like(self.grad_angles_yx)
        
        with ip.SetConst("SHOW_PROGRESS", False):
            tasks = []
            for i in range(1, self.ntomograms-1):
                task = lazy_angle_corr(self.subtomograms[i], self.grad_angles_yx[i])
                tasks.append(da.from_delayed(task, shape=(), dtype=np.float32))
            new_angles = np.array([0] + da.compute(tasks)[0] + [0])
        
        size = 2*int(round(48/self.interval/self._step)) + 1
        if size > 1:
            new_angles = ndi.median_filter(new_angles, size=size, mode="mirror") # a b c d | c b a
        self.grad_angles_yx = new_angles
        return self

    def rot_correction(self):
        with ip.SetConst("SHOW_PROGRESS", False):
            for i, img in enumerate(self.subtomograms):
                angle = self.grad_angles_yx[i]
                img.rotate(-angle, cval=np.median(img), update=True)
        
        return self
    
    def zxshift_correction(self):
        with ip.SetConst("SHOW_PROGRESS", False):
            subtomo_proj = self.subtomograms.proj("y")
            iref = self.ntomograms//2
            imgref = subtomo_proj[iref]
            shape = np.array(imgref.shape)
            shifts = np.zeros((self.ntomograms, 2)) # zx-shift
            imgs = []
            bg = np.median(imgref)
            for i in range(self.ntomograms):
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
            self.shifts = shifts - center_shift/2
        
        return self
    
    def get_spline(self):
        coords = np.array([self.points[i] for i in _range(0, self.npoints, self._step)], dtype=np.float32)
        for i in range(self.ntomograms):
            shiftz, shiftx = -self.shifts[i]
            shift = np.array([shiftz, 0, shiftx])
            deg = self.grad_angles_yx[i]
            rad = -np.deg2rad(deg)
            cos = np.cos(rad)
            sin = np.sin(rad)
            shift = shift @ [[1.,   0.,  0.],
                             [0.,  cos, sin],
                             [0., -sin, cos]]
            coords[i] += shift * self.scale

        error_nm = 1.0
        sqsum = error_nm**2 * coords.shape[0] # unit: nm^2
        self.spl = Spline3D(coords, s=sqsum)
        
        # update gradients
        u = np.linspace(0, 1, self.npoints)
        self.points = self.spl(u)
        dr  = self.spl(u, der=1)
        self.grad_angles_zy, self.grad_angles_yx = vector_to_grad(dr)
        del self.shifts
        self._step = 1
        return self
    
    def rotate3d(self):
        tasks = []
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in range(self.npoints):
                img = self.subtomograms[i]
                zy = self.grad_angles_zy[i]
                yx = self.grad_angles_yx[i]
                tasks.append(da.from_delayed(lazy_rot3d(img, yx, zy), shape=img.shape, dtype=np.float32))
            out = da.compute(tasks)[0]
        self.subtomograms = np.stack(out, axis="p")
        return self
    
    def determine_radius(self):
        with ip.SetConst("SHOW_PROGRESS", False):
            sum_img = np.sum(self.subtomograms, axis="p")
            nbin = 17
            r_max = 17
            
            img2d = sum_img.proj("y")
            if self.light_background:
                r_peak = np.argmin(img2d.radial_profile(nbin=nbin, r_max=r_max))/nbin*r_max
            else:
                r_peak = np.argmax(img2d.radial_profile(nbin=nbin, r_max=r_max))/nbin*r_max
            self.radius_peak = r_peak # unit: nm
        return self
    
    def calc_pitch_lengths(self, upsample_factor:int=20):
        self.pitch_lengths = []
        ylen = int(self.radius[1]/self.scale)
        ylen0 = int(self.radius_pre[1]/self.scale)
        sl = (slice(None), slice(ylen0 - ylen, ylen0 + ylen + 1))
        tasks = []
        with ip.SetConst("SHOW_PROGRESS", False):
            for img in self.subtomograms:
                r = self.radius_peak/self.scale
                pitch = _calc_pitch_length(img[sl], 
                                           int(r*self.__class__.inner),
                                           int(r*self.__class__.outer),
                                           up=upsample_factor)
                tasks.append(da.from_delayed(pitch, shape=(), dtype=np.float64))
            
            self.pitch_lengths = da.compute(tasks)[0]

        return self

    def calc_pf_numbers(self):
        ylen = int(self.radius[1]/self.scale)
        ylen0 = int(self.radius_pre[1]/self.scale)
        sl = (slice(None), slice(ylen0 - ylen, ylen0 + ylen + 1))
        
        # make mask
        mask = self._mt_mask(self.subtomograms.sizesof("zx"))
        
        tasks = []
        with ip.SetConst("SHOW_PROGRESS", False):
            for img in self.subtomograms:
                npf = _calc_pf_number(img[sl].proj("y"), mask)
                tasks.append(da.from_delayed(npf, shape=(), dtype=np.float64))
            
            self.pf_numbers = da.compute(tasks)[0]
        
        return self
    
    def rot_ave_zx(self):
        average_images = []
        ylen = int(self.radius[1]/self.scale)
        ylen0 = int(self.radius_pre[1]/self.scale)
        sl = (slice(None), slice(ylen0 - ylen, ylen0 + ylen + 1))
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in range(self.npoints):
                npf = self.pf_numbers[i]
                img = self.subtomograms[i]
                av = rotational_average(img[sl].proj("y"), npf)
                average_images.append(av)
        self.average_images = average_images
        return self
    
    def to_dataframe(self):
        t, c, k = self.spl.tck

        data = {Header.label        : np.array([self.label]*self.npoints, dtype=np.uint16),
                Header.number       : np.arange(self.npoints, dtype=np.uint16),
                Header.z            : self.points[:, 0],
                Header.y            : self.points[:, 1],
                Header.x            : self.points[:, 2],
                Header.angle_zy     : self.grad_angles_zy,
                Header.angle_yx     : self.grad_angles_yx,
                Header.MTradius     : [self.radius_peak]*self.npoints,
                Header.pitch        : self.pitch_lengths,
                Header.nPF          : self.pf_numbers,
                Header.spl_knot_vec : _extend_with_nan([str(t.tolist())[1:-1]], self.npoints),
                Header.spl_coeff_z  : _extend_with_nan(c[0], self.npoints),
                Header.spl_coeff_y  : _extend_with_nan(c[1], self.npoints),
                Header.spl_coeff_x  : _extend_with_nan(c[2], self.npoints),
                Header.spl_u        : _extend_with_nan(self.spl.u, self.npoints)
                }
        
        df = pd.DataFrame(data)
        return df
    
    def from_dataframe(self, df:pd.DataFrame):
        self.points = df[Header.zyx()].values
        self.radius_peak = df[Header.MTradius].values[0]
        self.pitch_lengths = df[Header.pitch].values
        self.pf_numbers = df[Header.nPF].values
        self.grad_angles_zy = df[Header.angle_zy].values
        self.grad_angles_yx = df[Header.angle_yx].values
        
        knot: str = df[Header.spl_knot_vec].values[0]
        t = list(map(float, knot.split(",")))
        self.spl = Spline3D.prep(t = np.array(t),
                                 c = [df[Header.spl_coeff_z].dropna().values,
                                      df[Header.spl_coeff_y].dropna().values,
                                      df[Header.spl_coeff_x].dropna().values],
                                 u = df[Header.spl_u].values
                                 )
        
        return self
    
    def straighten(self, image, range: tuple[float, float] = (0, 1)):
        rz, ry, rx = (self.radius/self.scale).astype(np.int32)
        
        coords = self.spl.get_cartesian_coords((2*rz+1, 2*rx+1), s_range=range, scale=self.scale)
        transformed = ndi.map_coordinates(image, 
                                          np.moveaxis(coords, -1, 0),
                                          order=1,
                                          prefilter=False
                                          )
        transformed = ip.asarray(transformed, axes="zyx")
        transformed.set_scale(xyz=self.scale)
        return transformed
    
    # TODO: rewrite subtomogram averaging
    
    def average_subtomograms(self, niter:int=2, nshifts:int=19, nrots:int=9):
        df = pd.DataFrame([])
        
        kernel_box = (self.radius/self.scale).astype(np.int32)
        with ip.SetConst("SHOW_PROGRESS", False):
            input_imgs = [img.crop_kernel(kernel_box) for img in self.subtomograms]
        
        mask = np.stack([self._mt_mask(input_imgs[0].sizesof("zx"))]*input_imgs[0].sizeof("y"), axis=1)
        
        # initialize
        dshift = 4.5/self.scale
        drot = 1
        yshifts = np.zeros(self.npoints)
        zxrots = np.zeros(self.npoints)
        ref = input_imgs[self.npoints//2]
        offsets = None
                
        for it in range(1, niter+1):
            # parameter candidates
            shifts: np.ndarray = np.linspace(-dshift, dshift, nshifts).reshape(1, -1) + yshifts.reshape(-1, 1)
            rots: np.ndarray = np.linspace(-drot, drot, nrots).reshape(1, -1) + zxrots.reshape(-1, 1)
            
            # optimize parameters by correlation maximization
            out, yshifts, zxrots = zncc_all(input_imgs, ref, shifts, rots, mask=mask)
            
            # update
            col_shift = f"Yshift_{it}"
            col_rot = f"ZXrot_{it}"
            df[col_shift] = yshifts
            df[col_rot] = zxrots
            self.averaging_results = df[[col_shift, col_rot]]
            self.averaged_subtomogram = sum(out)/len(out)
            
            ref, offsets = self.rot_ave_subtomograms(dshift, nshifts, offsets=offsets)
            dshift = dshift / nshifts * 1.5
            drot = drot / nrots * 1.5
            
            self.tomogram_template = ref

            yield df, ref
        return df, ref
    
    def rot_ave_subtomograms(self, dshift:float, nshifts:int=9, offsets=None):
        rotsumimg = self.averaged_subtomogram.copy()
        npf = statistics.mode(self.pf_numbers)
        best_shifts = [] # unit: nm
        offsets = offsets or np.zeros(npf) # unit: nm
        
        with ip.SetConst("SHOW_PROGRESS", False):
            for n, deg in enumerate(360/npf*np.arange(npf)):
                rimg = self.averaged_subtomogram.rotate(deg, dims="zx")
                corrs = []
                offset = offsets[n]/self.scale # unit: nm -> pixel
                shifts = np.linspace(offset-dshift, offset+dshift, nshifts)
                for dy in shifts:
                    corrs.append(ip.zncc(
                        self.averaged_subtomogram, 
                        rimg.affine(translation=[0, dy, 0])
                        ))
                imax = np.argmax(corrs)
                shift = shifts[imax]
                best_shifts.append(shift*self.scale)
                rotsumimg.value[:] += rimg.affine(translation=[0, shift, 0])

        return rotsumimg, best_shifts
    
    def find_tubulin(self, bin_size:int=2):
        """Find tubulin coordinates in averaged tomogram"""
        if self.light_background:
            ref = -self.tomogram_template
        else:
            ref = self.tomogram_template
        
        length = self.tomogram_template.sizeof("y")*self.scale
        npf = int(statistics.mode(self.pf_numbers))
        ntub = int((length/4-2)*npf)
        
        with ip.SetConst("SHOW_PROGRESS", False):
            ref = ref.binning(bin_size, check_edges=False)
            ref_gauss = ref.gaussian_filter(1.0/ref.scale.x)
            mask2d = self._mt_mask(ref_gauss.sizesof("zx"), ref.scale.x)
            mask = np.stack([mask2d]*ref_gauss.sizeof("y"), axis=1)
            ref_gauss.append_label(~mask)
            peaks = ref_gauss.peak_local_max(min_distance=1.8/ref.scale.x, topn_per_label=ntub)
            peaks = ref_gauss.centroid_sm(peaks, radius=int(1.8/ref.scale.x))
            
        self.tubulin_local_coords = peaks.values*ref.scale
        return self.tubulin_local_coords
    
    def crop_out_tubulin(self, coords=None, r_nm=6.4, niter:int=3):
        if coords is None:
            coords = self.find_tubulin()
        
        coords_valid = []
        with ip.SetConst("SHOW_PROGRESS", False):
            # r_nm = max(self.radius[1] - self.interval/2, r_nm)
            edge = int(np.ceil(r_nm/self.scale))
            dr = int(np.ceil(4.8/self.scale))
            leny = self.averaged_subtomogram.sizeof("y")
            imgs = []
            for i, crd in enumerate(coords):
                # If y-coordinate is near either edges, ignore that.
                y = round(crd[1]/self.scale)
                if y < edge or leny - edge - 1 < y:
                    continue
                coords_valid.append(i)
                
                # Crop 3x3 tubulin region out
                imgs.append(
                    crop_out(self.averaged_subtomogram, crd, rz=dr, ry=edge, rx=edge)
                )
                
            imgs = np.stack(imgs, axis="p") # pzyx-image
            template = imgs.proj("p")
            
            shifts = np.zeros((imgs.sizeof("p"), 3))
            for _ in range(niter):
                for i, img in enumerate(imgs):
                    shift = ip.pcc_maximum(img, template)
                    imgs.value[i] = img.affine(translation=shift)
                    shifts[i] = shift
                template = imgs.proj("p")
        
        # Update tubulin template and local coordinates
        self.tubulin_template = template
        self.tubulin_local_coords = self.tubulin_local_coords[coords_valid]
        
        # Refine tubulin coordinates
        for i, coords in enumerate(self.tubulin_local_coords):
            
            angle = np.arctan2(coords[0], coords[2])
            sin = np.sin(angle)
            cos = np.cos(angle)
            coords += shifts[i] @ np.array([[ cos, 0, sin],
                                            [   0, 1,   0],
                                            [-sin, 0, cos]],
                                           dtype=np.float32)

            return template
    
    def transform_coordinates(self, coords=None):
        """
        Move tubulin center coordinates from the local coordinate system (around averaged subtomogram)
        to world coordinate system.

        Parameters
        ----------
        coords : (N, 3) ndarray
            Tubulin coordinates.
            
        Returns
        -------
        np.ndarray
            Tubulin coordinates. Shape is (P, N, 3), where P is the number of subtomograms.
        """        
        compose_affine_matrix = ip.arrays.utils._transform.compose_affine_matrix
        if coords is None:
            coords = self.tubulin_local_coords
        out = np.empty((self.npoints,) + coords.shape)
        template_center = (np.array(self.tomogram_template.shape)/2 - 0.5)*self.scale
        template_coords = np.concatenate([coords - template_center, 
                                          np.ones((coords.shape[0],1))],
                                         axis=1)

        for i, series in self.averaging_results.iterrows():
            yshift, zxrot = series
            zy = self.grad_angles_zy[i]
            yx = self.grad_angles_yx[i]
            vec = self.points[i]
            dy = yshift % self.pitch_lengths[i]
            if dy > self.pitch_lengths[i]/2:
                dy -= self.pitch_lengths[i]
            if dy != 0:
                dtheta = zxrot*(dy-yshift)/dy
            # local shift/rotation
            mtx0 = compose_affine_matrix(translation=[0, -dy, 0], rotation=[0, -dtheta, 0], ndim=3)
            mtx_move = compose_affine_matrix(translation=vec, ndim=3)
            mtx_rot_yx = compose_affine_matrix(rotation=[0,0,-np.deg2rad(yx+180)], ndim=3)
            mtx_rot_zy = compose_affine_matrix(rotation=[-np.deg2rad(zy), 0, 0], ndim=3)
            mtx = mtx_move @ mtx_rot_zy @ mtx_rot_yx @ mtx0

            points = template_coords @ mtx.T 
            out[i] = points[:,:3]

        return out
            

    def _mt_mask(self, shape, scale=None):
        scale = scale or self.scale
        mask = np.zeros(shape, dtype=np.bool_)
        z, x = np.indices(shape)
        cz, cx = np.array(shape)/2 - 0.5
        _sq = (z-cz)**2 + (x-cx)**2
        r = self.radius_peak/scale
        mask[_sq < (r*self.__class__.inner)**2] = True
        mask[_sq > (r*self.__class__.outer)**2] = True
        return mask
    
    def warp_spline_coordinate(self, image, start:int=0, stop:int=1):
        # TODO: not compatible with dask
        
        r = self.radius_peak/self.scale
        r_start = int(r*self.__class__.inner)
        r_stop = int(r*self.__class__.outer)
        coords_map = self.spl.get_cylindrical_coords((r_start, r_stop), (start, stop), self.scale)
        flat_img = ndi.map_coordinates(image, np.moveaxis(coords_map, -1, 0), order=1, prefilter=False)
        n_angle = int(round((r_start + r_stop) * np.pi))
        n_radius = r_stop - r_start
        img = flat_img.reshape(-1, n_angle, n_radius)
        return ip.asarray(img, axes="y<r")

    
@delayed
def _affine_zncc(ref, img, mask, translation, rotation):
    img_transformed = img.affine(translation=translation, rotation=rotation)
    return ip.zncc(ref, img_transformed, mask=mask)

def zncc_all(imgs, ref, shifts, rots, mask=None):
    yshifts = []
    zxrots = []
    out = []
    with ip.SetConst("SHOW_PROGRESS", False):
        for i, img in enumerate(imgs):
            if img is ref:
                yshifts.append(0.0)
                zxrots.append(0.0)
            else:
                ref = delayed(ref)
                corrs = np.zeros((len(shifts[0]), len(rots[0]))).tolist()                
                for j, dy in enumerate(shifts[i]):
                    for k, dtheta in enumerate(rots[i]):
                        task = _affine_zncc(ref, img, mask, [0, dy, 0], [0, dtheta, 0])
                        corrs[j][k] = da.from_delayed(task, shape=(), dtype=np.float32)
                        
                corrs = np.array(da.compute(corrs)[0], dtype=np.float32)
                jmax, kmax = np.unravel_index(np.argmax(corrs), corrs.shape)
                shift = shifts[i, jmax]
                yshifts.append(shift)
                deg = rots[i, kmax]
                zxrots.append(deg)
            out.append(img.affine(translation=[0, shift, 0], rotation=[0, deg, 0]))
            
    return out, np.array(yshifts), np.array(zxrots)

def crop_out(img, p0, rz=18, ry=24, rx=24):
    pz, py, px = p0/img.scale
    img0 = img[:, int(round(py-ry)):int(round(py+ry+1))]
    cz, cy, cx = np.array(img.shape)/2 + 0.5
    vc = np.array([cz, cx])
    vz, vx = np.array([pz, px]) - vc
    length = np.sqrt(vz**2+vx**2)
    e1 = np.array([-vx, vz]) / length
    e2 = np.array([vz, vx]) / length
    ori = np.array([vz, vx])- rx*e1 - rz*e2 + vc
    d1 = ori + (2*rx+1)*e1
    d2 = ori + (2*rz+1)*e2
    out = img0.rotated_crop(ori, d2, d1, dims="zx")
    return out