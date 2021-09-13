from __future__ import annotations
import statistics
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from scipy.signal import medfilt
from scipy import ndimage as ndi
from ._impy import impy as ip
import pandas as pd

def make_slice_and_pad(center, radius, size):
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

def load_subimage(img, pos, radius:tuple[int, int, int]):
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
    reg = img[sl_z, sl_y, sl_x].data
    with ip.SetConst("SHOW_PROGRESS", False):
        pads = [pad_z, pad_y, pad_x]
        if np.any(np.array(pads) > 0):
            reg = reg.pad(pads, dims="zyx", constant_values=np.median(reg))
    return reg

@jit(nopython=True)
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
    coords = np.zeros((ndim, int(total_length/interval)+1)) - 1
    _get_coordinate(path.copy(), coords, interval)
    while coords[0, -1]<0:
        coords = coords[:,:-1]
    return coords.T

def angle_corr(img, ang_center):
    # img: 3D
    img_z = img.proj("z")
    mask = ip.circular_mask(img_z.sizeof("y")/2+2, img_z.shape)
    img_mirror = img_z["x=::-1"]
    angs = np.linspace(ang_center-7, ang_center+7, 29, endpoint=True)
    corrs = []
    with ip.SetConst("SHOW_PROGRESS", False):
        for ang in angs:
            cor = ip.fourier_zncc(img_z, img_mirror.rotate(ang*2), mask)
            corrs.append(cor)
    angle = angs[np.argmax(corrs)]
    
    return angle
    
def warp_polar_3d(img3d, along="y", center=None, radius=None, angle_freq=360):
    out = []
    for sl, img in img3d.iter(along):
        out.append(warp_polar(img, center=center, radius=radius, output_shape=(angle_freq, radius)))
    out = ip.array(np.stack(out, axis=2), axes=f"<r{along}")
    return out

def make_rotate_mat(deg_yx, deg_zy, shape):
    compose_affine_matrix = ip.arrays.utils._transform.compose_affine_matrix
    center = np.array(shape)/2. - 0.5 # 3d array
    translation_0 = compose_affine_matrix(translation=center, ndim=3)
    rotation_yx = compose_affine_matrix(rotation=[0, 0, -np.deg2rad(deg_yx+180)], ndim=3)
    rotation_zy = compose_affine_matrix(rotation=[np.deg2rad(deg_zy), 0, 0], ndim=3)
    translation_1 = compose_affine_matrix(translation=-center, ndim=3)
    
    mx = translation_0 @ rotation_yx @ rotation_zy @ translation_1
    mx[-1, :] = [0, 0, 0, 1]
    return mx

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


def _calc_pitch_length(img3d, rmin, rmax):
    up = 20 # upsample factor along y-axis
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


def _calc_pf_number(img2d):
    corrs = []
    for pf in [12, 13, 14, 15, 16]:
        av = rotational_average(img2d, pf)
        corrs.append(ip.zncc(img2d, av))
    
    return np.argmax(corrs) + 12


class MTPath:
    inner = 0.7
    outer = 1.6
    def __init__(self, scale:float, interval_nm:float=24, radius_pre_nm=(22, 32, 32), radius_nm=(16.7, 16.7, 16.7),
                 light_background:bool=True, label:int=-1):
        self.scale = scale
        self.interval = interval_nm
        self.radius_pre = np.array(radius_pre_nm)
        self.radius = np.array(radius_nm)
        self.light_background = light_background
        
        self._even_interval_points = None
        
        self._sub_images = []
        
        self.grad_angles_yx = None
        self.grad_angles_zy = None
        
        self._pf_numbers = None
        self._average_images = None
        
        self.label = int(label)
    
    @property
    def npoints(self):
        return self._even_interval_points.shape[0]
    
    @property
    def pf_numbers(self):
        if self._pf_numbers is None:
            self.calc_pf_number()
        return self._pf_numbers
    
    @pf_numbers.setter
    def pf_numbers(self, value:list):
        self._pf_numbers = value
        
    @property
    def average_images(self):
        if self._average_images is None:
            self.rotational_averages()
        return self._average_images
    
    @average_images.setter
    def average_images(self, value):
        self._average_images = value
    
    @property
    def curvature(self):
        # https://en.wikipedia.org/wiki/Curvature#Space_curves
        z = self._even_interval_points[:, 0]
        y = self._even_interval_points[:, 1]
        x = self._even_interval_points[:, 2]
        dz = np.gradient(z)
        dy = np.gradient(y)
        dx = np.gradient(x)
        ddz = np.gradient(dz)
        ddy = np.gradient(dy)
        ddx = np.gradient(dx)
        a = (ddz*dy-ddy*dz)**2 + (ddx*dz-ddz*dx)**2 + (ddy*dx-ddx*dy)**2
        return np.sqrt(a)/(dx**2+dy**2+dz**2)**1.5/self.interval
    
    
    def __add__(self, other:MTPath):
        if not isinstance(other, MTPath):
            raise TypeError("Only MTPath can be added to MTPath")
        new = self.__class__(self, self.scale, interval_nm=self.interval, radius_pre_nm=self.radius_pre, 
                             radius_nm=self.radius, light_background=self.light_background, label=-1)
        for attr in ["_sub_images", "_even_interval_points", "grad_angles_yx", "grad_angles_zy", "_pf_numbers"]:
            setattr(new, attr, getattr(self, attr) + getattr(other, attr))
        return new
        
    def set_path(self, coordinates):
        original_points = np.asarray(coordinates)
        self._even_interval_points = get_coordinates(original_points, self.interval/self.scale)
        return self
    
    def load_images(self, img):
        """
        From a large image crop out sub-images around each point of the path.
        """        
        self._sub_images.clear()
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in range(self.npoints):
                subimg = load_subimage(img, self._even_interval_points[i], self.radius_pre)
                self._sub_images.append(subimg)
        return self
    
    def grad_path(self):
        """
        Calculate gradient at each point of the path.
        """        
        dr  = np.gradient(self._even_interval_points, axis=0)
        
        # Here, it is important that gradients in yx-plane are in range of -2pi:2pi while those in zy-plane
        # are restricted in range of -pi:pi. Otherwise, MT polarity will be reversed more than once. This
        # causes a problem that MT polarity appears the same no matter in which direction you see.
        self.grad_angles_yx = np.rad2deg(np.arctan2(-dr[:,2], dr[:,1]))
        self.grad_angles_zy = np.rad2deg(np.arctan(-dr[:,0]/np.abs(dr[:,1])))
        return self
    
    def smooth_path(self):
        """
        Correct/smoothen calculated gradients angular correlation and median filter.
        Gradient at either tip is set to the same value as the adjacent one respectively.
        """        
        new_angles = np.zeros_like(self.grad_angles_yx)
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in range(1, self.npoints-1):
                angle = angle_corr(self._sub_images[i], self.grad_angles_yx[i])
                new_angles[i] = angle
        
        size = 2*int(round(48/self.interval)) + 1
        if size > 1:
            new_angles = ndi.median_filter(new_angles, size=size, mode="mirror") # a b c d | c b a
        
        self.grad_angles_yx = new_angles
        return self
    
    def rot_correction(self):
        with ip.SetConst("SHOW_PROGRESS", False):
            for i, img in enumerate(self._sub_images):
                angle = self.grad_angles_yx[i]
                img.rotate(-angle, cval=np.median(img), update=True)
        
        return self
    
    def zxshift_correction(self):
        xlen0 = int(self.radius_pre[2]/self.scale)
        xlen = int(xlen0*0.8)
        sl = (slice(None), slice(None), slice(xlen0 - xlen, xlen0 + xlen + 1))
        with ip.SetConst("SHOW_PROGRESS", False):
            iref = self.npoints//2
            imgref = self._sub_images[iref].proj("y")
            shape = np.array(imgref.shape)
            shifts = [] # zx-shift
            bg = np.median(imgref)
            for i in range(self.npoints):
                if i != iref:
                    corr = imgref.ncc_filter(self._sub_images[i][sl].proj("y"), bg=bg) # ncc or pcc??
                    shift = np.unravel_index(np.argmax(corr), shape) - shape/2
                else:
                    shift = np.array([0, 0])
                shifts.append(list(shift))
            
        self.shifts = np.array(shifts)
        return self

    def calc_center_shift(self):
        xlen0 = int(self.radius_pre[2]/self.scale)
        xlen = int(xlen0*0.8)
        sl = (slice(None), slice(xlen0 - xlen, xlen0 + xlen + 1))
        with ip.SetConst("SHOW_PROGRESS", False):
            imgs = []
            for i in range(self.npoints):
                img = self._sub_images[i].proj("y")
                shift = self.shifts[i]
                imgs.append(img.affine(translation=-shift)[sl])
            imgs = np.stack(imgs, axis="y")
            imgcory = imgs.proj("y")
            center_shift = ip.pcc_maximum(imgcory, imgcory[::-1,::-1])
            self.shifts = self.shifts - center_shift/2
        
        return self
    
    def update_points(self):
        coords = self._even_interval_points.copy()
        for i in range(self.npoints):
            shiftz, shiftx = -self.shifts[i]
            shift = np.array([shiftz, 0, shiftx])
            deg = self.grad_angles_yx[i]
            rad = -np.deg2rad(deg)
            cos = np.cos(rad)
            sin = np.sin(rad)
            shift = shift @ [[1.,   0.,  0.],
                             [0.,  cos, sin],
                             [0., -sin, cos]]
            coords[i] += shift
        
        self._even_interval_points = coords
        
        size = 2*int(round(48/self.interval)) + 1
        if size > 1:
            self.grad_angles_yx = ndi.median_filter(self.grad_angles_yx, size=size, mode="mirror")
            self.grad_angles_zy = ndi.median_filter(self.grad_angles_zy, size=size, mode="mirror")
        
        return self
    
    def rotate3d(self):
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in range(self.npoints):
                mat = make_rotate_mat(self.grad_angles_yx[i], self.grad_angles_zy[i], 
                                      self._sub_images[i].shape)
                self._sub_images[i].affine(matrix=mat, update=True)
        return self
    
    def determine_radius(self):
        with ip.SetConst("SHOW_PROGRESS", False):
            sum_img = sum(self._sub_images)
            nbin = 17
            r_max = 17
            
            img2d = sum_img.proj("y")
            if self.light_background:
                r_peak = np.argmin(img2d.radial_profile(nbin=nbin, r_max=r_max))/nbin*r_max
            else:
                r_peak = np.argmax(img2d.radial_profile(nbin=nbin, r_max=r_max))/nbin*r_max
            self.radius_peak = r_peak
        return self
            
    
    def calc_pitch_xyz(self):
        self.pitch_lengths = []
        with ip.SetConst("SHOW_PROGRESS", False):
            for img in self._sub_images:
                img = img.crop_kernel((self.radius/self.scale).astype(np.int32))
                pitch = _calc_pitch_length_xyz(img)
                self.pitch_lengths.append(pitch)
        return self
    
    def calc_pitch_length(self):
        self.pitch_lengths = []
        ylen = int(self.radius[1]/self.scale)
        ylen0 = int(self.radius_pre[1]/self.scale)
        sl = (slice(None), slice(ylen0 - ylen, ylen0 + ylen + 1))
        with ip.SetConst("SHOW_PROGRESS", False):
            for img in self._sub_images:
                r = self.radius_peak/self.scale
                pitch = _calc_pitch_length(img[sl], 
                                           int(r*self.__class__.inner),
                                           int(r*self.__class__.outer))
                self.pitch_lengths.append(pitch)
        return self

    def calc_pf_number(self):
        pf_numbers = []
        ylen = int(self.radius[1]/self.scale)
        ylen0 = int(self.radius_pre[1]/self.scale)
        sl = (slice(None), slice(ylen0 - ylen, ylen0 + ylen + 1))
        with ip.SetConst("SHOW_PROGRESS", False):
            for img in self._sub_images:
                npf = _calc_pf_number(img[sl].proj("y"))
                pf_numbers.append(npf)
        self.pf_numbers = pf_numbers
        return self
    
    def rotational_averages(self):
        average_images = []
        ylen = int(self.radius[1]/self.scale)
        ylen0 = int(self.radius_pre[1]/self.scale)
        sl = (slice(None), slice(ylen0 - ylen, ylen0 + ylen + 1))
        with ip.SetConst("SHOW_PROGRESS", False):
            for i in range(self.npoints):
                npf = self.pf_numbers[i]
                img = self._sub_images[i]
                av = rotational_average(img[sl].proj("y"), npf)
                average_images.append(av)
        self.average_images = average_images
        return self
    
    def to_dataframe(self):
        data = {"label": np.array([self.label]*self.npoints, dtype=np.uint16),
                "number": np.arange(self.npoints, dtype=np.uint16),
                "z": self._even_interval_points[:, 0],
                "y": self._even_interval_points[:, 1],
                "x": self._even_interval_points[:, 2],
                "MTradius": [self.radius_peak]*self.npoints,
                "curvature": self.curvature,
                "pitch": self.pitch_lengths,
                "nPF": self.pf_numbers,
                }
        df = pd.DataFrame(data)
        return df
    
    def imshow_yx_raw(self, index:int, ax=None):
        if ax is None:
            ax = plt.gca()
        
        lz, ly, lx = self._sub_images[index].shape
        with ip.SetConst("SHOW_PROGRESS", False):
            ax.imshow(self._sub_images[index].proj("z"), cmap="gray")
        
        ylen = int(self.radius[1]/self.scale)
        ymin, ymax = ly/2 - ylen, ly/2 + ylen
        r = self.radius_peak/self.scale*self.__class__.outer
        xmin, xmax = -r + lx/2, r + lx/2
        ax.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color="lime")
        ax.text(1, 1, f"{self.label}-{index}", color="lime", font="Consolas", size=28, va="top")
        ax.tick_params(labelbottom=False,labelleft=False, labelright=False, labeltop=False)
        return None
    
    def imshow_zx_raw(self, index:int, ax=None):
        if ax is None:
            ax = plt.gca()
        lz, ly, lx = self._sub_images[index].shape
        with ip.SetConst("SHOW_PROGRESS", False):
            ax.imshow(self._sub_images[index].proj("y"), cmap="gray")
        theta = np.deg2rad(np.arange(360))
        r = self.radius_peak/self.scale*self.__class__.inner
        ax.plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = self.radius_peak/self.scale*self.__class__.outer
        ax.plot(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        ax.text(1, 1, f"{self.label}-{index}", color="lime", font="Consolas", size=28, va="top")
        ax.tick_params(labelbottom=False,labelleft=False, labelright=False, labeltop=False)
        return None
    
    def imshow_zx_ave(self, index:int, ax=None):
        if ax is None:
            ax = plt.gca()
        with ip.SetConst("SHOW_PROGRESS", False):
            ax.imshow(self.average_images[index], cmap="gray")
        ax.text(1, 1, f"{self.label}-{index}", color="lime", font="Consolas", size=28, va="top")
        ax.tick_params(labelbottom=False,labelleft=False, labelright=False, labeltop=False)
        return None

    def iter_run(self, img, path):
        yield self.set_path(path)
        yield self.load_images(img)
        yield self.grad_path()
        yield self.smooth_path()
        yield self.rot_correction()
        yield self.zxshift_correction()
        yield self.calc_center_shift()
        yield self.update_points()
        yield self.load_images(img)
        yield self.grad_path()
        yield self.rotate3d()
        yield self.determine_radius()
        yield self.calc_pitch_length()
        yield self.calc_pf_number()
        yield self.rotational_averages()
    
    def average_tomograms(self):
        df = pd.DataFrame([])
        
        kernel_box = (self.radius/self.scale).astype(np.int32)
        with ip.SetConst("SHOW_PROGRESS", False):
            input_imgs = [img.crop_kernel(kernel_box) for img in self._sub_images]
        
        # 1st iteration
        ref = input_imgs[self.npoints//2]
        shifts = np.linspace(-4.5/self.scale, 4.5/self.scale, 19)
        rots = np.linspace(-1, 1, 5)
        out, yshifts, zxrots = zncc_all(input_imgs, ref, shifts, rots)
        
        df["Yshift-1"] = yshifts
        df["ZXrot-1"] = zxrots
        
        self.tomogram_averaged_image = sum(out)/len(out)
        
        # 2nd iteration
        shifts = np.linspace(-0.5/self.scale, 0.5/self.scale, 19)
        if zxrots[0] == zxrots[-1]:
            rots = np.linspace(-0.2, 0.2, 5)
        else:
            theta_slope = (zxrots[-1] - zxrots[0])/len(out)
            rots = np.array([np.linspace(theta_slope*i-0.2, theta_slope*i+0.2, 5).tolist() 
                             for i in range(len(out))])
        out, yshifts, zxrots = zncc_all(out, self.rot_average_tomogram(), shifts, rots)
        
        df["Yshift-2"] = yshifts
        df["ZXrot-2"] = zxrots
        
        self.tomogram_averaged_image[:] = sum(out)/len(out)
        
        return df
    
    def rot_average_tomogram(self):
        shifts = np.linspace(-4.2/self.scale, 4.2/self.scale, 20)
        rotsumimg = self.tomogram_averaged_image.copy()
        npf = statistics.mode(self.pf_numbers)
        with ip.SetConst("SHOW_PROGRESS", False):
            for deg in (360/npf*np.arange(1, npf)):
                rimg = self.tomogram_averaged_image.rotate(deg, dims="zx")
                corrs = []
                for i, dy in enumerate(shifts):
                    corrs.append(ip.zncc(
                        self.tomogram_averaged_image, 
                        rimg.affine(translation=[0,dy,0])
                        ))
                imax = np.argmax(corrs)
                shift = shifts[imax]
                rotsumimg += rimg.affine(translation=[0,shift,0])
                shifts = np.linspace(shift - 1/self.scale, shift + 1/self.scale, 7)
        
        return rotsumimg

def zncc_all(imgs, ref, shifts, rots, mask=None):
    yshifts = []
    zxrots = []
    out = []
    if rots.ndim == 1:
        rots = np.stack([rots]*len(shifts), axis=0)
    corrs = np.zeros((len(shifts), len(rots[0])))
    with ip.SetConst("SHOW_PROGRESS", False):
        for img in imgs:
            if img is ref:
                yshifts.append(0.0)
                zxrots.append(0.0)
            else:
                for i, dy in enumerate(shifts):
                    for j, dtheta in enumerate(rots[i]):
                        corrs[i,j] = ip.zncc(
                            ref, 
                            img.affine(translation=[0, dy, 0], 
                                       rotation=[0, dtheta, 0]),
                            mask=mask
                        )
                imax, jmax = np.unravel_index(np.argmax(corrs), corrs.shape)
                shift = shifts[imax]
                yshifts.append(shift)
                deg = rots[i, jmax]
                zxrots.append(deg)
            out.append(img.affine(translation=[0,shift,0], rotation=[0,deg,0]))
    return out, yshifts, zxrots

