from __future__ import annotations
import numpy as np
import numba as nb
from scipy.interpolate import splprep, splev, interp1d
from skimage.transform._warps import _linear_polar_mapping

class Spline3D:
    tck: tuple[np.ndarray, list[np.ndarray], int]
    u: np.ndarray
    def __init__(self, coords:np.ndarray=None, k=3, s=None):
        if coords is None:
            return None

        npoints = coords.shape[0]
        if npoints < 4:
            lin = interp1d(np.linspace(0, 1, npoints), coords.T)
            coords = lin(np.linspace(0,1,4)).T
        self.tck, self.u = splprep(coords.T, k=k, s=s)

    @classmethod
    def prep(cls, t, c, u):
        self = cls()
        self.tck = (t, c, 3)
        self.u = u
        return self

    def __call__(self, u:np.ndarray|float, der:int=0) -> np.ndarray:
        if np.isscalar(u):
            coord = splev([u], self.tck, der=der)
            return np.concatenate(coord).astype(np.float32)
        else:
            coords = splev(u, self.tck, der=der)
            return np.stack(coords, axis=1).astype(np.float32)

    def partition(self, n:int, der:int=0):
        u = np.linspace(0, 1, n)
        return self(u, der)

    def length(self, start:float=0, stop:float=1, nknots:int=100):
        """
        Approximate the length of B-spline between [start, stop] by partitioning
        the spline with 'nknots' knots.
        """
        u = np.linspace(start, stop, nknots)
        dz, dy, dx = map(np.diff, splev(u, self.tck, der=0))
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

    def get_cartesian_coords(self, 
                             shape: tuple[int, int], 
                             s_range: tuple[float, float] = (0, 1), 
                             scale: float = 1
                             ) -> np.ndarray:
        return self._get_coords(_cartesian_coords_2d, shape, s_range, scale)

    def get_cylindrical_coords(self, 
                               r_range: tuple[int, int],
                               s_range: tuple[float, float] = (0, 1), 
                               scale: float = 1
                               ) -> np.ndarray:
        return self._get_coords(_polar_coords_2d, r_range, s_range, scale)

    def _get_coords(self, map_func, map_params:tuple, s_range:tuple[float, float], scale:float):
        s0, s1 = s_range
        length = self.length(start=s0, stop=s1)
        n_pixels = int(length/scale) + 1
        u = np.linspace(s0, s1, n_pixels)
        y_ax_coords = self(u)/scale # world coordinates of y-axis in spline coords system
        drlist = self(u, 1).astype(np.float32)
        map_ = map_func(*map_params)
        zeros = np.zeros(map_.shape[:-1], dtype=np.float32)
        ones = np.ones(map_.shape[:-1], dtype=np.float32)
        map_slice = np.stack([map_[..., 0], 
                              zeros,
                              map_[..., 1],
                              ones
                              ], axis=2) # V, H, D
        return _rot_with_vector(map_slice, y_ax_coords, drlist)

_V = slice(None) # vertical dimension
_S = slice(None) # longitudinal dimension along spline curve
_H = slice(None) # horizontal dimension
_D = slice(None, None, 1) # dimension of dimension (such as d=0: z, d=1: y,...)

@nb.njit(cache=True)
def _rot_point_with_vector(point: nb.float32[_V,_H,_D], 
                           dr: nb.float32[_D]
                           ) -> nb.float32[_V,_H,_D]:
    yx = np.rad2deg(np.arctan2(-dr[2], dr[1]))
    zy = np.rad2deg(np.arctan(np.sign(dr[1])*dr[0]/np.abs(dr[1])))
    zy = np.deg2rad(zy)
    yx = np.deg2rad(yx)
    cos = np.cos(yx)
    sin = np.sin(yx)
    rotation_yx = np.array([[1.,  0.,   0., 0.],
                            [0., cos, -sin, 0.],
                            [0., sin,  cos, 0.],
                            [0.,  0.,   0., 1.]],
                            dtype=np.float32)
    cos = np.cos(zy)
    sin = np.sin(zy)
    rotation_zy = np.array([[cos, -sin, 0., 0.],
                            [sin,  cos, 0., 0.],
                            [ 0.,   0., 1., 0.],
                            [ 0.,   0., 0., 1.]],
                            dtype=np.float32)

    mx = rotation_zy.dot(rotation_yx)
    mx[-1, :] = [0, 0, 0, 1]
    mx = np.ascontiguousarray(mx)
    
    out = np.empty(point.shape[:2] + (3,), dtype=np.float32)
    for i, p in enumerate(point):
        out[i] = p.dot(mx)[:, :3]
    return out

@nb.njit(cache=True)
def _rot_with_vector(maps: nb.float32[_V,_H,_D],
                     ax_coords: nb.float32[_S,_D],
                     vectors: nb.float32[_S,_D],
                     ) -> nb.float32[_V,_S,_H,_D]:

    coords = np.empty((maps.shape[0],
                       ax_coords.shape[0],
                       maps.shape[1],
                       3),
                      dtype=np.float32
                      )
    for i, (y, dr) in enumerate(zip(ax_coords, vectors)):
        slice_out = _rot_point_with_vector(maps, dr)
        coords[:, i] = slice_out + y
    return coords

def _polar_coords_2d(r_start: int, r_stop: int) -> np.ndarray:
    n_angle = int(round((r_start + r_stop) * np.pi))
    n_radius = r_stop - r_start
    r_, ang_ = np.indices((n_radius, n_angle))
    r_ += r_start
    output_coords = np.column_stack([r_.ravel(), ang_.ravel()])
    coords = _linear_polar_mapping(np.array(output_coords), n_angle/2/np.pi, 1, [0, 0]
                                   ).astype(np.float32)
    return coords.reshape(n_radius, n_angle, 2) # V, H, 2

def _cartesian_coords_2d(lenv, lenh):
    v, h = np.indices((lenv, lenh), dtype=np.float32)
    v -= (lenv/2 - 0.5)
    h -= (lenh/2 - 0.5)
    return np.stack([v, h], axis=2) # V, H, 2