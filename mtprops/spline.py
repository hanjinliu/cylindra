from __future__ import annotations
from typing import Callable, Iterable
import warnings
import numpy as np
import numba as nb
import json
from scipy.interpolate import splprep, splev, interp1d
from skimage.transform._warps import _linear_polar_mapping
from .const import nm

class Spline3D:
    """
    3D spline curve model with coordinate system.
    
    References
    ----------
    - Scipy document
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
    """    
    def __init__(self, scale:float=1, k=3):
        self._tck = None
        self._u = None
        self.scale = scale
        self._k = k
        self._anchors = None
        self._updates = 0
    
    @property
    def tck(self) -> tuple[np.ndarray, list[np.ndarray], int]:
        return self._tck
    
    @property
    def u(self) -> np.ndarray:
        return self._u
    
    @property
    def k(self) -> int:
        return self._k
        
    @property
    def anchors(self) -> np.ndarray:
        if self._anchors is None:
            raise ValueError("Anchor has not been set yet.")
        return self._anchors
    
    @anchors.setter
    def anchors(self, positions: float|Iterable[float]):
        positions = np.atleast_1d(np.asarray(positions, dtype=np.float32))
        if positions.ndim != 1:
            raise TypeError(f"Could not convert positions into 1D array.")
        elif positions.min() < 0 or positions.max() > 1:
            msg = f"Anchor positions should be set between 0 and 1. Otherwise spline " \
                  f"curve does not fit well."
            warnings.warn(msg, UserWarning)
        self._anchors = positions
        self._updates += 1
    
    def make_anchors(self, interval: nm = None, n: int=None):
        length = self.length()
        if interval is not None:
            n = int(length/interval) + 1
            end = (n-1)*interval/length
        elif n is not None:
            end = 1
        else:
            raise ValueError("Either 'interval' or 'n' must be specified.")
        
        self.anchors = np.linspace(0, end, n)
        return None
    
    def __hash__(self) -> int:
        return hash((id(self), self._updates))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}<{hex(id(self))}>"
    
    def fit(self, coords:np.ndarray, s=None):
        """
        Fit spline model using a list of coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates. Must be (N, 3).
        s : float, optional
            Total variation , by default None
        """        
        npoints = coords.shape[0]
        if npoints < 4:
            lin = interp1d(np.linspace(0, 1, npoints), coords.T)
            coords = lin(np.linspace(0,1,4)).T
        self._tck, self._u = splprep(coords.T, k=self._k, s=s)
        self._updates += 1
        self._anchors = None # Anchor should be deleted after spline is updated
        return self
    
    def distances(self, u: Iterable[float]=None) -> np.ndarray:
        if u is None:
            u = self.anchors
        length = self.length()
        return length * np.asarray(u)

    def __call__(self, u:np.ndarray|float=None, der:int=0) -> np.ndarray:
        if u is None:
            u = self.anchors
        if np.isscalar(u):
            coord = splev([u], self._tck, der=der)
            return np.concatenate(coord).astype(np.float32)
        else:
            coords = splev(u, self._tck, der=der)
            return np.stack(coords, axis=1).astype(np.float32)

    def partition(self, n:int, der:int=0):
        u = np.linspace(0, 1, n)
        return self(u, der)

    def length(self, start:float=0, stop:float=1, nknots:int=100) -> nm:
        """
        Approximate the length of B-spline between [start, stop] by partitioning
        the spline with 'nknots' knots.
        """
        u = np.linspace(start, stop, nknots)
        dz, dy, dx = map(np.diff, splev(u, self._tck, der=0))
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    
    def curvature(self, u=None):
        if u is None:
            u = self.anchors
        # https://en.wikipedia.org/wiki/Curvature#Space_curves
        dz, dy, dx = self(u, 1).T
        ddz, ddy, ddx = self(u, 2).T
        a = (ddz*dy-ddy*dz)**2 + (ddx*dz-ddz*dx)**2 + (ddy*dx-ddx*dy)**2
        return np.sqrt(a)/(dx**2+dy**2+dz**2)**1.5/self.scale # TODO: not /scale 
    
    def to_dict(self) -> dict:
        t = self.tck[0]
        c = self.tck[1]
        k = self.tck[2]
        u = self.u
        scale = self.scale
        return {"t": t.tolist(), 
                "c": {"z": c[0].tolist(),
                      "y": c[1].tolist(),
                      "x": c[2].tolist()},
                "k": k,
                "u": u.tolist(),
                "scale": scale}
    
    @classmethod
    def from_dict(cls, d: dict):
        self = cls(d["scale"], d["k"])
        t = np.array(d["t"])
        c = [np.array(d["c"][k]) for k in "zyx"]
        k = int(d["k"])
        self._tck = (t, c, k)
        self._u = d["u"]
        return self
    
    def to_json(self, path: str):
        path = str(path)
        
        with open(path, mode="w") as f:
            json.dump(self.to_dict(), f)
        
        return None
    
    def rotation_matrix(self, u=None, center=None, inverse:bool=False) -> np.ndarray:
        """
        Calculate list of Affine transformation matrix along spline, which correcpond to
        the orientation of spline curve.

        Parameters
        ----------
        u : array-like
            Positions. Between 0 and 1.
        center : array-like, optional
            If not provided, rotation will be executed around the origin. If an array is provided,
            it will be considered as the coordinates of rotation center. This is useful for 
            rotating images.
        inverse : bool, default is False
            If True, rotation matrix will be inversed.
        """        
        if u is None:
            u = self.anchors
        ds = self(u, 1)
        # matrix_func = _vector_to_inv_rotation_matrix if inverse else _vector_to_rotation_matrix
        matrix_func = _vector_to_rotation_matrix
        if np.isscalar(u):
            out = matrix_func(ds)
        else:
            out = np.stack([matrix_func(ds0) for ds0 in ds], axis=0)
        
        if inverse:
            out = np.linalg.inv(out)
            
        if center is not None:
            dz, dy, dx = center
            translation_0 = np.array([[1., 0., 0., dz],
                                      [0., 1., 0., dy],
                                      [0., 0., 1., dx],
                                      [0., 0., 0., 1.]],
                                     dtype=np.float32)
            
            translation_1 = np.array([[1., 0., 0., -dz],
                                      [0., 1., 0., -dy],
                                      [0., 0., 1., -dx],
                                      [0., 0., 0.,  1.]],
                                     dtype=np.float32)
            
            out = translation_0 @ out @ translation_1
        
        return out
    
    def local_cartesian(self,
                        shape: tuple[int, int],
                        n_pixels: int,
                        u=None):
        """
        Generate local Cartesian coordinate systems that can be used for ``ndi.map_coordinates``.
        The result coordinate systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        shape : tuple of two int
            Vertical and horizontal length of Cartesian coordinates. Corresponds to zx axes.
        n_pixels : int
            Length of y axis.
        u : float
            Position on the spline at which local Cartesian coordinates will be built.
        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to vertical, longitudinal, horizontal and 
            dimensional axis.
        """        
        return self._get_local_coords(_cartesian_coords_2d, shape, u, n_pixels, self.scale)
    
    def local_cylindrical(self,
                          r_range: tuple[int, int],
                          position,
                          n_pixels):
        
        return self._get_local_coords(_polar_coords_2d, r_range, position, n_pixels, self.scale)
        
    
    def _get_local_coords(self,
                          map_func: Callable[[tuple], np.ndarray],
                          map_params:tuple, 
                          u: np.ndarray, 
                          n_pixels:int):
        if u is None:
            u = self.anchors
        ds = self(u, 1).astype(np.float32)
        len_ds = np.sqrt(sum(ds**2))
        dy = ds.reshape(-1, 1)/len_ds * np.linspace(-n_pixels/2+0.5, n_pixels/2-0.5, n_pixels)
        y_ax_coords = (self(u)/self.scale).reshape(1, -1) + dy.T
        dslist = np.stack([ds]*n_pixels, axis=0)
        map_ = map_func(*map_params)
        zeros = np.zeros(map_.shape[:-1], dtype=np.float32)
        ones = np.ones(map_.shape[:-1], dtype=np.float32)
        map_slice = np.stack([map_[..., 0], 
                              zeros,
                              map_[..., 1],
                              ones
                              ], axis=2) # V, H, D
        return _rot_with_vector(map_slice, y_ax_coords, dslist)

    def cartesian(self, 
                  shape: tuple[int, int], 
                  s_range: tuple[float, float] = (0, 1)
                  ) -> np.ndarray:
        
        return self._get_coords(_cartesian_coords_2d, shape, s_range)

    def cylindrical(self, 
                    r_range: tuple[int, int],
                    s_range: tuple[float, float] = (0, 1)
                    ) -> np.ndarray:
        
        return self._get_coords(_polar_coords_2d, r_range, s_range)
    
    def inv_cartesian(self,
                      points: np.ndarray):
        # TODO: (z,y,x) in straight image to world coordinate
        pass
    

    def _get_coords(self,
                    map_func: Callable[[tuple], np.ndarray],
                    map_params:tuple,
                    s_range:tuple[float, float]):
        s0, s1 = s_range
        length = self.length(start=s0, stop=s1)
        n_pixels = int(length/self.scale) + 1
        if n_pixels < 2:
            raise ValueError("Too short. Change 's_range'.")
        u = np.linspace(s0, s1, n_pixels)
        y_ax_coords = self(u)/self.scale # world coordinates of y-axis in spline coords system
        dslist = self(u, 1).astype(np.float32)
        map_ = map_func(*map_params)
        zeros = np.zeros(map_.shape[:-1], dtype=np.float32)
        ones = np.ones(map_.shape[:-1], dtype=np.float32)
        map_slice = np.stack([map_[..., 0], 
                              zeros,
                              map_[..., 1],
                              ones
                              ], axis=2) # V, S, H, D
        return _rot_with_vector(map_slice, y_ax_coords, dslist)

_V = slice(None) # vertical dimension
_S = slice(None) # longitudinal dimension along spline curve
_H = slice(None) # horizontal dimension
_D = slice(None, None, 1) # dimension of dimension (such as d=0: z, d=1: y,...)

@nb.njit(cache=True)
def _vector_to_rotation_matrix(ds: nb.float32[_D]) -> nb.float32[_D,_D]:
    yx = np.arctan2(-ds[2], ds[1])
    zy = np.arctan(ds[0]/np.abs(ds[1]))
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
    return np.ascontiguousarray(mx)

@nb.njit(cache=True)
def _rot_point_with_vector(point: nb.float32[_V,_H,_D], 
                           dr: nb.float32[_D]
                           ) -> nb.float32[_V,_H,_D]:
    mx = _vector_to_rotation_matrix(dr)
    
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
        slice_out = _rot_point_with_vector(maps, dr) # TODO: contiguous
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