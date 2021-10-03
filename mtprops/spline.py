from __future__ import annotations
from typing import Callable, Iterable
import warnings
import numpy as np
import numba as nb
import json
from scipy.interpolate import splprep, splev, interp1d
from skimage.transform._warps import _linear_polar_mapping
from .utils import interval_divmod
from .const import nm

class Spline3D:
    """
    3D spline curve model with coordinate system. Anchor points can be set via ``anchor``
    property. Every time spline parameters or anchors are updated, hash value of Spline3D
    object will be changed, thus it is safe to map Spline3D object to some result along
    the corresponding curve.
    
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
        """
        Make anchor points at constant intervals. Either interval or the number of anchor
        points can be specified.

        Parameters
        ----------
        interval : nm, optional
            Interval between anchor points.
        n : int, optional
            Number of anchor points, including both ends.
        """        
        length = self.length()
        if interval is not None:
            stop, n_segs = interval_divmod(length, interval)
            end = stop/length
            n = n_segs + 1
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
    
    def fit(self, coords:np.ndarray, s=None) -> Spline3D:
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
        """
        Get the distances from u=0.

        Parameters
        ----------
        u : Iterable[float], optional
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        np.ndarray
            Distances for each ``u``.
        """        
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
    
    def curvature(self, u: Iterable[float]=None) -> np.ndarray:
        """
        Calculate curvature of spline curve.

        Parameters
        ----------
        u : Iterable[float], optional
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        np.ndarray
            Array of curvature.
            
        References
        ----------
        - https://en.wikipedia.org/wiki/Curvature#Space_curves
        """        
        
        if u is None:
            u = self.anchors
        
        dz, dy, dx = self(u, 1).T
        ddz, ddy, ddx = self(u, 2).T
        a = (ddz*dy-ddy*dz)**2 + (ddx*dz-ddz*dx)**2 + (ddy*dx-ddx*dy)**2
        return np.sqrt(a)/(dx**2+dy**2+dz**2)**1.5/self.scale
    
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
        t = np.asarray(d["t"])
        c = [np.asarray(d["c"][k]) for k in "zyx"]
        k = int(d["k"])
        self._tck = (t, c, k)
        self._u = np.asarray(d["u"])
        return self
    
    def to_json(self, file_path: str):
        file_path = str(file_path)
        
        with open(file_path, mode="w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(",", ": "))
        
        return None
    
    def rotation_matrix(self, u: Iterable[float]=None, center=None, inverse:bool=False) -> np.ndarray:
        """
        Calculate list of Affine transformation matrix along spline, which correcpond to
        the orientation of spline curve.

        Parameters
        ----------
        u : array-like, (N, )
            Positions. Between 0 and 1.
        center : array-like, optional
            If not provided, rotation will be executed around the origin. If an array is provided,
            it will be considered as the coordinates of rotation center. This is useful for 
            rotating images.
        inverse : bool, default is False
            If True, rotation matrix will be inversed.
            
        Returns
        -------
        np.ndarray (N, M, M)
            3D array of matrices, where the first dimension corresponds to each point.
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
            Length of y axis in pixels.
        u : float
            Position on the spline at which local Cartesian coordinates will be built.
        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to vertical, longitudinal, horizontal and 
            dimensional axis.
        """        
        return self._get_local_coords(_cartesian_coords_2d, shape, u, n_pixels)
    
    def local_cylindrical(self,
                          r_range: tuple[int, int],
                          n_pixels,
                          u=None):
        """
        Generate local cylindrical coordinate systems that can be used for ``ndi.map_coordinates``.
        The result coordinate systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        r_range : tuple of two int
            Lower and upper bound of radius.
        n_pixels : int
            Length of y axis in pixels.
        u : float
            Position on the spline at which local cylindrical coordinates will be built.
            
        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to radius, longitudinal, angle and 
            dimensional axis.
        """        
        return self._get_local_coords(_polar_coords_2d, r_range, u, n_pixels)
        
    
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
        """
        Generate a Cartesian coordinate system along spline that can be used for
        ``ndi.map_coordinate``. Note that this coordinate system is distorted, thus
        does not reflect real geometry (such as distance and derivatives).

        Parameters
        ----------
        shape : tuple[int, int]
            The ZX-shape of output coordinate system. Center of the array will be
            spline curve itself after coodinate transformation.
        s_range : tuple[float, float], default is (0, 1)
            Range of spline domain.

        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to vertical, longitudinal, horizontal and 
            dimensional axis.
        """        
        return self._get_coords(_cartesian_coords_2d, shape, s_range)

    def cylindrical(self, 
                    r_range: tuple[int, int],
                    s_range: tuple[float, float] = (0, 1)
                    ) -> np.ndarray:
        """
        Generate a cylindrical coordinate system along spline that can be used for
        ``ndi.map_coordinate``. Note that this coordinate system is distorted, thus
        does not reflect real geometry (such as distance and derivatives).

        Parameters
        ----------
        r_range : tuple[int, int]
            Range of radius in pixels. r=0 will be spline curve itself after coodinate 
            transformation.
        s_range : tuple[float, float], default is (0, 1)
            Range of spline domain.

        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to radius, longitudinal, angle and 
            dimensional axis.
        """   
        return self._get_coords(_polar_coords_2d, r_range, s_range)
    
    def inv_cartesian(self,
                      coords: np.ndarray,
                      shape: tuple[int, int, int]):
        
        # TODO: implement for loop in numba
        ncoords = coords.shape[0]
        u = coords[:, 1]/self.scale/shape[1]
        s = self(u)
        ds = self(u, 1)
        
        coords_ext = np.stack([coords[:, 0] - shape[0] * self.scale / 2, 
                               np.zeros(ncoords, dtype=np.float32),
                               coords[:, 2] - shape[2] * self.scale / 2, 
                               np.zeros(ncoords, dtype=np.float32)],
                              axis=1)
        s_ext = np.concatenate([s, np.zeros((ncoords, 1), dtype=np.float32)], axis=1)
        for crd, s0, ds0 in zip(coords_ext, s_ext, ds):
            mtx = np.linalg.inv(_vector_to_rotation_matrix(ds0))
            crd[:] = mtx.dot(crd) + s0
        
        return coords_ext[:, :3]
    
    def inv_cylindrical(self, 
                        coords: np.ndarray,
                        shape: tuple[int, int, int],
                        rmin: int):
        radius = coords[:, 0] + rmin * self.scale
        y = coords[:, 1]
        theta = coords[:, 2]/shape[2]*np.pi
        cart_coords = np.stack([radius*np.cos(theta), 
                                y, 
                                radius*np.sin(theta)],
                               axis=1)
        
        return self.inv_cartesian(cart_coords, (0, shape[1], 0))

    def _get_coords(self,
                    map_func: Callable[[tuple], np.ndarray],
                    map_params:tuple,
                    s_range:tuple[float, float]):
        s0, s1 = s_range
        length = self.length(start=s0, stop=s1)
        stop_length, n_segs = interval_divmod(length, self.scale)
        n_pixels = n_segs + 1
        s2 = (s1 - s0) * stop_length/length + s0
        if n_pixels < 2:
            raise ValueError("Too short. Change 's_range'.")
        u = np.linspace(s0, s2, n_pixels)
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
    maps = np.ascontiguousarray(maps)
    vectors = np.ascontiguousarray(vectors)
    
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
    coords = coords.reshape(n_radius, n_angle, 2) # V, H, 2
    return np.flip(coords, axis=1)

def _cartesian_coords_2d(lenv, lenh):
    v, h = np.indices((lenv, lenh), dtype=np.float32)
    v -= (lenv/2 - 0.5)
    h -= (lenh/2 - 0.5)
    return np.stack([v, h], axis=2) # V, H, 2