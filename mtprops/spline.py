from __future__ import annotations
from functools import lru_cache
from typing import Callable, Iterable, TypedDict, TYPE_CHECKING
import warnings
import numpy as np
import numba as nb
import json
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from skimage.transform._warps import _linear_polar_mapping
from .utils import ceilint, interval_divmod, oblique_meshgrid, roundint
from .const import nm
from .molecules import Molecules, axes_to_rotator

class Coords3D(TypedDict):
    z: list[float]
    y: list[float]
    x: list[float]

    
class SplineInfo(TypedDict):
    t: list[float]
    c: Coords3D
    k: int
    u: list[float]
    scale: float


class Spline:
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
    # global cache will be re-initialized every time spline curve is updated.
    _global_cache: tuple[str] = ()
    
    # local cache will be re-initialized every time spline curve is updated or anchor
    # is changed.
    _local_cache: tuple[str] = ()
    
    def __init__(self, scale: float = 1.0, k: int = 3, *, lims: tuple[float, float] = (0., 1.)):
        self._tck = None
        self._u = None
        self.scale = scale
        self._k = k
        self._anchors = None
        
        # check lims
        _min, _max = lims
        if _min < 0 or _max > 1:
            raise ValueError(f"'lims' must fit in range of [0, 1] but got {list(lims)!r}.")
        self._lims = lims
    
    def copy(self, copy_cache: bool = True) -> Spline:
        """
        Copy Spline3D object.

        Parameters
        ----------
        copy_cache : bool, default is True
            Also copy cached properties if true.

        Returns
        -------
        Spline3D
            Copied object.
        """
        new = self.__class__(self.scale, self.k, lims=self._lims)
        new._tck = self._tck
        new._u = self._u
        new._anchors = self._anchors
        
        if copy_cache:
            for name in self._global_cache + self._local_cache:
                setattr(new, name, getattr(self, name, None))
        
        return new
    
    __copy__ = copy
    
    @property
    def tck(self) -> tuple[np.ndarray, list[np.ndarray], int]:
        return self._tck
    
    @property
    def u(self) -> np.ndarray:
        return self._u
    
    @property
    def k(self) -> int:
        return self._k
    
    def __eq__(self, other: Spline):
        if not isinstance(other, self.__class__):
            return False
        t0, c0, k0 = self.tck
        t1, c1, k1 = other.tck
        return (
            np.allclose(t0, t1) and 
            all(np.allclose(x, y) for x, y in zip(c0, c1)) 
            and k0 == k1
            )
    
    def clear_cache(self, loc: bool = True, glob: bool = True):
        """
        Clear caches stored on the spline.

        Parameters
        ----------
        loc : bool, default is True
            Clear local cache if true.
        glob : bool, default is True
            Clear global cache if true.
        """
        if loc:
            for name in self._local_cache:
                setattr(self, name, None)
        if glob:
            for name in self._global_cache:
                setattr(self, name, None)
        
    @property
    def anchors(self) -> np.ndarray:
        """Local anchors along spline."""
        if self._anchors is None:
            raise ValueError("Anchor has not been set yet.")
        return self._anchors
    
    @anchors.setter
    def anchors(self, positions: float | Iterable[float]) -> None:
        positions = np.atleast_1d(np.asarray(positions, dtype=np.float32))
        if positions.ndim != 1:
            raise TypeError(f"Could not convert positions into 1D array.")
        elif positions.min() < 0 or positions.max() > 1:
            msg = f"Anchor positions should be set between 0 and 1. Otherwise spline " \
                  f"curve does not fit well."
            warnings.warn(msg, UserWarning)
        self._anchors = positions
        self.clear_cache(loc=True, glob=False)
        return None
    
    @anchors.deleter
    def anchors(self) -> None:
        self._anchors = None
        self.clear_cache(loc=True, glob=False)
        return None

    def make_anchors(
        self, 
        interval: nm | None = None,
        n: int | None = None,
        max_interval: nm | None = None
    ) -> None:
        """
        Make anchor points at constant intervals. Either interval, number of anchor or the 
        maximum interval between anchors can be specified.

        Parameters
        ----------
        interval : nm, optional
            Interval between anchor points.
        n : int, optional
            Number of anchor points, including both ends.
        max_interval: nm, optional
            Spline will be split by as little anchors as possible but interval between anchors
            will not be larger than this. The number of anchors are also guaranteed to be larger
            than spline order.
        """        
        length = self.length()
        if interval is not None:
            stop, n_segs = interval_divmod(length, interval)
            end = stop/length
            n = n_segs + 1
        elif n is not None:
            end = 1
        elif max_interval is not None:
            n = max(ceilint(length/max_interval), self.k) + 1
            end = 1
        else:
            raise ValueError("Either 'interval' or 'n' must be specified.")
        
        self.anchors = np.linspace(0, end, n)
        return None


    def __repr__(self) -> str:
        """Use start/end points to describe a spline."""
        start, end = self(self._lims)
        start = "({:.1f}, {:.1f}, {:.1f})".format(*start)
        end = "({:.1f}, {:.1f}, {:.1f})".format(*end)
        return f"{self.__class__.__name__}({start} to {end})"

    
    def clip(self, start: float, stop: float) -> Spline:
        """
        Clip spline and generate a new one.
        
        This method does not convert spline bases. ``_lims`` is updated instead.

        Parameters
        ----------
        start : float
            New starting position.
        stop : float
            New stopping position.

        Returns
        -------
        Spline3D
            Clipped spline.
        """
        u0 = _linear_conversion(start, *self._lims)
        u1 = _linear_conversion(stop, *self._lims)
        new = self.__class__(self.scale, self.k, lims=(u0, u1))
        new._tck = self._tck
        new._u = self._u
        return new
    

    def fit(self, coords: np.ndarray, w: np.ndarray = None, s: float = None) -> Spline:
        """
        Fit spline model using a list of coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates. Must be (N, 3).
        w : np.ndarray, optional
            Weight of each coordinate.
        s : float, optional
            Total variation , by default None
        """        
        npoints = coords.shape[0]
        if npoints < 2:
            raise ValueError("npoins must be > 1.")
        elif npoints <= self._k:
            self._k = npoints - 1
        self._tck, self._u = splprep(coords.T, k=self._k, w=w, s=s)
        del self.anchors # Anchor should be deleted after spline is updated
        self.clear_cache(loc=True, glob=True)
        return self
    

    def shift_fit(self, u: Iterable[float] = None, shifts: np.ndarray = None, s: float = None) -> Spline:
        """
        Fit spline model using a list of shifts in XZ-plane.

        Parameters
        ----------
        u : Iterable[float], optional
            Positions. Between 0 and 1. If not given, anchors are used instead.
        shifts : np.ndarray
            Shift from center in nm. Must be (N, 2).
        w : np.ndarray, optional
            Weight of each coordinate.
        s : float, optional
            Total variation, by default None
        """        
        coords = self(u)
        rot = self.get_rotator(u)
        shifts = np.stack([shifts[:, 0], np.zeros(len(rot)), shifts[:, 1]], axis=1)
        coords += rot.apply(shifts)
        self.fit(coords, s=s)
        return self

    
    def distances(self, u: Iterable[float] = None) -> np.ndarray:
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


    def __call__(self, u: np.ndarray | float = None, der: int = 0) -> np.ndarray:
        """
        Calculate coordinates (or n-th derivative) at points on the spline.

        Parameters
        ----------
        u : np.ndarray or float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.
        der : int, default is 0
            ``der``-th derivative will be calculated.

        Returns
        -------
        np.ndarray
            Positions or vectors in (N, 3) shape.
        """        
        if u is None:
            u = self.anchors
        u0, u1 = self._lims
        if np.isscalar(u):
            u_tr = _linear_conversion(np.array([u]), u0, u1)
            coord = splev(u_tr, self._tck, der=der)
            out = np.concatenate(coord).astype(np.float32)
        else:
            u_tr = _linear_conversion(np.asarray(u), u0, u1)
            coords = splev(u_tr, self._tck, der=der)
            out = np.stack(coords, axis=1).astype(np.float32)
        
        if u0 > u1 and der % 2 == 1:
            out = -out
        return out

    def partition(self, n: int, der: int = 0):
        u = np.linspace(0, 1, n)
        return self(u, der)


    def length(self, start: float = 0, stop: float = 1, nknots: int = 256) -> nm:
        """
        Approximate the length of B-spline between [start, stop] by partitioning
        the spline with 'nknots' knots. nknots=256 is large enough for most cases.
        """
        u = np.linspace(start, stop, nknots)
        u_tr = _linear_conversion(u, *self._lims)
        dz, dy, dx = map(np.diff, splev(u_tr, self._tck, der=0))
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

    def invert(self) -> Spline:
        """
        Invert the direction of spline.

        Returns
        -------
        Spline3D
            Inverted object
        """
        anchors = self._anchors
        inverted = self.clip(1., 0.)
        if anchors is not None:
            inverted.anchors = 1 - anchors[::-1]
        return inverted
    
    def curvature(self, u: Iterable[float] = None) -> np.ndarray:
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
        a = (ddz*dy - ddy*dz)**2 + (ddx*dz - ddz*dx)**2 + (ddy*dx - ddx*dy)**2
        return np.sqrt(a)/(dx**2 + dy**2 + dz**2)**1.5 / self.scale

    
    def to_dict(self) -> SplineInfo:
        """
        Convert spline info into a dict.
        """        
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
                "scale": scale
                }
    
    @classmethod
    def from_dict(cls, d: SplineInfo):
        self = cls(d["scale"], d["k"])
        t = np.asarray(d["t"])
        c = [np.asarray(d["c"][k]) for k in "zyx"]
        k = roundint(d["k"])
        self._tck = (t, c, k)
        self._u = np.asarray(d["u"])
        return self
    

    def to_json(self, file_path: str) -> None:
        """
        Save spline model in a json format.

        Parameters
        ----------
        file_path : str
            Path to the file.
        """
        file_path = str(file_path)
        
        with open(file_path, mode="w") as f:
            json.dump(self.to_dict(), f, indent=4, separators=(", ", ": "))
        
        return None
    

    def affine_matrix(self, 
                      u: Iterable[float] = None,
                      center: Iterable[float] = None, 
                      inverse: bool = False) -> np.ndarray:
        """
        Calculate list of Affine transformation matrix along spline, which correspond to
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
        np.ndarray (N, 4, 4)
            3D array of matrices, where the first dimension corresponds to each point.
        """        
        if u is None:
            u = self.anchors
        ds = self(u, 1)
        
        if np.isscalar(u):
            out = _vector_to_rotation_matrix(ds)
        else:
            out = np.stack([_vector_to_rotation_matrix(ds0) for ds0 in ds], axis=0)
        
        if inverse:
            out = np.linalg.inv(out)
            
        if center is not None:
            dz, dy, dx = center
            # center to corner
            translation_0 = np.array([[1., 0., 0., dz],
                                      [0., 1., 0., dy],
                                      [0., 0., 1., dx],
                                      [0., 0., 0., 1.]],
                                     dtype=np.float32)
            # corner to center
            translation_1 = np.array([[1., 0., 0., -dz],
                                      [0., 1., 0., -dy],
                                      [0., 0., 1., -dx],
                                      [0., 0., 0.,  1.]],
                                     dtype=np.float32)
            
            out = translation_0 @ out @ translation_1
        
        return out
    
    def get_rotator(
        self, 
        u: Iterable[float] = None,
        inverse: bool = False
    ) -> Rotation:
        """
        Calculate list of Affine transformation matrix along spline, which correspond to
        the orientation of spline curve.

        Parameters
        ----------
        u : array-like, (N, )
            Positions. Between 0 and 1.
        inverse : bool, default is False
            If True, rotation matrix will be inversed.
            
        Returns
        -------
        Rotation
            Rotation object at each anchor.
        """        
        if u is None:
            u = self.anchors
        ds = self(u, 1)
        out = axes_to_rotator(None, -ds)
        
        if inverse:
            out = out.inv()
        
        return out

    
    def local_cartesian(self,
                        shape: tuple[int, int],
                        n_pixels: int,
                        u: float = None):
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
                          r_range: tuple[float, float],
                          n_pixels: int,
                          u: float = None):
        """
        Generate local cylindrical coordinate systems that can be used for ``ndi.map_coordinates``.
        The result coordinate systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        r_range : tuple[float, float]
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
                          map_params: tuple, 
                          u: np.ndarray, 
                          n_pixels: int):
        if u is None:
            u = self.anchors
        ds = self(u, 1).astype(np.float32)
        len_ds = np.sqrt(sum(ds**2))
        dy = ds.reshape(-1, 1)/len_ds * np.linspace(-n_pixels/2+0.5, n_pixels/2-0.5, n_pixels)
        y_ax_coords = (self(u)/self.scale).reshape(1, -1) + dy.T
        dslist = np.stack([ds]*n_pixels, axis=0)
        map_ = map_func(*map_params)
        map_slice = _stack_coords(map_)
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
                    r_range: tuple[float, float],
                    s_range: tuple[float, float] = (0, 1)
                    ) -> np.ndarray:
        """
        Generate a cylindrical coordinate system along spline that can be used for
        ``ndi.map_coordinate``. Note that this coordinate system is distorted, thus
        does not reflect real geometry (such as distance and derivatives).

        Parameters
        ----------
        r_range : tuple[float, float]
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


    def cartesian_to_world(self, coords: np.ndarray) -> np.ndarray:
        """
        Inverse Cartesian coordinate mapping, (z', y', x') to world coordinate.

        Parameters
        ----------
        coords : np.ndarray
            Spline Cartesian coordinates. All the coordinates must be in nm unit.

        Returns
        -------
        np.ndarray
            World coordinates.
        """        
        ncoords = coords.shape[0]
        u = coords[:, 1]/self.length()
        s = self(u)
        ds = self(u, 1)
        
        coords_ext = np.stack([coords[:, 0], 
                               np.zeros(ncoords, dtype=np.float32),
                               coords[:, 2], 
                               -np.zeros(ncoords, dtype=np.float32)],
                              axis=1)
        s_ext = np.concatenate([s, np.zeros((ncoords, 1), dtype=np.float32)], axis=1)
        for crd, s0, ds0 in zip(coords_ext, s_ext, ds):
            mtx = np.linalg.inv(_vector_to_rotation_matrix(ds0))
            crd[:] = mtx.dot(crd) + s0
        
        return coords_ext[:, :3]


    def cylindrical_to_world(self, coords: np.ndarray) -> np.ndarray:
        """
        Inverse cylindrical coordinate mapping, (r, y, angle) to world coordinate.

        Parameters
        ----------
        coords : np.ndarray
            Cylindrical coordinates. "r" and "y" must be in scale of "nm", while angle
            must be in radian.

        Returns
        -------
        np.ndarray
            World coordinates.
        """        
        radius = coords[:, 0]
        y = coords[:, 1]
        theta = coords[:, 2]
        cart_coords = np.stack([radius*np.sin(theta), 
                                y, 
                                radius*np.cos(theta)],
                               axis=1)
        
        return self.cartesian_to_world(cart_coords)

    
    # def world_to_cylindrical(self, 
    #                          coords: np.ndarray,
    #                          precision: float = 1e-3,
    #                          angle_tol: float = 1e-2) -> Spline:
    #     # WIP
    #     u = np.linspace(0, 1, 1/precision)
    #     sample_points = self(u) # (N, 3)
    #     vector_map = sample_points.reshape(-1, 1, 3) - coords.reshape(1, -1, 3) # (S, N, 3)
    #     dist2_map = np.sum(vector_map**2, axis=2)
    #     argmins = np.argmin(dist2_map, axis=0).tolist()
    #     argmin_pos = u[argmins]
    #     s = self(argmin_pos)
    #     ds = self(argmin_pos, der=1)
    #     norm_vector = coords - s
    #     inner = np.tensordot(ds, norm_vector, [(1,), (1,)])
    #     theta = np.arccos(
    #         inner/np.sqrt(np.sum(ds**2, axis=1)*np.sum(norm_vector**2, axis=1))
    #         )
    #     valid = np.abs(np.abs(theta) - np.pi/2) < angle_tol


    def anchors_to_molecules(
        self, 
        u: Iterable[float] | None = None,
        rotation: Iterable[float] | None = None
    ) -> Molecules:
        """
        Convert coordinates of anchors to ``Molecules`` instance.
        
        Coordinates of anchors must be in range from 0 to 1. The y-direction of
        ``Molecules`` always points at the direction of spline and the z-
        direction always in the plane orthogonal to YX-plane.

        Parameters
        ----------
        u : Iterable[float] | None
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        Molecules
            Molecules object of points.
        """
        if u is None:
            u = self.anchors
        pos = self(u)
        yvec = self(u, der=1)
        rot = axes_to_rotator(None, yvec)
        if rotation is not None:
            rotvec = np.zeros((len(rot), 3), dtype=np.float32)
            rotvec[:, 1] = rotation
            rot = rot * Rotation.from_rotvec(rotvec)
        return Molecules(pos=pos, rot=rot)


    def cylindrical_to_molecules(self, coords: np.ndarray) -> Molecules:
        """
        Convert coordinates of points near the spline to ``Molecules`` instance.
        
        Coordinates of points must be those in spline cylindrical coordinate system.

        Parameters
        ----------
        coords : (N, 3) array
            Spline cylindrical coordinates of points.

        Returns
        -------
        Molecules
            Molecules object of points.
        """        
        world_coords = self.cylindrical_to_world(coords)
        
        # world coordinates of the projection point of coords onto the spline
        u = coords[:, 1]/self.length()
        ycoords = self(u)
        zvec = world_coords - ycoords
        yvec = self(u, der=1)
        return Molecules.from_axes(pos=world_coords, z=zvec, y=yvec)


    def _get_coords(self,
                    map_func: Callable[[tuple], np.ndarray],
                    map_params: tuple,
                    s_range: tuple[float, float]):
        """
        Make coordinate system using function ``map_func`` and stack the same point cloud
        in the direction of the spline, in the range of ``s_range``.
        """
        s0, s1 = s_range
        length = self.length(start=s0, stop=s1)
        stop_length, n_segs = interval_divmod(length, self.scale)
        n_pixels = n_segs + 1
        s2 = (s1 - s0) * stop_length/length + s0
        if n_pixels < 2:
            raise ValueError("Too short. Change 's_range'.")
        u = np.linspace(s0, s2, n_pixels)
        y_ax_coords = self(u)/self.scale  # world coordinates of y-axis in spline coords system
        dslist = self(u, 1).astype(np.float32)
        map_ = map_func(*map_params)
        map_slice = _stack_coords(map_)
        return _rot_with_vector(map_slice, y_ax_coords, dslist)


def _linear_conversion(u, start: float, stop: float):
    return (1 - u) * start + u * stop

_V = slice(None) # vertical dimension
_S = slice(None) # longitudinal dimension along spline curve
_H = slice(None) # horizontal dimension
_D = slice(None, None, 1) # dimension of dimension (such as d=0: z, d=1: y,...)

@nb.njit(cache=True)
def _vector_to_rotation_matrix(ds: nb.float32[_D]) -> nb.float32[_D,_D]:
    xy = np.arctan2(ds[2], -ds[1])
    zy = np.arctan(-ds[0]/np.abs(ds[1]))
    cos = np.cos(xy)
    sin = np.sin(xy)
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
    """
    Rotate 'point' with vector 'dr'.
    """    
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
    ax_coords = np.ascontiguousarray(ax_coords)
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

@lru_cache(maxsize=12)
def _polar_coords_2d(r_start: float, r_stop: float, center=None) -> np.ndarray:
    n_angle = roundint((r_start + r_stop) * np.pi)
    n_radius = roundint(r_stop - r_start)
    r_, ang_ = np.indices((n_radius, n_angle))
    r_ = r_ + (r_start + r_stop - n_radius + 1)/2
    output_coords = np.column_stack([r_.ravel(), ang_.ravel()])
    if center is None:
        center = [0, 0]
    coords = _linear_polar_mapping(np.array(output_coords), 
                                   k_angle=n_angle/2/np.pi, 
                                   k_radius=1,
                                   center=center[::-1]
                                   ).astype(np.float32)
    coords = coords.reshape(n_radius, n_angle, 2) # V, H, 2
    
    # Here, the first coordinate should be theta=0, and theta moves anti-clockwise
    coords[:] = np.flip(coords, axis=0) # flip around y=0
    coords[:] = np.flip(coords, axis=2) # flip around y=x
    return coords
    

@lru_cache(maxsize=12)
def _cartesian_coords_2d(lenv: int, lenh: int):
    v, h = np.indices((lenv, lenh), dtype=np.float32)
    v -= lenv/2 - 0.5
    h -= lenh/2 - 0.5
    return np.stack([v, h], axis=2) # V, H, 2


def _stack_coords(coords: np.ndarray): # V, H, D
    shape = coords.shape[:-1]
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    stacked = np.stack([coords[..., 0], 
                        zeros,
                        -coords[..., 1],
                        ones
                        ], axis=2) # V, S, H, D
    return stacked
