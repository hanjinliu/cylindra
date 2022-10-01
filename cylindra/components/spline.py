from __future__ import annotations
from functools import lru_cache
from typing import Callable, Sequence, TypedDict, TYPE_CHECKING
import warnings
import numpy as np
import json
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from skimage.transform._warps import _linear_polar_mapping
from acryo import Molecules
from acryo.molecules import axes_to_rotator
from ..utils import ceilint, interval_divmod, roundint
from ..const import Mode, nm

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import ArrayLike

class Coords3D(TypedDict):
    """3D coordinates in list used in json."""
    
    z: list[float]
    y: list[float]
    x: list[float]

    
class SplineInfo(TypedDict):
    """Spline parameters used in json."""
    
    t: list[float]
    c: Coords3D
    k: int
    u: list[float]
    lims: tuple[float, float]


class Spline:
    """
    3D spline curve model with coordinate system. Anchor points can be set via ``anchor``
    property. Every time spline parameters or anchors are updated, hash value of Spline
    object will be changed, thus it is safe to map Spline object to some result along
    the corresponding curve.
    
    References
    ----------
    - Scipy document
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
    """    
    # global cache will be re-initialized every time spline curve is updated.
    _global_cache: tuple[str, ...] = ()
    
    # local cache will be re-initialized every time spline curve is updated or anchor
    # is changed.
    _local_cache: tuple[str, ...] = ()
    
    def __init__(
        self, 
        degree: int = 3, 
        *, 
        lims: tuple[float, float] = (0., 1.)
    ):
        self._tck: tuple[np.ndarray | None, list[np.ndarray] | None, int] = (None, None, degree)
        self._u: np.ndarray | None = None
        self._anchors = None
        
        # check lims
        _min, _max = lims
        if _min < 0 or _max > 1:
            raise ValueError(f"'lims' must fit in range of [0, 1] but got {list(lims)!r}.")
        self._lims = lims
    
    def copy(self, copy_cache: bool = True) -> Self:
        """
        Copy Spline object.

        Parameters
        ----------
        copy_cache : bool, default is True
            Also copy cached properties if true.

        Returns
        -------
        Spline
            Copied object.
        """
        new = self.__class__(degree=self.degree, lims=self._lims)
        new._tck = self._tck
        new._u = self._u
        new._anchors = self._anchors
        
        if copy_cache:
            for name in self._global_cache + self._local_cache:
                setattr(new, name, getattr(self, name, None))
        
        return new
    
    __copy__ = copy
    
    @property
    def knots(self) -> np.ndarray:
        """Spline knots."""
        return self._tck[0]
    
    @property
    def coeff(self) -> list[np.ndarray]:
        """Spline coefficient."""
        return self._tck[1]
    
    @property
    def degree(self) -> int:
        """Spline degree."""
        return self._tck[2]
    
    @property
    def params(self) -> np.ndarray:
        """Spline parameters."""
        return self._u
    
    def __eq__(self: Self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return False
        t0, c0, k0 = self._tck
        t1, c1, k1 = other._tck
        return (
            np.allclose(t0, t1) and 
            all(np.allclose(x, y) for x, y in zip(c0, c1)) and
            k0 == k1 and
            np.allclose(self._u, other._u) and
            np.allclose(self._lims, other._lims)
        )
    
    @classmethod
    def line(cls, start: ArrayLike, end: ArrayLike) -> Self:
        """
        Create a line spline.

        Parameters
        ----------
        start : array_like
            Start point of the line.
        end : array_like
            End point of the line.

        Returns
        -------
        Spline
            Line spline.
        """
        spl = cls()
        coords = np.stack([start, end], axis=0)
        return spl.fit_voa(coords)
    
    def translate(self, shift: tuple[nm, nm, nm]):
        new = self.copy()
        c = [x + s for x, s in zip(self.coeff, shift)]
        new._tck = (self.knots, c, self.degree)
        return new
        
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
    def anchors(self, positions: float | Sequence[float]) -> None:
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
    
    @property
    def inverted(self) -> bool:
        """Return true if spline is inverted."""
        return self._lims[0] > self._lims[1]

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
            n = max(ceilint(length/max_interval), self.degree) + 1
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
        return f"{self.__class__.__name__}[{start}:{end}]"

    
    def clip(self, start: float, stop: float) -> Self:
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
        Spline
            Clipped spline.
        """
        u0 = _linear_conversion(start, *self._lims)
        u1 = _linear_conversion(stop, *self._lims)
        new = self.__class__(degree=self.degree, lims=(u0, u1))
        new._tck = self._tck
        new._u = self._u
        return new
    
    
    def restore(self) -> Self:
        """
        Restore the original, not-clipped spline.

        Returns
        -------
        Spline
            Copy of the original spline.
        """
        original = self.__class__(degree=self.degree, lims=(0, 1))
        original._tck = self._tck
        original._u = self._u
        return original

    def fit_coa(
        self, 
        coords: ArrayLike,
        *,
        weight: ArrayLike = None,
        n: int = 256,
        min_radius: nm = 1.0,
        tol: float = 1e-2,
        max_iter: int = 100,
    ) -> Self:
        """
        Fit spline model to coordinates by "Curvature-Oriented Approximation".

        Parameters
        ----------
        coords : np.ndarray
            Coordinates. Must be (N, 3).
        weight : np.ndarray, optional
            Weight of each coordinate.
        n : int, default is 256
            Number of partition for curvature sampling.
        min_radius : nm, default is 1.0
            Minimum allowed curvature radius. Fitting iteration continues until curvature
            radii are larger at any sampled points.
        tol : float, default is 1e-2
            Tolerance of fitting. Fitting iteration continues until ratio of maximum
            curvature to curvature upper limit is larger than 1 - tol.
        max_iter : int, default is 100
            Maximum number of iteration. Fitting stops when exceeded.
        
        Returns
        -------
        Spline
            Spline fit to given coordinates.
        """        
        coords = np.asarray(coords)
        npoints = coords.shape[0]
        if npoints < 2:
            raise ValueError("npoins must be > 1.")
        elif npoints <= self.degree:
            degree = npoints - 1
        else:
            degree = self.degree
        if self.inverted:
            coords = coords[::-1]
            
        # initialize
        s = 0.
        smax = np.sum(np.var(coords, axis=0)) * npoints
        u = np.linspace(0, 1, n)
        niter = 0
        
        if degree == 1:
            # curvature is not defined for a linear spline curve
            self._tck, self._u = splprep(coords.T, k=degree, w=weight, s=0.)
        
        else:
            while True:
                niter += 1
                self._tck, self._u = splprep(coords.T, k=degree, w=weight, s=s)
                curvature = self.curvature(u)
                ratio = np.max(curvature) * min_radius
                if ratio < 1. - tol:  # curvature too small = underfit
                    smax = s
                    s /= 2
                elif 1. < ratio:  # curvature too large = overfit
                    s = s / 2 + smax / 2
                else:
                    break
                if niter > max_iter:
                    break
            
        del self.anchors  # Anchor should be deleted after spline is updated
        self.clear_cache(loc=True, glob=True)
        return self

    def fit_voa(
        self,
        coords: ArrayLike,
        *,
        weight: ArrayLike = None,
        variance: float = None
    ) -> Self:
        """
        Fit spline model to coordinates by "Variance-Oriented Approximation".
        
        This method conduct variance-based fitting, which is well-formulated by the function
        ``scipy.interpolate.splprep``. The fitting result confirms that total variance 
        between spline and given coordinates does not exceed the value ``variance``. 

        Parameters
        ----------
        coords : np.ndarray
            Coordinates. Must be (N, 3).
        weight : np.ndarray, optional
            Weight of each coordinate.
        variance : float, optional
            Total variation.
        
        Returns
        -------
        Spline
            Spline fit to given coordinates.
        """        
        coords = np.asarray(coords)
        npoints = coords.shape[0]
        if npoints < 2:
            raise ValueError("npoins must be > 1.")
        elif npoints <= self.degree:
            self._tck = self._tck[:2] + (npoints - 1,)
        if variance is None:
            s = None
        else:
            s = variance * npoints
        if self.inverted:
            coords = coords[::-1]
        self._tck, self._u = splprep(coords.T, k=self.degree, w=weight, s=s)
        del self.anchors  # Anchor should be deleted after spline is updated
        self.clear_cache(loc=True, glob=True)
        return self
    
    def shift_coa(
        self,
        positions: Sequence[float] | None = None,
        shifts: np.ndarray | None = None,
        n: int = 256,
        min_radius: nm = 1.0,
        tol = 1e-2,
        max_iter: int = 100,
    ):
        coords = self(positions)
        rot = self.get_rotator(positions)
        # insert 0 in y coordinates. 
        shifts = np.stack([shifts[:, 0], np.zeros(len(rot)), shifts[:, 1]], axis=1)
        coords += rot.apply(shifts)
        self.fit_coa(coords, n=n, min_radius=min_radius, tol=tol, max_iter=max_iter)
        return self

    def shift_voa(
        self,
        positions: Sequence[float] | None = None,
        shifts: np.ndarray | None = None,
        variance: float = None
    ) -> Self:
        """
        Fit spline model using a list of shifts in XZ-plane.

        Parameters
        ----------
        positions : sequence of float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.
        shifts : np.ndarray
            Shift from center in nm. Must be (N, 2).
        variance : float, optional
            Total variation, by default None
            
        Returns
        -------
        Spline
            Spline shifted by fitting to given coordinates.
        """        
        coords = self(positions)
        rot = self.get_rotator(positions)
        # insert 0 in y coordinates. 
        shifts = np.stack([shifts[:, 0], np.zeros(len(rot)), shifts[:, 1]], axis=1)
        coords += rot.apply(shifts)
        self.fit_voa(coords, variance=variance)
        return self

    
    def distances(self, positions: Sequence[float] = None) -> np.ndarray:
        """
        Get the distances from u=0.

        Parameters
        ----------
        positions : sequence of float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        np.ndarray
            Distances for each ``u``.
        """        
        if positions is None:
            positions = self.anchors
        length = self.length()
        return length * np.asarray(positions)


    def __call__(self, positions: np.ndarray | float = None, der: int = 0) -> np.ndarray:
        """
        Calculate coordinates (or n-th derivative) at points on the spline.

        Parameters
        ----------
        positions : np.ndarray or float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.
        der : int, default is 0
            ``der``-th derivative will be calculated.

        Returns
        -------
        np.ndarray
            Positions or vectors in (N, 3) shape.
        """        
        if positions is None:
            positions = self.anchors
        u0, u1 = self._lims
        if np.isscalar(positions):
            u_tr = _linear_conversion(np.array([positions]), u0, u1)
            coord = splev(u_tr, self._tck, der=der)
            out = np.concatenate(coord).astype(np.float32)
        else:
            u_tr = _linear_conversion(np.asarray(positions), u0, u1)
            coords = splev(u_tr, self._tck, der=der)
            out = np.stack(coords, axis=1).astype(np.float32)
        
        if u0 > u1 and der % 2 == 1:
            out = -out
        return out

    def partition(self, n: int, der: int = 0) -> np.ndarray:
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

    def invert(self) -> Self:
        """
        Invert the direction of spline.

        Returns
        -------
        Spline
            Inverted object
        """
        anchors = self._anchors
        inverted = self.clip(1., 0.)
        if anchors is not None:
            inverted.anchors = 1 - anchors[::-1]
        return inverted
    
    def curvature(self, positions: Sequence[float] = None) -> np.ndarray:
        """
        Calculate curvature of spline curve.

        Parameters
        ----------
        positions : sequence of float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        np.ndarray
            Array of curvature.
            
        References
        ----------
        - https://en.wikipedia.org/wiki/Curvature#Space_curves        
        """        
        
        if positions is None:
            positions = self.anchors
        
        dz, dy, dx = self(positions, 1).T
        ddz, ddy, ddx = self(positions, 2).T
        a = (ddz*dy - ddy*dz)**2 + (ddx*dz - ddz*dx)**2 + (ddy*dx - ddx*dy)**2
        return np.sqrt(a)/(dx**2 + dy**2 + dz**2)**1.5

    def curvature_radii(self, positions: Sequence[float] = None) -> np.ndarray:
        """Inverse of curvature."""
        return 1. / self.curvature(positions)
    
    def to_dict(self) -> SplineInfo:
        """Convert spline info into a dict."""
        t, c, k = self._tck
        u = self._u
        return {"t": t.tolist(), 
                "c": {"z": c[0].tolist(),
                      "y": c[1].tolist(),
                      "x": c[2].tolist()},
                "k": k,
                "u": u.tolist(),
                "lims": self._lims,
                }
    
    @classmethod
    def from_dict(cls: type[Self], d: SplineInfo) -> Self:
        """
        Construct a spline model from a dictionary.

        Parameters
        ----------
        d: dict
            Dictionary with keys {"t", "c", "k", "u", "lims"}.

        Returns
        -------
        Spline
            Spline object constructed from the dictionary.
        """
        self = cls(degree=d.get("k", 3), lims=d.get("lims", (0, 1)))
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
    
    @classmethod
    def from_json(cls, file_path: str) -> Self:
        """
        Construct a spline model from a json file.

        Parameters
        ----------
        file_path : str
            Path to json file.

        Returns
        -------
        Spline
            Spline object constructed from the json file.
        """
        file_path = str(file_path)
        
        with open(file_path, mode="r") as f:
            js = json.load(f)
        return cls.from_dict(js)
        
    def affine_matrix(
        self, 
        positions: Sequence[float] = None,
        center: Sequence[float] = None, 
        inverse: bool = False,
    ) -> np.ndarray:
        """
        Calculate list of Affine transformation matrix along spline, which correspond to
        the orientation of spline curve.

        Parameters
        ----------
        positions : array-like, (N,)
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
        if positions is None:
            positions = self.anchors
        ds = self(positions, 1)
        
        if ds.ndim == 1:
            ds = ds[np.newaxis]
        rot = axes_to_rotator(None, ds)
        if inverse:
            rot = rot.inv()
        out = np.zeros((len(rot), 4, 4), dtype=np.float32)
        out[:, :3, :3] = rot.as_matrix()
        out[:, 3, 3] = 1.
            
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
        if np.isscalar(positions):
            out = out[0]
        return out
    
    def get_rotator(
        self, 
        positions: Sequence[float] = None,
        inverse: bool = False,
    ) -> Rotation:
        """
        Calculate list of Affine transformation matrix along spline, which correspond to
        the orientation of spline curve.

        Parameters
        ----------
        positions : array-like, (N,)
            Positions. Between 0 and 1.
        inverse : bool, default is False
            If True, rotation matrix will be inversed.
            
        Returns
        -------
        Rotation
            Rotation object at each anchor.
        """        
        if positions is None:
            positions = self.anchors
        ds = self(positions, 1)
        out = axes_to_rotator(None, -ds)
        
        if inverse:
            out = out.inv()
        
        return out

    def local_cartesian(
        self,
        shape: tuple[int, int],
        n_pixels: int,
        u: float | Sequence[float] = None,
        scale: nm = 1.,
    ):
        """
        Generate local Cartesian coordinate systems that can be used for ``ndi.map_coordinates``.
        The result coordinate systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        shape : tuple of two int
            Vertical and horizontal length of Cartesian coordinates. Corresponds to zx axes.
        n_pixels : int
            Length of y axis in pixels.
        u : float, optional
            Position on the spline at which local Cartesian coordinates will be built.
        scale: nm, default is 1.0
            Scale of coordinates, i.e. spacing of the grid.
            
        Returns
        -------
        np.ndarray
            (D, V, S, H) shape. Each cooresponds to dimensional vertical, longitudinal and
            horizontal axis, which is ready to be used in ``ndi.map_coordinates``.
        """
        
        mole = self.anchors_to_molecules(u)
        coords = mole.cartesian_at(
            slice(None), 
            shape=(shape[0], n_pixels, shape[1]), 
            scale=scale
        )
        if np.isscalar(u):
            coords = coords[0]
        return coords
    
    def local_cylindrical(
        self,
        r_range: tuple[float, float],
        n_pixels: int,
        u: float = None,
        scale: nm = 1.,
    ):
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
        scale: nm, default is 1.0
            Scale of coordinates, i.e. spacing of the grid.
            
        Returns
        -------
        np.ndarray
            (D, V, S, H) shape. Each cooresponds to dimensional, radius, longitudinal and
            angle axis, which is ready to be used in ``ndi.map_coordinates``.
        """        
        if u is None:
            u = self.anchors
        ds = self(u, 1).astype(np.float32)
        len_ds = np.sqrt(sum(ds**2))
        dy = ds.reshape(-1, 1)/len_ds * np.linspace(-n_pixels / 2 + 0.5, n_pixels / 2 - 0.5, n_pixels)
        y_ax_coords = (self(u)/scale).reshape(1, -1) + dy.T
        dslist = np.stack([ds]*n_pixels, axis=0)
        map_ = _polar_coords_2d(*r_range)
        map_slice = _stack_coords(map_)
        out = _rot_with_vector(map_slice, y_ax_coords, dslist)
        return np.moveaxis(out, -1, 0)


    def cartesian(
        self, 
        shape: tuple[int, int], 
        s_range: tuple[float, float] = (0, 1),
        scale: nm = 1.,
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
            Range of spline. Spline coordinate system will be built between 
            ``spl[s_range[0]]`` and ``spl[s_range[1]]``.
        scale: nm, default is 1.0
            Scale of coordinates, i.e. spacing of the grid.

        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to vertical, longitudinal, horizontal and 
            dimensional axis.
        """        
        return self._get_coords(_cartesian_coords_2d, shape, s_range, scale)


    def cylindrical(
        self, 
        r_range: tuple[float, float],
        s_range: tuple[float, float] = (0, 1),
        scale: nm = 1.,
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
            Range of spline. Spline coordinate system will be built between 
            ``spl[s_range[0]]`` and ``spl[s_range[1]]``.
        scale: nm, default is 1.0
            Scale of coordinates, i.e. spacing of the grid.

        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to radius, longitudinal, angle and 
            dimensional axis.
        """   
        return self._get_coords(_polar_coords_2d, r_range, s_range, scale)


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
        positions = coords[:, 1]/self.length()
        s = self(positions)
        coords_ext = np.stack([
            coords[:, 0], 
            np.zeros(ncoords, dtype=np.float32),
            coords[:, 2], 
            ], axis=1)
        rot = self.get_rotator(positions)
        out = rot.apply(coords_ext) + s
                
        return out


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

    
    def world_to_y(
        self, 
        coords: np.ndarray,
        precision: nm = 0.2,
    ) -> np.ndarray:
        """
        Convert world coordinates into y-coordinate in spline coordinate system.
        
        .. warning::

            This conversion is not well-defined mathematically. Results only make
            sence when spline has low-curvature and all the coordinates are near
            the spline.

        Parameters
        ----------
        coords : (3,) or (N, 3) array
            World coordinates.
        precision : nm, default is 0.2
            Precision of y coordinate in nm.

        Returns
        -------
        np.ndarray
            Corresponding y-coordinates in spline coordinate system
        """
        if coords.ndim == 1:
            coords = coords[np.newaxis]
        length = self.length()
        u = np.linspace(0, 1, ceilint(length/precision))
        sample_points = self(u) # (N, 3)
        vector_map = sample_points.reshape(-1, 1, 3) - coords.reshape(1, -1, 3) # (S, N, 3)
        dist2_map = np.sum(vector_map**2, axis=2)
        argmins = np.argmin(dist2_map, axis=0).tolist()
        return u[argmins]

    def anchors_to_molecules(
        self, 
        positions: float | Sequence[float] | None = None,
        rotation: Sequence[float] | None = None,
    ) -> Molecules:
        """
        Convert coordinates of anchors to ``Molecules`` instance.
        
        Coordinates of anchors must be in range from 0 to 1. The y-direction of
        ``Molecules`` always points at the direction of spline and the z-
        direction always in the plane orthogonal to YX-plane.

        Parameters
        ----------
        positions : iterable of float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        Molecules
            Molecules object of points.
        """
        if positions is None:
            positions = self.anchors
        pos = self(positions)
        yvec = self(positions, der=1)
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

    def slice_along(
        self, 
        array: np.ndarray,
        s_range: tuple[float, float] = (0., 1.),
        order: int = 3,
        mode: str = Mode.constant,
        cval: float = 0.,
    ) -> np.ndarray:
        _, coords = self._get_y_ax_coords(s_range)
        from scipy import ndimage as ndi
        return ndi.map_coordinates(
            array,
            coords.T,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=order>1,
        )

    def _get_coords(
        self,
        map_func: Callable[[tuple], np.ndarray],
        map_params: tuple,
        s_range: tuple[float, float],
        scale: nm
        ):
        """
        Make coordinate system using function ``map_func`` and stack the same point cloud
        in the direction of the spline, in the range of ``s_range``.
        """
        u, y_ax_coords = self._get_y_ax_coords(s_range, scale)
        dslist = self(u, 1).astype(np.float32)
        map_ = map_func(*map_params)
        map_slice = _stack_coords(map_)
        out = _rot_with_vector(map_slice, y_ax_coords, dslist)
        return np.moveaxis(out, -1, 0)
    
    def _get_y_ax_coords(self, s_range: tuple[float, float], scale: nm):
        s0, s1 = s_range
        length = self.length(start=s0, stop=s1)
        stop_length, n_segs = interval_divmod(length, scale)
        n_pixels = n_segs + 1
        s2 = (s1 - s0) * stop_length/length + s0
        if n_pixels < 2:
            raise ValueError("Too short. Change 's_range'.")
        u = np.linspace(s0, s2, n_pixels)
        y = self(u) / scale  # world coordinates of y-axis in spline coords system
        return u, y


def _linear_conversion(u, start: float, stop: float):
    return (1 - u) * start + u * stop


def _rot_with_vector(maps: np.ndarray, ax_coords: np.ndarray, vectors: np.ndarray):
    rot = axes_to_rotator(None, vectors)
    mat = rot.as_matrix()
    out = np.einsum("nij,vhj->vnhi", mat, maps)
    out += ax_coords[np.newaxis, :, np.newaxis]
    return out
    

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
    stacked = np.stack([coords[..., 0], 
                        zeros,
                        coords[..., 1],
                        ], axis=2) # V, S, H, D
    return stacked