from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Sequence,
    TypeVar,
    overload,
)

import numpy as np
from acryo import Molecules
from acryo.molecules import axes_to_rotator
from scipy.interpolate import splev, splprep, splrep
from scipy.spatial.transform import Rotation

from cylindra.components._base import BaseComponent
from cylindra.components.spline._config import SplineConfig
from cylindra.components.spline._props import SplineProps
from cylindra.components.spline._types import SplineFitResult, SplineInfo, TCKType
from cylindra.const import ExtrapolationMode, nm
from cylindra.const import MoleculesHeader as Mole
from cylindra.cyltransform import polar_coords_2d
from cylindra.utils import ceilint, interval_divmod, roundint

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

logger = logging.getLogger("cylindra")


class Spline(BaseComponent):
    """
    3D spline curve model with coordinate system.

    Anchor points can be set via `anchor` property. A spline object is semi-immutable.
    Different spline curves are always of different objects, but the anchors and
    properties can be dynamically changed.

    References
    ----------
    - Scipy document
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
    """

    def __init__(
        self,
        order: int = 3,
        *,
        lims: tuple[float, float] = (0.0, 1.0),
        extrapolate: ExtrapolationMode | str = ExtrapolationMode.linear,
        config: dict[str, Any] | SplineConfig = {},
    ):
        self._tck: TCKType = (None, None, order)
        self._u: NDArray[np.float32] | None = None
        self._anchors = None
        self._extrapolate = ExtrapolationMode(extrapolate)

        self._lims = lims
        self._props = SplineProps()
        if isinstance(config, SplineConfig):
            self._config = config
        else:
            self._config = SplineConfig.construct(**config)

    @property
    def props(self) -> SplineProps:
        """Return the spline properties"""
        return self._props

    @property
    def config(self) -> SplineConfig:
        """Return the spline configuration"""
        return self._config

    def has_props(self) -> bool:
        """True if there are any properties."""
        return len(self.props.loc) > 0 or len(self.props.glob) > 0

    def copy(self, copy_props: bool = True, copy_config: bool = True) -> Self:
        """
        Copy Spline object.

        Parameters
        ----------
        copy_props : bool, default True
            Also copy local/global properties if true.

        Returns
        -------
        Spline
            Copied object.
        """
        new = self.__class__(
            order=self.order, lims=self._lims, extrapolate=self.extrapolate
        )
        new._tck = self._tck
        new._u = self._u
        new._anchors = self._anchors

        if copy_props:
            new._props = self.props.copy()
        if copy_config:
            new._config = self.config.copy()
        return new

    __copy__ = copy

    def with_extrapolation(self, extrapolate: ExtrapolationMode | str) -> Self:
        """Return a copy of the spline with a new extrapolation mode."""
        new = self.copy()
        new._extrapolate = ExtrapolationMode(extrapolate)
        return new

    def with_config(
        self,
        config: dict[str, Any] | SplineConfig,
        copy_props: bool = False,
    ) -> Self:
        """Return a copy of the spline with a new config."""
        new = self.copy(copy_props=copy_props, copy_config=False)
        if not isinstance(config, SplineConfig):
            config = SplineConfig.construct(**config)
        new._config = config
        return new

    @property
    def knots(self) -> np.ndarray:
        """Spline knots."""
        return self._tck[0]

    @property
    def coeff(self) -> list[np.ndarray]:
        """Spline coefficient."""
        return self._tck[1]

    @property
    def order(self) -> int:
        """Spline order."""
        return self._tck[2]

    @property
    def extrapolate(self) -> ExtrapolationMode:
        """Extrapolation mode of the spline."""
        return self._extrapolate

    @property
    def params(self) -> np.ndarray:
        """Spline parameters."""
        return self._u

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
        return spl.fit(coords, err_max=0.0)

    def translate(self, shift: tuple[nm, nm, nm]):
        """Translate the spline by given shift vectors."""
        new = self.copy()
        c = [x + s for x, s in zip(self.coeff, shift, strict=True)]
        new._tck = (self.knots, c, self.order)
        return new

    @property
    def has_anchors(self) -> bool:
        """True if there are any anchors."""
        return self._anchors is not None

    @property
    def anchors(self) -> NDArray[np.float32]:
        """Local anchors along spline."""
        if self._anchors is None:
            raise ValueError("Anchor has not been set yet.")
        return self._anchors

    @anchors.setter
    def anchors(self, positions: float | Sequence[float]) -> None:
        _anc = np.atleast_1d(np.asarray(positions, dtype=np.float32))
        if _anc.ndim != 1:
            raise TypeError("Could not convert positions into 1D array.")
        elif _anc.min() < 0 or _anc.max() > 1:
            msg = (
                "Anchor positions should be set between 0 and 1. Otherwise spline "
                "curve does not fit well."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        _old = self._anchors
        if (
            _old is None
            or _anc.size != _old.size
            or not np.allclose(_anc, _old, rtol=1e-4, atol=1e-4)
        ):
            self.props.clear_loc()
        self._anchors = _anc
        return None

    @anchors.deleter
    def anchors(self) -> None:
        self._anchors = None
        self.props.clear_loc()
        return None

    def is_inverted(self) -> bool:
        """Return true if spline is inverted."""
        return self._lims[0] > self._lims[1]

    @property
    def lims(self) -> tuple[float, float]:
        """Return spline limit positions."""
        return self._lims

    def _set_params(self, tck, u) -> Self:
        self._tck = tck
        self._u = u
        return self

    def prep_anchor_positions(
        self,
        interval: nm | None = None,
        n: int | None = None,
        max_interval: nm | None = None,
    ) -> NDArray[np.float32]:
        length = self.length()
        if interval is not None:
            stop, n_segs = interval_divmod(length, interval)
            n = n_segs + 1
        elif n is not None:
            stop = length
        elif max_interval is not None:
            n = max(ceilint(length / max_interval), self.order) + 1
            stop = length
        else:
            raise ValueError("Either 'interval' or 'n' must be specified.")
        y = np.linspace(0, stop, n)
        return self.y_to_position(y)

    def make_anchors(
        self,
        interval: nm | None = None,
        n: int | None = None,
        max_interval: nm | None = None,
    ) -> Self:
        """
        Make anchor points at constant intervals. Either interval, number of anchor or
        the maximum interval between anchors can be specified.

        Parameters
        ----------
        interval : nm, optional
            Interval between anchor points.
        n : int, optional
            Number of anchor points, including both ends.
        max_interval: nm, optional
            Spline will be split by as little anchors as possible but interval between
            anchors will not be larger than this. The number of anchors are also
            guaranteed to be larger than spline order.
        """
        self.anchors = self.prep_anchor_positions(interval, n, max_interval)
        return self

    def __repr__(self) -> str:
        """Use start/end points to describe a spline."""
        start, end = self.map(self._lims)
        start = "({:.1f}, {:.1f}, {:.1f})".format(*start)
        end = "({:.1f}, {:.1f}, {:.1f})".format(*end)
        return f"Spline[{start}:{end}]"

    def close_to(self: Self, other: Self) -> bool:
        """True if two objects draws the same curve."""
        if not isinstance(other, self.__class__):
            return False
        t0, c0, k0 = self._tck
        t1, c1, k1 = other._tck
        return (
            np.allclose(t0, t1)
            and all(np.allclose(x, y) for x, y in zip(c0, c1, strict=True))
            and k0 == k1
            and np.allclose(self._u, other._u)
            and np.allclose(self._lims, other._lims)
        )

    def clip(self, start: float, stop: float) -> Self:
        """
        Clip spline and generate a new one.

        This method does not convert spline bases. `_lims` is updated instead.

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
        return self.__class__(
            order=self.order,
            lims=(u0, u1),
            extrapolate=self.extrapolate,
            config=self.config,
        )._set_params(self._tck, self._u)

    def restore(self) -> Self:
        """
        Restore the original, not-clipped spline.

        Returns
        -------
        Spline
            Copy of the original spline.
        """
        return self.__class__(
            order=self.order,
            lims=(0, 1),
            extrapolate=self.extrapolate,
            config=self.config,
        )._set_params(self._tck, self._u)

    def resample(self, max_interval: nm = 1.0, err_max: nm = 0.1) -> Self:
        """
        Resample a new spline along the original spline.

        Parameters
        ----------
        max_interval : nm, default 1.0
            Maximum interval between resampling points.
        err_max : float, default 0.1
            Spline fitting maximum error.

        Returns
        -------
        Spline
            Resampled spline object.
        """
        l = self.length()
        points = self.map(np.linspace(0, 1, ceilint(l / max_interval)))
        return self.fit(points, err_max=err_max)

    def fit(
        self,
        coords: ArrayLike,
        *,
        err_max: nm = 1.0,
    ) -> Self:
        """
        Fit spline model to coordinates.

        This method uses `scipy.interpolate.splprep` to fit given coordinates to a
        spline.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates. Must be (N, 3).
        err_max : float, default 1.0
            Error allowed for fitting. Several upper limit of residual values will be
            used and the fit that results in error lower than this value and minimize
            the maximum curvature will be chosen.

        Returns
        -------
        Spline
            New spline fit to given coordinates.
        """
        crds = np.asarray(coords)
        npoints = crds.shape[0]
        if npoints < 2:
            raise ValueError("Number of input coordinates must be > 1.")
        if npoints <= self.order:
            k = npoints - 1
        else:
            k = self.order

        if self.is_inverted():
            crds = crds[::-1]

        if err_max > 4.0:
            raise ValueError("std_max must be smaller than 4.0.")
        if err_max < 1e-3:
            std_list = [err_max]
        else:
            ntrial = max(int(err_max / 0.02), 2)
            std_list = np.linspace(0, err_max, ntrial)[1:]

        fit_results = list[SplineFitResult]()
        new = self.__class__(order=k, extrapolate=self.extrapolate, config=self.config)
        with warnings.catch_warnings():
            # fitting may fail for some std
            warnings.simplefilter("ignore", RuntimeWarning)
            for std in std_list:
                _tck, _u = splprep(crds.T, k=k, s=std**2 * npoints)
                new._set_params(_tck, _u)
                _crds_at_u = new.map(_u)
                res: NDArray[np.float32] = np.sqrt(
                    np.sum((_crds_at_u - crds) ** 2, axis=1)
                )
                _knots = _tck[0][new.order : -new.order]
                nedge = _knots.size - 1
                assert nedge > 0
                nanc = nedge * 20 + 1
                anc = np.interp(
                    np.linspace(0, 1, nanc), np.linspace(0, 1, nedge + 1), _knots
                )
                max_curvature = new.curvature(anc).max()
                success = res.max() <= err_max
                fit_results.append(
                    SplineFitResult((_tck, _u), max_curvature, res, success)
                )

        fit_results_filt = list(filter(lambda x: x.success, fit_results))
        if len(fit_results_filt) == 0:
            fit_results_filt = fit_results

        reult_opt = min(fit_results_filt, key=lambda x: x.curvature)
        return new._set_params(*reult_opt.params)

    def shift(
        self,
        positions: Sequence[float] | None = None,
        shifts: NDArray[np.floating] | None = None,
        *,
        err_max: nm = 1.0,
    ) -> Self:
        """
        Fit spline model using a list of shifts in XZ-plane.

        Parameters
        ----------
        positions : sequence of float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.
        shifts : np.ndarray
            Shift from center in nm. Must be (N, 2).
        err_max : float, default 1.0
            Error allowed for fitting. See `Spline.fit`.

        Returns
        -------
        Spline
            Spline shifted by fitting to given coordinates.
        """
        if shifts is None:
            raise ValueError("Shifts must be given.")
        coords = self.map(positions)
        rot = self.get_rotator(positions)
        # insert 0 in y coordinates.
        shifts = np.stack([shifts[:, 0], np.zeros(len(rot)), shifts[:, 1]], axis=1)
        coords += rot.apply(shifts)
        return self.fit(coords, err_max=err_max)

    def distances(
        self,
        positions: Sequence[float] | None = None,
        nknots: int = 512,
    ) -> NDArray[np.float32]:
        """
        Get the distances from u=0.

        Parameters
        ----------
        positions : sequence of float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.

        Returns
        -------
        np.ndarray
            Distances for each `u`.
        """
        if positions is None:
            _u = self.anchors
        else:
            _u = np.asarray(positions, dtype=np.float32)
            if _u.ndim != 1:
                raise ValueError("Positions must be 1D array.")
        u = np.linspace(0, 1, nknots)
        u_tr = _linear_conversion(u, *self._lims)
        dz, dy, dx = map(np.diff, splev(u_tr, self._tck, der=0))
        dist = np.concatenate([[0], np.sqrt(dx**2 + dy**2 + dz**2)]).cumsum()
        tck = splrep(u, dist, k=1)
        out = splev(_u, tck)
        return out

    def map(
        self,
        positions: float | NDArray[np.number] | None = None,
        der: int = 0,
    ) -> NDArray[np.float32]:
        """
        Calculate coordinates (or n-th derivative) at points on the spline.

        Parameters
        ----------
        positions : np.ndarray or float, optional
            Positions. Between 0 and 1. If not given, anchors are used instead.
        der : int, default 0
            `der`-th derivative will be calculated.

        Returns
        -------
        np.ndarray
            Positions or vectors in (3,) or (N, 3) shape.
        """
        _assert_fitted(self)
        if positions is None:
            positions = self.anchors
        u0, u1 = self._lims
        if np.isscalar(positions):
            if self.order < der:
                return np.zeros(3, dtype=np.float32)
            u_tr = _linear_conversion(float(positions), u0, u1)
            if 0 <= u_tr <= 1 or self.extrapolate is ExtrapolationMode.default:
                coord = splev([u_tr], self._tck, der=der)
            elif self.extrapolate is ExtrapolationMode.linear:
                if der == 0:
                    if u_tr < 0:
                        der0 = splev([0], self._tck, der=0)
                        der1 = splev([0], self._tck, der=1)
                        dr = u_tr
                    else:
                        der0 = splev([1], self._tck, der=0)
                        der1 = splev([1], self._tck, der=1)
                        dr = u_tr - 1
                    coord = [a0 + a1 * dr for a0, a1 in zip(der0, der1, strict=True)]
                elif der == 1:
                    if u_tr < 0:
                        coord = splev([0], self._tck, der=1)
                    else:
                        coord = splev([1], self._tck, der=1)
                else:
                    coord = [[0], [0], [0]]
            else:
                raise ValueError(f"Invalid extrapolation mode: {self.extrapolate!r}.")
            out = np.concatenate(coord).astype(np.float32)

        else:
            u_tr = _linear_conversion(np.asarray(positions, dtype=np.float32), u0, u1)
            if self.order < der:
                return np.zeros((u_tr.size, 3), dtype=np.float32)
            if self.extrapolate is ExtrapolationMode.default:
                out = np.stack(splev(u_tr, self._tck, der=der), axis=1).astype(
                    np.float32
                )
            elif self.extrapolate is ExtrapolationMode.linear:
                sl_small = u_tr < 0
                sl_large = u_tr > 1
                n_small = np.count_nonzero(sl_small)
                n_large = np.count_nonzero(sl_large)
                out = np.stack(splev(u_tr, self._tck, der=der), axis=1).astype(
                    np.float32
                )
                if der == 0:
                    if n_small > 0:
                        der0 = np.array(splev(0, self._tck, der=0), dtype=np.float32)
                        der1 = np.array(splev(0, self._tck, der=1), dtype=np.float32)
                        dr = u_tr[sl_small]
                        coords_new = der0 + der1 * dr[:, np.newaxis]
                        out[sl_small] = coords_new

                    if n_large > 0:
                        der0 = splev(1, self._tck, der=0)
                        der1 = splev(1, self._tck, der=1)
                        dr = u_tr[sl_large] - 1
                        coords_new = der0 + der1 * dr[:, np.newaxis]
                        out[sl_large] = coords_new

                elif der == 1:
                    if n_small > 0:
                        out[sl_small] = splev(0, self._tck, der=1)
                    if n_large > 0:
                        out[sl_large] = splev(1, self._tck, der=1)
                else:
                    if n_small > 0:
                        out[sl_small] = 0
                    if n_large > 0:
                        out[sl_large] = 0
            else:
                raise ValueError(f"Invalid extrapolation mode: {self.extrapolate!r}.")

        if u0 > u1 and der % 2 == 1:
            out = -out
        return out

    __call__ = map  # scipy-style alias

    def partition(self, n: int, der: int = 0) -> NDArray[np.float32]:
        """Return the n-partitioning coordinates of the spline."""
        u = np.linspace(0, 1, n)
        return self.map(u, der)

    def length(self, start: float = 0, stop: float = 1, nknots: int = 512) -> nm:
        """
        Approximate the length of B-spline between [start, stop] by partitioning
        the spline with 'nknots' knots. nknots=256 is large enough for most cases.
        """
        _assert_fitted(self)
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
        inverted = self.clip(1.0, 0.0)
        if anchors is not None:
            inverted.anchors = 1 - anchors[::-1]
        return inverted

    def curvature(
        self,
        positions: Sequence[float] | None = None,
    ) -> NDArray[np.float32]:
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

        dz, dy, dx = self.map(positions, der=1).T
        ddz, ddy, ddx = self.map(positions, der=2).T
        a = (
            (ddz * dy - ddy * dz) ** 2
            + (ddx * dz - ddz * dx) ** 2
            + (ddy * dx - ddx * dy) ** 2
        )
        return np.sqrt(a) / (dx**2 + dy**2 + dz**2) ** 1.5

    def curvature_radii(self, positions: Sequence[float] = None) -> NDArray[np.float32]:
        """Inverse of curvature."""
        return 1.0 / self.curvature(positions)

    def to_dict(self) -> SplineInfo:
        """Convert spline info into a dict."""
        t, c, k = self._tck
        u = self._u
        return {
            "t": t.tolist(),
            "c": {"z": c[0].tolist(), "y": c[1].tolist(), "x": c[2].tolist()},
            "k": k,
            "u": u.tolist(),
            "lims": self._lims,
            "localprops_window_size": dict(self.props.window_size),
            "binsize_loc": dict(self.props.binsize_loc),
            "binsize_glob": dict(self.props.binsize_glob),
            "extrapolate": self._extrapolate.name,
            "config": self.config.asdict(),
        }

    @classmethod
    def from_dict(cls: type[Self], d: SplineInfo) -> Self:
        """
        Construct a spline model from a dictionary.

        Parameters
        ----------
        d: dict
            Dictionary with keys "t", "c", "k", "u" and "lims".

        Returns
        -------
        Spline
            Spline object constructed from the dictionary.
        """
        self = cls(
            order=d.get("k", 3),
            lims=d.get("lims", (0, 1)),
            extrapolate=d.get("extrapolate", "linear"),
        )
        t = np.asarray(d["t"])
        c = [np.asarray(d["c"][k]) for k in "zyx"]
        k = roundint(d["k"])
        self._tck = (t, c, k)
        self._u = np.asarray(d["u"])
        self.props._window_size = d.get("localprops_window_size", {})
        self.props._binsize_loc = d.get("binsize_loc", {})
        self.props._binsize_glob = d.get("binsize_glob", {})
        if cfg := d.get("config", None):
            self._config = SplineConfig.from_dict(cfg)
        return self

    def get_rotator(
        self,
        positions: Sequence[float] | None = None,
        inverse: bool = False,
    ) -> Rotation:
        """
        Calculate list of Affine transformation matrix along spline, which correspond to
        the orientation of spline curve.

        Parameters
        ----------
        positions : array-like, (N,)
            Positions. Between 0 and 1.
        inverse : bool, default False
            If True, rotation matrix will be inversed.

        Returns
        -------
        Rotation
            Rotation object at each anchor.
        """
        if positions is None:
            positions = self.anchors
        ds = self.map(positions, der=1)
        out = axes_to_rotator(None, -ds)

        if inverse:
            out = out.inv()

        return out

    def local_cartesian(
        self,
        shape: tuple[nm, nm],
        depth: nm,
        u: float | Sequence[float] = None,
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
        """
        Generate local Cartesian coordinate systems.

        The generated array can be used for `ndi.map_coordinates`. The result coordinate
        systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        shape : (float, float)
            Vertical and horizontal length of Cartesian coordinates. Corresponds to zx
            axes.
        depth : float
            Length of y axis in nm.
        u : float, optional
            Position on the spline at which local Cartesian coordinates will be built.
        scale: nm, default 1.0
            Scale of coordinates, i.e. spacing of the grid.

        Returns
        -------
        np.ndarray
            (D, V, S, H) shape. Each cooresponds to dimensional vertical, longitudinal
            and horizontal axis, which is ready to be used in `ndi.map_coordinates`.
        """

        mole = self.anchors_to_molecules(u)
        nz = roundint(shape[0] / scale)
        ny = roundint(depth / scale)
        nx = roundint(shape[1] / scale)
        return mole.local_coordinates(shape=(nz, ny, nx), scale=scale)

    def local_cylindrical(
        self,
        r_range: tuple[nm, nm],
        depth: nm,
        u: float | None = None,
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
        """
        Generate local cylindrical coordinate systems.

        The generated array can be used for `ndi.map_coordinates`. The result coordinate
        systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        r_range : (float, float)
            Lower and upper bound of radius in nm.
        depth : nm
            Length of y axis in nm.
        u : float
            Position on the spline at which local cylindrical coordinates will be built.
        scale: nm, default 1.0
            Scale of coordinates, i.e. spacing of the grid.

        Returns
        -------
        np.ndarray
            (D, V, S, H) shape. Each cooresponds to dimensional, radius, longitudinal
            and angle axis, which is ready to be used in `ndi.map_coordinates`.
        """
        if u is None:
            u = self.anchors
        rmin, rmax = r_range
        ds = self.map(u, der=1)
        ds_norm: NDArray[np.float32] = ds.reshape(-1, 1) / np.sqrt(sum(ds**2))
        depth_px = roundint(depth / scale)
        grid = np.linspace(-depth_px / 2 + 0.5, depth_px / 2 - 0.5, depth_px)
        dy = ds_norm * grid
        y_ax_coords = (self.map(u) / scale).reshape(1, -1) + dy.T
        dslist = np.stack([ds] * depth_px, axis=0)
        map_ = polar_coords_2d(rmin / scale, rmax / scale)
        map_slice = _stack_coords(map_)
        out = _rot_with_vector(map_slice, y_ax_coords, dslist)
        return np.moveaxis(out, -1, 0)

    def cartesian(
        self,
        shape: tuple[nm, nm],
        s_range: tuple[float, float] = (0, 1),
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
        """
        Generate a Cartesian coordinate system along spline that can be used for
        `ndi.map_coordinate`. Note that this coordinate system is distorted, thus
        does not reflect real geometry (such as distance and derivatives).

        Parameters
        ----------
        shape : (float, float)
            The ZX-shape of output coordinate system. Center of the array will be
            spline curve itself after coodinate transformation.
        s_range : tuple[float, float], default (0, 1)
            Range of spline. Spline coordinate system will be built between
            `spl[s_range[0]]` and `spl[s_range[1]]`.
        scale: nm, default 1.0
            Scale of coordinates, i.e. spacing of the grid.

        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to vertical, longitudinal, horizontal
            and dimensional axis.
        """
        dz = roundint(shape[0] / scale)
        dx = roundint(shape[1] / scale)
        return self._get_coords(_cartesian_coords_2d, (dz, dx), s_range, scale)

    def cylindrical(
        self,
        r_range: tuple[nm, nm],
        s_range: tuple[float, float] = (0, 1),
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
        """
        Generate a cylindrical coordinate system along spline that can be used for
        `ndi.map_coordinate`. Note that this coordinate system is distorted, thus
        does not reflect real geometry (such as distance and derivatives).

        Parameters
        ----------
        r_range : (nm, nm)
            Range of radius in nm. r=0 will be spline curve itself after coodinate
            transformation.
        s_range : tuple[float, float], default (0, 1)
            Range of spline. Spline coordinate system will be built between
            `spl[s_range[0]]` and `spl[s_range[1]]`.
        scale: nm, default 1.0
            Scale of coordinates, i.e. spacing of the grid.

        Returns
        -------
        np.ndarray
            (V, S, H, D) shape. Each cooresponds to radius, longitudinal, angle and
            dimensional axis.
        """
        rmin = r_range[0] / scale
        rmax = r_range[1] / scale
        return self._get_coords(polar_coords_2d, (rmin, rmax), s_range, scale)

    def y_to_position(
        self, y: NDArray[np.float32], nknots: int = 512
    ) -> NDArray[np.float32]:
        """
        Convert y-coordinate to spline position parameter.

        Parameters
        ----------
        y : array-like
            Y coordinates.
        nknots : int, optional
            Number of knots. Increasing the number of knots will increase the accuracy.
        """
        # almost equal to y / self.length()
        u = np.linspace(0, 1, nknots)
        u_tr = _linear_conversion(u, *self._lims)
        dz, dy, dx = map(np.diff, splev(u_tr, self._tck, der=0))
        dist = np.concatenate([[0], np.sqrt(dx**2 + dy**2 + dz**2)]).cumsum()
        tck = splrep(dist, u, k=1)
        out = splev(y, tck)
        return out

    def cartesian_to_world(
        self,
        coords: NDArray[np.float32],
        nknots: int = 512,
    ) -> NDArray[np.float32]:
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
        _zs = coords[:, 0]
        _us = self.y_to_position(coords[:, 1], nknots=nknots)
        _xs = coords[:, 2]
        _zeros = np.zeros(ncoords, dtype=np.float32)
        coords_ext = np.stack([_zs, _zeros, _xs], axis=1)
        rot = self.get_rotator(_us)
        out = rot.apply(coords_ext) + self.map(_us)

        return out

    def cylindrical_to_world(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
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
        cart_coords = np.stack(
            [radius * np.sin(theta), y, radius * np.cos(theta)], axis=1
        )

        return self.cartesian_to_world(cart_coords)

    def anchors_to_molecules(
        self,
        positions: float | Sequence[float] | None = None,
        rotation: Sequence[float] | None = None,
    ) -> Molecules:
        """
        Convert coordinates of anchors to `Molecules` instance.

        Coordinates of anchors must be in range from 0 to 1. The y-direction of
        `Molecules` always points at the direction of spline and the z- direction always
        in the plane orthogonal to YX-plane.

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
        elif np.isscalar(positions):
            positions = [positions]
        pos = self.map(positions)
        yvec = self.map(positions, der=1)
        rot = axes_to_rotator(None, yvec)
        if rotation is not None:
            rotvec = np.zeros((len(rot), 3), dtype=np.float32)
            rotvec[:, 1] = rotation
            rot = rot * Rotation.from_rotvec(rotvec)
        return Molecules(pos=pos, rot=rot, features={Mole.nth: np.arange(len(pos))})

    def cylindrical_to_molecules(
        self,
        coords: NDArray[np.float32],
    ) -> Molecules:
        """
        Convert coordinates of points near the spline to `Molecules` instance.

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
        u = self.y_to_position(coords[:, 1])
        ycoords = self.map(u)
        zvec = world_coords - ycoords
        yvec = self.map(u, der=1)
        return Molecules.from_axes(pos=world_coords, z=zvec, y=yvec)

    def _get_coords(
        self,
        map_func: Callable[[tuple], NDArray[np.float32]],
        map_params: tuple,
        s_range: tuple[float, float],
        scale: nm,
    ):
        """
        Make coordinate system using function `map_func` and stack the same point cloud
        in the direction of the spline, in the range of `s_range`.
        """
        u, y_ax_coords = self._get_y_ax_coords(s_range, scale)
        dslist = self.map(u, 1).astype(np.float32)
        map_ = map_func(*map_params)
        map_slice = _stack_coords(map_)
        out = _rot_with_vector(map_slice, y_ax_coords, dslist)
        return np.moveaxis(out, -1, 0)

    def _get_y_ax_coords(self, s_range: tuple[float, float], scale: nm):
        s0, s1 = s_range
        length = self.length(start=s0, stop=s1)
        stop_length, n_segs = interval_divmod(length, scale)
        n_pixels = n_segs + 1
        s2 = (s1 - s0) * stop_length / length + s0
        if n_pixels < 2:
            raise ValueError("Too short. Change 's_range'.")
        d0, d2 = self.distances([s0, s2])
        u = self.y_to_position(np.linspace(d0, d2, n_pixels))
        y = self.map(u) / scale  # world coordinates of y-axis in spline coords system
        return u, y


# fmt: off
_T = TypeVar("_T", bound=np.generic)
@overload
def _linear_conversion(u: float, start: float, stop: float) -> float: ...
@overload
def _linear_conversion(u: NDArray[_T], start: float, stop: float) -> NDArray[_T]: ...
# fmt: on


def _linear_conversion(u, start, stop):
    return (1 - u) * start + u * stop


def _assert_fitted(spl: Spline):
    if spl._u is None:
        raise ValueError("Spline is not fitted yet.")


def _rot_with_vector(
    maps: NDArray[_T], ax_coords: NDArray[_T], vectors: NDArray[_T]
) -> NDArray[_T]:
    rot = axes_to_rotator(None, vectors)
    mat = rot.as_matrix()
    out = np.einsum("nij,vhj->vnhi", mat, maps)
    out += ax_coords[np.newaxis, :, np.newaxis]
    return out


@lru_cache(maxsize=12)
def _cartesian_coords_2d(lenv: int, lenh: int):
    v, h = np.indices((lenv, lenh), dtype=np.float32)
    v -= lenv / 2 - 0.5
    h -= lenh / 2 - 0.5
    return np.stack([v, h], axis=2)  # V, H, 2


def _stack_coords(coords: NDArray[_T]) -> NDArray[_T]:  # V, H, D
    shape = coords.shape[:-1]
    zeros = np.zeros(shape, dtype=np.float32)
    stacked = np.stack(
        [
            coords[..., 0],
            zeros,
            coords[..., 1],
        ],
        axis=2,
    )  # V, S, H, D
    return stacked
