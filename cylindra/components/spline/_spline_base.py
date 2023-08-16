from __future__ import annotations

from functools import lru_cache
from typing import (
    Any,
    Callable,
    Sequence,
    TypeVar,
    TypedDict,
    TYPE_CHECKING,
    overload,
)
import warnings
import logging

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation

from acryo import Molecules
from acryo.molecules import axes_to_rotator

import polars as pl

from cylindra.utils import ceilint, interval_divmod, roundint
from cylindra.const import Mode, nm, ExtrapolationMode
from cylindra.components._base import BaseComponent
from ._props import SplineProps
from ._config import SplineConfig
from ._types import TCKType, SplineInfo, SplineFitResult

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger("cylindra")


class Spline(BaseComponent):
    """
    3D spline curve model with coordinate system.

    Anchor points can be set via ``anchor`` property. A spline object is semi-immutable.
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

    @property
    def localprops(self) -> pl.DataFrame:
        """Local properties of the spline."""
        return self.props.loc

    @property
    def globalprops(self) -> pl.DataFrame:
        """Global properties of the spline."""
        return self.props.glob

    @globalprops.setter
    def globalprops(self, df: pl.DataFrame):
        self.props.glob = df

    def has_props(self) -> bool:
        """True if there are any properties."""
        return len(self.props.loc) > 0 or len(self.props.glob) > 0

    def copy(self, copy_props: bool = True, copy_config: bool = True) -> Self:
        """
        Copy Spline object.

        Parameters
        ----------
        copy_props : bool, default is True
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

    def with_config(self, config: dict[str, Any] | SplineConfig) -> Self:
        """Return a copy of the spline with a new config."""
        new = self.copy(copy_config=False)
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
        c = [x + s for x, s in zip(self.coeff, shift)]
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
            raise TypeError(f"Could not convert positions into 1D array.")
        elif _anc.min() < 0 or _anc.max() > 1:
            msg = (
                f"Anchor positions should be set between 0 and 1. Otherwise spline "
                f"curve does not fit well."
            )
            warnings.warn(msg, UserWarning)
        self._anchors = _anc
        self.props.clear_loc()
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
            end = stop / length
            n = n_segs + 1
        elif n is not None:
            end = 1
        elif max_interval is not None:
            n = max(ceilint(length / max_interval), self.order) + 1
            end = 1
        else:
            raise ValueError("Either 'interval' or 'n' must be specified.")
        return np.linspace(0, end, n)

    def make_anchors(
        self,
        interval: nm | None = None,
        n: int | None = None,
        max_interval: nm | None = None,
    ) -> Self:
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
            and all(np.allclose(x, y) for x, y in zip(c0, c1))
            and k0 == k1
            and np.allclose(self._u, other._u)
            and np.allclose(self._lims, other._lims)
        )

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

    def resample(self, max_interval: nm = 1.0, std_max: nm = 0.1) -> Self:
        """
        Resample a new spline along the original spline.

        Parameters
        ----------
        max_interval : nm, default is 1.0
            Maximum interval between resampling points.
        variance : float, default is 0.0
            Spline fitting variance.

        Returns
        -------
        Spline
            Resampled spline object.
        """
        l = self.length()
        points = self.map(np.linspace(0, 1, ceilint(l / max_interval)))
        return self.fit(points, err_max=std_max)

    def fit(
        self,
        coords: ArrayLike,
        *,
        weight_ramp: tuple[float, float] | None = None,
        err_max: nm = 1.0,
    ) -> Self:
        """
        Fit spline model to coordinates.

        This method uses ``scipy.interpolate.splprep`` to fit given coordinates to a spline.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates. Must be (N, 3).
        err_max : float, default is 1.0
            Error allowed for fitting. Several upper limit of residual values will be used and
            the fit that results in error lower than this value and minimize the maximum
            curvature will be chosen.

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

        # weight
        if weight_ramp is None:
            weight_ramp = self.config.weight_ramp.astuple()
        weight = _normalize_weight(weight_ramp, crds)

        if self.is_inverted():
            crds = crds[::-1]

        if err_max > 4.0:
            raise ValueError("std_max must be smaller than 4.0.")
        if err_max == 0:
            std_list = [0]
        else:
            ntrial = max(int(err_max / 0.02), 2)
            std_list = np.linspace(0, err_max, ntrial)[1:]

        fit_results = list[SplineFitResult]()
        new = self.__class__(order=k, extrapolate=self.extrapolate, config=self.config)
        length = np.sum(np.sqrt(np.sum(np.diff(crds, axis=0) ** 2, axis=1)))
        anc = np.linspace(0, 1, max(int(length / 5), 2))
        for std in std_list:
            param = splprep(crds.T, k=k, w=weight, s=std**2 * npoints)
            new._set_params(*param)
            _crds_at_u = new.map(param[1])
            res: NDArray[np.float32] = np.sqrt(
                np.sum(((_crds_at_u - crds) * weight[:, np.newaxis]) ** 2, axis=1)
            )
            max_curvature = new.curvature(anc).max()
            fit_results.append(
                SplineFitResult(param, max_curvature, res, res.max() <= err_max)
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
        weight_ramp: tuple[float, float] | None = None,
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
        err_max : float, default is 1.0
            Error allowed for fitting. See ``Spline.fit``.

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
        return self.fit(coords, err_max=err_max, weight_ramp=weight_ramp)

    def distances(
        self, positions: Sequence[float] | None = None
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
            Distances for each ``u``.
        """
        if positions is None:
            positions = self.anchors
        length = self.length()
        return length * np.asarray(positions, dtype=np.float32)

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
        der : int, default is 0
            ``der``-th derivative will be calculated.

        Returns
        -------
        np.ndarray
            Positions or vectors in (3,) or (N, 3) shape.
        """
        if self._u is None:
            raise ValueError("Spline is not fitted yet.")
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
                    coord = [a0 + a1 * dr for a0, a1 in zip(der0, der1)]
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
        if self._u is None:
            raise ValueError("Spline is not fitted yet.")
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
            Dictionary with keys {"t", "c", "k", "u", "lims"}.

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
        if cfg := d.get("config", None):
            self._config = SplineConfig.from_dict(cfg)
        return self

    def affine_matrix(
        self,
        positions: Sequence[float] = None,
        center: Sequence[float] = None,
        *,
        inverse: bool = False,
    ) -> NDArray[np.float32]:
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
        ds = self.map(positions, der=1)

        if ds.ndim == 1:
            ds = ds[np.newaxis]
        rot = axes_to_rotator(None, ds)
        if inverse:
            rot = rot.inv()
        out = np.zeros((len(rot), 4, 4), dtype=np.float32)
        out[:, :3, :3] = rot.as_matrix()
        out[:, 3, 3] = 1.0

        if center is not None:
            dz, dy, dx = center
            # center to corner
            translation_0 = np.array(
                [
                    [1.0, 0.0, 0.0, dz],
                    [0.0, 1.0, 0.0, dy],
                    [0.0, 0.0, 1.0, dx],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            # corner to center
            translation_1 = np.array(
                [
                    [1.0, 0.0, 0.0, -dz],
                    [0.0, 1.0, 0.0, -dy],
                    [0.0, 0.0, 1.0, -dx],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            out = translation_0 @ out @ translation_1
        if np.isscalar(positions):
            out = out[0]
        return out

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
        inverse : bool, default is False
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
        shape: tuple[int, int],
        n_pixels: int,
        u: float | Sequence[float] = None,
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
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
        return mole.local_coordinates(shape=(shape[0], n_pixels, shape[1]), scale=scale)

    def local_cylindrical(
        self,
        r_range: tuple[float, float],
        n_pixels: int,
        u: float | None = None,
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
        """
        Generate local cylindrical coordinate systems that can be used for ``ndi.map_coordinates``.
        The result coordinate systems are flat, i.e., not distorted by the curvature of spline.

        Parameters
        ----------
        r_range : (float, float)
            Lower and upper bound of radius in pixels.
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
        ds = self.map(u, der=1)
        ds_norm: NDArray[np.float32] = ds.reshape(-1, 1) / np.sqrt(sum(ds**2))
        grid = np.linspace(-n_pixels / 2 + 0.5, n_pixels / 2 - 0.5, n_pixels)
        dy = ds_norm * grid
        y_ax_coords = (self.map(u) / scale).reshape(1, -1) + dy.T
        dslist = np.stack([ds] * n_pixels, axis=0)
        map_ = _polar_coords_2d(*r_range)
        map_slice = _stack_coords(map_)
        out = _rot_with_vector(map_slice, y_ax_coords, dslist)
        return np.moveaxis(out, -1, 0)

    def cartesian(
        self,
        shape: tuple[int, int],
        s_range: tuple[float, float] = (0, 1),
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
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
        scale: nm = 1.0,
    ) -> NDArray[np.float32]:
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

    def cartesian_to_world(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
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
        _rs = coords[:, 0]
        _us = coords[:, 1] / self.length()
        _as = coords[:, 2]
        _zeros = np.zeros(ncoords, dtype=np.float32)
        coords_ext = np.stack([_rs, _zeros, _as], axis=1)
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
        pos = self.map(positions)
        yvec = self.map(positions, der=1)
        rot = axes_to_rotator(None, yvec)
        if rotation is not None:
            rotvec = np.zeros((len(rot), 3), dtype=np.float32)
            rotvec[:, 1] = rotation
            rot = rot * Rotation.from_rotvec(rotvec)
        return Molecules(pos=pos, rot=rot)

    def cylindrical_to_molecules(
        self,
        coords: NDArray[np.float32],
    ) -> Molecules:
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
        u = coords[:, 1] / self.length()
        ycoords = self.map(u)
        zvec = world_coords - ycoords
        yvec = self.map(u, der=1)
        return Molecules.from_axes(pos=world_coords, z=zvec, y=yvec)

    def slice_along(
        self,
        array: NDArray[np.float32],
        s_range: tuple[float, float] = (0.0, 1.0),
        order: int = 3,
        mode: str = Mode.constant,
        cval: float = 0.0,
    ) -> NDArray[np.float32]:
        """Slice input array along the spline."""
        from scipy import ndimage as ndi

        _, coords = self._get_y_ax_coords(s_range)
        return ndi.map_coordinates(
            array,
            coords.T,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=order > 1,
        )

    def _get_coords(
        self,
        map_func: Callable[[tuple], NDArray[np.float32]],
        map_params: tuple,
        s_range: tuple[float, float],
        scale: nm,
    ):
        """
        Make coordinate system using function ``map_func`` and stack the same point cloud
        in the direction of the spline, in the range of ``s_range``.
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
        u = np.linspace(s0, s2, n_pixels)
        y = self.map(u) / scale  # world coordinates of y-axis in spline coords system
        return u, y


_T = TypeVar("_T", bound=np.generic)


@overload
def _linear_conversion(u: float, start: float, stop: float) -> float:
    ...


@overload
def _linear_conversion(u: NDArray[_T], start: float, stop: float) -> NDArray[_T]:
    ...


def _linear_conversion(u, start, stop):
    return (1 - u) * start + u * stop


def _rot_with_vector(
    maps: NDArray[_T], ax_coords: NDArray[_T], vectors: NDArray[_T]
) -> NDArray[_T]:
    rot = axes_to_rotator(None, vectors)
    mat = rot.as_matrix()
    out = np.einsum("nij,vhj->vnhi", mat, maps)
    out += ax_coords[np.newaxis, :, np.newaxis]
    return out


@lru_cache(maxsize=12)
def _polar_coords_2d(r_start: float, r_stop: float, center=None) -> NDArray[np.float32]:
    n_angle = roundint((r_start + r_stop) * np.pi)
    n_radius = roundint(r_stop - r_start)
    r_, ang_ = np.indices((n_radius, n_angle))
    r_ = r_ + (r_start + r_stop - n_radius + 1) / 2
    # NOTE: r_.mean() is always (r_start + r_stop) / 2
    output_coords = np.column_stack([r_.ravel(), ang_.ravel()])
    if center is None:
        center = [0, 0]
    coords = _linear_polar_mapping(
        np.array(output_coords),
        k_angle=n_angle / 2 / np.pi,
        k_radius=1,
        center=center[::-1],
    ).astype(np.float32)
    coords = coords.reshape(n_radius, n_angle, 2)  # V, H, 2

    # Here, the first coordinate should be theta=0, and theta moves anti-clockwise
    coords[:] = np.flip(coords, axis=2)  # flip y, x
    return coords


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


def _construct_ramping_weight(
    norm_length: float, weight_min: float, size: int
) -> NDArray[np.float64]:
    """
    Prepare an weight array with linear ramping flanking regions.

    Parameters
    ----------
    norm_length : float
        Normalized length (total length is 1.0) of the ramping region. The value should be
        between 0 and 0.5.
    weight_min : float
        weight at the edge. The value should be between 0 and 1.
    size : int
        Size of the output array.
    """
    assert norm_length > 0
    assert 0 <= weight_min < 1
    norm_length = min(norm_length, 0.5)

    u = np.linspace(0, 1, size)
    weight = np.ones(size, dtype=np.float64)

    # update ramping weight on the left side
    spec_left = u < norm_length
    weight[spec_left] = (1 - weight_min) / norm_length * (
        u[spec_left] - norm_length
    ) + 1

    spec_right = u > 1 - norm_length
    weight[spec_right] = (
        -(1 - weight_min) / norm_length * (u[spec_right] - 1 + norm_length) + 1
    )

    return weight


def _normalize_weight(
    weight_ramp: tuple[float, float], coords: np.ndarray
) -> np.ndarray:
    _edge_length, _weight_min = weight_ramp
    length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))
    weight = _construct_ramping_weight(
        _edge_length / length, _weight_min, coords.shape[0]
    )
    return weight


def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords
