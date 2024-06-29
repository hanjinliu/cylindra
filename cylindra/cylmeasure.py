from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterable, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from cylindra._cylindra_ext import RegionProfiler as _RegionProfiler
from cylindra.const import POLARS_INTEGER_DTYPES
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.utils import assert_column_exists

if TYPE_CHECKING:
    from acryo import Molecules

    from cylindra._napari import MoleculesLayer
    from cylindra.components import CylSpline


def calc_spacing(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the interval of each molecule to the next one."""
    return (
        calc_localvec_long(mole, spl, fill=np.nan)
        .select(
            (pl.col("vecy") ** 2 + pl.col("veca") ** 2)
            .sqrt()
            .fill_nan(-float("inf"))
            .alias(Mole.spacing)
        )
        .to_series()
    )


def calc_elevation_angle(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the elevation angle of the longitudinal neighbors."""
    return (
        calc_localvec_long(mole, spl, fill=np.nan)
        .select(
            pl.arctan2("vecr", "vecy")
            .degrees()
            .fill_nan(-float("inf"))
            .alias(Mole.elev_angle)
        )
        .to_series()
    )


def calc_skew(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the skew of each molecule to the next one."""
    return (
        calc_localvec_long(mole, spl, fill=np.nan)
        .select(
            pl.arctan2("veca", "vecy")
            .degrees()
            .fill_nan(-float("inf"))
            .alias(Mole.skew)
        )
        .to_series()
    )


def calc_twist(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the twist of each molecule to the next one."""
    return (
        calc_localvec_long(mole, spl, fill=np.nan)
        .select(
            (pl.col("veca") / 2 / spl.radius)
            .arcsin()
            .degrees()
            .fill_nan(-float("inf"))
            .alias(Mole.twist)
            * 2
        )
        .to_series()
    )


def calc_radius(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the radius of each molecule."""
    _u = spl.y_to_position(mole.features[Mole.position])
    _spl_pos = spl.map(_u, der=0)
    _spl_vec = spl.map(_u, der=1)
    _spl_vec_norm = _norm(_spl_vec)
    _radius_vec = _spl_pos - mole.pos
    result = np.sqrt(_dot(_radius_vec, _radius_vec) - _dot(_radius_vec, _spl_vec_norm))
    return pl.Series(Mole.radius, result).cast(pl.Float32)


def calc_rise(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Add a column of rise angles of each molecule."""
    # NOTE: molecules must be in the canonical arrangement.
    sign = spl.config.rise_sign
    return (
        calc_localvec_lat(mole, spl, fill=np.nan)
        .select(
            pl.arctan2("vecy", "veca")
            .degrees()
            .fill_nan(-float("inf"))
            .alias(Mole.rise)
        )
        .to_series()
        * sign
    )


def calc_lateral_interval(mole: Molecules, spl: CylSpline) -> pl.Series:
    # NOTE: molecules must be in the canonical arrangement.
    return (
        calc_localvec_lat(mole, spl, fill=np.nan)
        .select(
            (pl.col("vecy") ** 2 + pl.col("veca") ** 2)
            .sqrt()
            .fill_nan(-float("inf"))
            .alias(Mole.lateral_interval)
        )
        .to_series()
    )


def _pad_molecules_at_seam(mole: Molecules, spl: CylSpline) -> tuple[Molecules, int]:
    """
    Molecules at the seam boundary.

    If the molecules are N-pf, molecules corresponding to the (N+1)-th pf will be
    returned. This is useful for calculating the structures that need the lateral
    interaction.
    """
    _nrise = int(round(spl.props.get_glob(H.start))) * spl.config.rise_sign
    new_pf_id = mole.features[Mole.pf].max() + 1
    pf_dtype = mole.features[Mole.pf].dtype
    nth_rng = mole.features[Mole.nth].min(), mole.features[Mole.nth].max()
    mole_ext = (
        mole.filter(pl.col(Mole.pf) == 0)
        .with_features(
            pl.col(Mole.nth) - _nrise,
            pl.repeat(new_pf_id, pl.len(), dtype=pf_dtype).alias(Mole.pf),
        )
        .filter(pl.col(Mole.nth).is_between(*nth_rng))
    )
    return mole_ext, new_pf_id


def calc_curve_index(mole: Molecules, spl: CylSpline):
    """
    The curve orientation index.

    The curve orientation is defined as the cosine of the angle between the
    second derivative and the relative molecule vector. That is, the inside
    of the curve is positive.
    """
    _u = spl.y_to_position(mole.features[Mole.position])
    der0 = spl.map(_u, der=0)
    der2 = spl.map(_u, der=2)
    mole_vec_normed = _norm(mole.pos - der0)
    der2_normed = _norm(der2, fill=0.0)
    _cos = _dot(mole_vec_normed, der2_normed)
    return pl.Series(Mole.curve_index, _cos).cast(pl.Float32)


def calc_localvec_long(
    mole: Molecules, spl: CylSpline, fill: float = 0.0
) -> pl.DataFrame:
    subsets = list["Molecules"]()
    surf = CylinderSurface(spl)
    for _, sub in _groupby_with_index(mole, Mole.pf):
        _interv_vec = np.diff(sub.pos, axis=0, append=0)
        _start = _mole_to_coords(sub)
        _vec_tr = surf.transform_vector(_interv_vec, _start)
        _vec_tr[-1, :] = fill
        subsets.append(
            sub.with_features(
                pl.Series("vecr", _vec_tr[:, 0], dtype=pl.Float32),
                pl.Series("vecy", _vec_tr[:, 1], dtype=pl.Float32),
                pl.Series("veca", _vec_tr[:, 2], dtype=pl.Float32),
            )
        )
    return _concat_groups(subsets).features.select("vecr", "vecy", "veca")


def calc_localvec_lat(
    mole: Molecules, spl: CylSpline, fill: float = 0.0
) -> pl.DataFrame:
    subsets = list["Molecules"]()
    mole_ext, new_pf_id = _pad_molecules_at_seam(mole, spl)
    surf = CylinderSurface(spl)
    for _, sub in _groupby_with_index(mole.concat_with(mole_ext), Mole.nth):
        sub0 = sub.sort(Mole.pf)
        _interv_vec = np.diff(sub0.pos, axis=0, append=np.nan)
        _start = _mole_to_coords(sub0)
        _vec_tr = surf.transform_vector(_interv_vec, _start)
        if sub0.features[Mole.pf][-1] == new_pf_id:
            _vec_tr = _vec_tr[:-1, :]
            sub0 = sub0.subset(slice(None, -1))
        else:
            _vec_tr[:-1, :] = fill
        subsets.append(
            sub0.with_features(
                pl.Series("vecr", _vec_tr[:, 0], dtype=pl.Float32),
                pl.Series("vecy", _vec_tr[:, 1], dtype=pl.Float32),
                pl.Series("veca", _vec_tr[:, 2], dtype=pl.Float32),
            )
        )
    return _concat_groups(subsets).features.select("vecr", "vecy", "veca")


class CylinderSurface:
    """Class to define the surface of a spline cylinder."""

    def __init__(self, spl: CylSpline):
        self._spl = spl
        self._radius = spl.radius
        if self._radius is None:
            raise ValueError("The spline must have a radius.")

    def transform_vector(
        self,
        vec: NDArray[np.float32],  # (N, 3)
        start: NDArray[np.float32],  # (N, 4)
    ) -> NDArray[np.float32]:
        """Transform vector(s) to (r, y, a)."""
        start = np.atleast_2d(start)
        start_zyx = start[:, :3]
        start_pos = start[:, 3]

        # the displacement in the spline coordinate
        dpos = _dot(self._spline_vec_norm(start_pos), vec)
        end_pos = start_pos + dpos
        dy = _dot(self._spline_vec_norm(start_pos + dpos / 2), vec)

        # surface normal vectors at the start and end points
        er0 = self._surface_vec(_concat(start_zyx, start_pos))
        er1 = self._surface_vec(_concat(start_zyx + vec, end_pos))
        ey0 = self._spline_vec_norm(start_pos)
        ey1 = self._spline_vec_norm(end_pos)

        # the "mean" unit vectors
        ey = _norm(ey0 + ey1)
        er = _norm(er0 + er1)
        ea = np.cross(er, ey, axis=1)

        er0y = _norm(_cancel_component(er0, ey))
        er1y = _norm(_cancel_component(er1, ey))
        v_ang_len = 2 * self._radius * _half_sin(er0y, er1y)
        _sign = np.sign(_dot(_norm(er1y - er0y), ea))
        v_ang = v_ang_len * _sign
        v_ang[np.isnan(v_ang)] = 0.0
        return np.stack([_dot(vec, er), dy, v_ang], axis=1)

    def _surface_vec(
        self,
        coords: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Get the surface vector at the given coordinates.

        Parameters
        ----------
        coords : (N, 4) array
            Coordinate of points. The last column is the spline parameter.
        """
        coords = np.atleast_2d(coords)
        assert coords.ndim == 2 and coords.shape[1] == 4
        zyx = coords[:, :3]
        u = self._spl.y_to_position(coords[:, 3])
        _spl_coords = self._spl.map(u, der=0)
        _spl_vec_norm = _norm(self._spl.map(u, der=1))
        _mole_to_spl_vec = _spl_coords - zyx
        return _norm(_cancel_component(_mole_to_spl_vec, _spl_vec_norm))

    def _spline_vec_norm(self, pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalized spline tangent vector for given positions (nm)."""
        u = self._spl.y_to_position(pos)
        _spl_vec = self._spl.map(u, der=1)
        return _norm(_spl_vec)


class RegionProfiler:
    CHOICES = [
        "area", "length", "width", "sum", "mean", "median", "max", "min", "std",
    ]  # fmt: skip

    def __init__(self, rust_obj: _RegionProfiler) -> None:
        self._rust_obj = rust_obj

    @classmethod
    def from_arrays(
        cls,
        image: NDArray[np.float32],
        label_image: NDArray[np.uint32],
        nrise: int,
    ) -> RegionProfiler:
        return cls(_RegionProfiler.from_arrays(image, label_image, nrise))

    @classmethod
    def from_components(
        cls,
        mole: Molecules,
        spl: CylSpline,
        target: str = "nth",
        label: str = "pf-id",
    ) -> RegionProfiler:
        """
        Construct a region profiler from molecules and splines.

        Parameters
        ----------
        mole : Molecules
            Molecules to be profiled. Must have features "nth", "pf-id".
        spl : CylSpline
            Spline from which the molecules are generated.
        target : str, optional
            Column name of the target values. This is not needed if properties that
            do not require target values are to be calculated.
        label : str
            Column name of the label values. Must be an integer column.
        """
        feat = mole.features
        assert_column_exists(feat, [target, label])
        feat_label = feat[label]

        if (dtype := feat_label.dtype) not in POLARS_INTEGER_DTYPES:
            raise TypeError(f"label must be an integer column, got {dtype}.")
        nth = feat[Mole.nth].cast(pl.Int32).to_numpy()
        pf = feat[Mole.pf].cast(pl.Int32).to_numpy()
        values = feat[target].cast(pl.Float32).to_numpy()
        labels = feat_label.cast(pl.UInt32).to_numpy()
        nrise = spl.nrise()
        npf = spl.props.get_glob(H.npf)

        reg = _RegionProfiler.from_features(nth, pf, values, labels, npf, nrise)

        return cls(reg)

    def calculate(self, props: Iterable[str], *more_props: str) -> pl.DataFrame:
        """
        Calculate properties for each region.

        Parameters
        ----------
        props : str or list of str
            Property names. Must be chosen from following:
            - area: total number of molecules.
            - length: longitudinal length of the region.
            - width: lateral width of the region.
            - sum: sum of target values.
            - mean: mean of target values.
            - median: median of target values.
            - max: max of target values.
            - min: min of target values.
            - std: standard deviation of target values.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns corresponding to the given property names.
        """
        if isinstance(props, str):
            all_props = [props, *more_props]
        else:
            if more_props:
                raise TypeError(
                    "Must be calculate(str, str, ...) or calculate([str, str, ...])"
                )
            all_props = list(props)
        props = list(props)
        # NOTE: output dict is not sorted
        out = self._rust_obj.calculate(all_props)
        return pl.DataFrame({k: out[k] for k in all_props})

    def n_regions(self) -> int:
        """Number of regions."""
        _area = "area"
        return self._rust_obj.calculate([_area])[_area].size


class LatticeParameters(Enum):
    spacing = "spacing"
    elev_angle = "elev_angle"
    twist = "twist"
    skew_angle = "skew_angle"
    radius = "radius"
    rise_angle = "rise_angle"
    lat_interv = "lat_interv"
    curve_index = "curve_index"

    @overload
    def calculate(self, mole: Molecules, spl: CylSpline) -> pl.Series:
        ...

    @overload
    def calculate(self, layer: MoleculesLayer) -> pl.Series:
        ...

    def calculate(self, mole, spl=None):
        """Calculate this lattice parameter for the given molecule."""
        if spl is None:
            from cylindra._napari import MoleculesLayer

            if not isinstance(mole, MoleculesLayer):
                raise TypeError("mole must be a MoleculesLayer.")
            mole, spl = mole.molecules, mole.source_spline
            if spl is None:
                raise ValueError("The source spline is not defined.")
        match self:
            case LatticeParameters.spacing:
                return calc_spacing(mole, spl)
            case LatticeParameters.elev_angle:
                return calc_elevation_angle(mole, spl)
            case LatticeParameters.twist:
                return calc_twist(mole, spl)
            case LatticeParameters.skew_angle:
                return calc_skew(mole, spl)
            case LatticeParameters.radius:
                return calc_radius(mole, spl)
            case LatticeParameters.rise_angle:
                return calc_rise(mole, spl)
            case LatticeParameters.lat_interv:
                return calc_lateral_interval(mole, spl)
            case LatticeParameters.curve_index:
                return calc_curve_index(mole, spl)
            case _:  # pragma: no cover
                raise ValueError(f"Unknown lattice parameter {self!r}.")

    @classmethod
    def choices(cls) -> list[str]:
        return [v.name for v in cls]


def _norm(vec: NDArray[np.float32], fill=np.nan) -> NDArray[np.float32]:
    """Normalize vectors."""
    vec_len = np.linalg.norm(vec, axis=1)
    vec_len[vec_len == 0] = np.nan
    out = vec / vec_len[:, np.newaxis]
    out[np.isnan(out)] = fill
    return out


def _dot(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Vectorized dot product."""
    return np.sum(a * b, axis=1)


def _concat(
    coords: NDArray[np.float32], pos: NDArray[np.float32]
) -> NDArray[np.float32]:
    return np.concatenate([coords, pos.reshape(-1, 1)], axis=1)


def _mole_to_coords(mole: Molecules) -> NDArray[np.float32]:
    """Convert molecules to (N, 4) coordinates."""
    return _concat(mole.pos, mole.features[Mole.position].to_numpy())


def _cancel_component(
    vec: NDArray[np.float32], other: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Cancel the `other` component from `vec`."""
    to_cancel = _dot(vec, other)[:, np.newaxis] * other
    out = vec - to_cancel
    return out


_INDEX_KEY = "._index_column"


def _groupby_with_index(mole: Molecules, by: str):
    """
    Call groupby with an additional index column.

    This groupby function adds an index column for the later ordered concatenation.
    """
    if _INDEX_KEY in mole.features.columns:
        raise ValueError(f"Column name {_INDEX_KEY!r} already exists.")
    mole0 = mole.with_features([pl.int_range(0, pl.len()).alias(_INDEX_KEY)])
    yield from mole0.groupby(by)


def _concat_groups(subsets: list[Molecules]) -> Molecules:
    """Concatenate groups generated by `_groupby_with_index`."""
    from acryo import Molecules

    return Molecules.concat(subsets).sort(_INDEX_KEY).drop_features(_INDEX_KEY)


def _half_sin(er0y, er1y):
    """Calculate sin(a/2) from two vectors, where a is the angle between them."""
    # sin^2(a) = (1 - cos(2a)) / 2, when 0 < a < pi
    cos2a = _dot(er0y, er1y) / np.sqrt(_dot(er0y, er0y) * _dot(er1y, er1y))
    sina = np.sqrt((1 - np.clip(cos2a, -1, 1)) / 2)
    return sina
