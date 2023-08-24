from __future__ import annotations
from enum import Enum

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import polars as pl
from acryo import Molecules

from cylindra.const import (
    MoleculesHeader as Mole,
    PropertyNames as H,
)

if TYPE_CHECKING:
    from cylindra.components import CylSpline


def calc_interval(
    mole: Molecules, spl: CylSpline, projective: bool = True
) -> pl.Series:
    """Calculate the interval of each molecule to the next one."""
    _spl_len = spl.length()
    subsets = list[Molecules]()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        _pos = sub.pos
        _interv_vec = np.diff(_pos, axis=0, append=0)
        _u = sub.features[Mole.position] / _spl_len
        if projective:
            _spl_vec_norm = _norm(spl.map(_u, der=1))
            _y_interv = np.abs(_dot(_interv_vec, _spl_vec_norm))
        else:
            _y_interv = np.sqrt(np.sum(_interv_vec**2, axis=1))
        _y_interv[-1] = -np.inf  # fill invalid values with 0
        subsets.append(
            sub.with_features(pl.Series(Mole.interval, _y_interv).cast(pl.Float32))
        )
    return _concat_groups(subsets).features[Mole.interval]


def calc_elevation_angle(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the elevation angle of the longitudinal neighbors."""
    _spl_len = spl.length()
    subsets = list[Molecules]()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        _pos = sub.pos
        _interv_vec = np.diff(_pos, axis=0, append=0)

        _u = sub.features[Mole.position] / _spl_len
        _spl_vec = spl.map(_u, der=1)

        _cos = _dot(_interv_vec, _spl_vec) / (
            np.linalg.norm(_interv_vec, axis=1) * np.linalg.norm(_spl_vec, axis=1)
        )
        if not np.all((-1 <= _cos) & (_cos <= 1)):
            raise ValueError(
                f"Cosine values must be in range [-1, 1] but got:\n{_cos!r}"
            )
        _deg = np.rad2deg(np.arccos(_cos))
        _deg[-1] = -np.inf  # fill invalid values with 0
        subsets.append(
            sub.with_features(pl.Series(Mole.elev_angle, _deg).cast(pl.Float32))
        )
    return _concat_groups(subsets).features[Mole.elev_angle]


def calc_skew_tilt(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the skew of each molecule to the next one."""
    _spl_len = spl.length()
    subsets = list[Molecules]()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        _pos = sub.pos
        _interv_vec = np.diff(_pos, axis=0, append=0)

        _u = sub.features[Mole.position] / _spl_len
        _spl_pos = spl.map(_u, der=0)
        _spl_vec = spl.map(_u, der=1)

        _mole_to_spl_vec = _spl_pos - _pos
        _interv_proj_norm = _norm(_cancel_component(_interv_vec, _mole_to_spl_vec))

        _spl_vec_norm = _norm(_spl_vec)

        _skew_cross = np.cross(_interv_proj_norm, _spl_vec_norm, axis=1)
        _inner = _dot(_skew_cross, _mole_to_spl_vec)
        _skew_sin = np.linalg.norm(_skew_cross, axis=1) * np.sign(_inner)

        _skew = np.rad2deg(np.arcsin(_skew_sin))
        _skew[-1] = -np.inf
        subsets.append(
            sub.with_features(pl.Series(Mole.skew_tilt, _skew).cast(pl.Float32))
        )

    return _concat_groups(subsets).features[Mole.skew_tilt]


def calc_skew(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the skew of each molecule to the next one."""
    _spl_len = spl.length()
    subsets = list[Molecules]()
    spacing = spl.props.get_glob(H.spacing)
    for _, sub in _groupby_with_index(mole, Mole.pf):
        _pos = sub.pos
        _interv_vec = np.diff(_pos, axis=0, append=0)

        _u = sub.features[Mole.position] / _spl_len
        _spl_pos = spl.map(_u, der=0)
        _spl_vec = spl.map(_u, der=1)

        _mole_to_spl_vec = _spl_pos - _pos
        _interv_proj_norm = _norm(_cancel_component(_interv_vec, _mole_to_spl_vec))
        _radius = np.linalg.norm(_mole_to_spl_vec, axis=1)

        _spl_vec_norm = _norm(_spl_vec)

        _skew_cross = np.cross(_interv_proj_norm, _spl_vec_norm, axis=1)
        _inner = _dot(_skew_cross, _mole_to_spl_vec)
        _skew_sin = np.linalg.norm(_skew_cross, axis=1) * np.sign(_inner)

        _skew = np.rad2deg(2 * spacing * _skew_sin / _radius)
        _skew[-1] = -np.inf
        subsets.append(sub.with_features(pl.Series(Mole.skew, _skew).cast(pl.Float32)))

    return _concat_groups(subsets).features[Mole.skew]


def calc_radius(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the radius of each molecule."""
    _u = mole.features[Mole.position] / spl.length()
    _spl_pos = spl.map(_u, der=0)
    _spl_vec = spl.map(_u, der=1)
    _spl_vec_norm = _norm(_spl_vec)
    _radius_vec = _spl_pos - mole.pos
    result = np.sqrt(_dot(_radius_vec, _radius_vec) - _dot(_radius_vec, _spl_vec_norm))
    return pl.Series(Mole.radius, result).cast(pl.Float32)


def calc_rise_angle(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Add a column of rise angles of each molecule."""
    _spl_len = spl.length()
    subsets = list[Molecules]()
    _nrise = int(round(spl.props.get_glob(H.start))) * spl.config.rise_sign
    new_pf_id = mole.features[Mole.pf].max() + 1
    nth_dtype = mole.features[Mole.nth].dtype
    pf_dtype = mole.features[Mole.pf].dtype
    mole_ext = (
        mole.filter(pl.col(Mole.pf) == 0)
        .with_features(
            pl.int_range(0, pl.count(), dtype=nth_dtype).alias(Mole.nth) - _nrise,
            pl.repeat(new_pf_id, pl.count(), dtype=pf_dtype).alias(Mole.pf),
        )
        .filter(pl.col(Mole.nth).is_between(0, mole.count() - 1))
    )
    for _, sub in _groupby_with_index(mole.concat_with(mole_ext), Mole.nth):
        sub = sub.sort(Mole.pf)
        _pos = sub.pos
        _u = sub.features[Mole.position] / _spl_len
        _spl_pos = spl.map(_u, der=0)
        _spl_vec = spl.map(_u, der=1)
        _spl_vec_norm = _norm(_spl_vec)

        _mole_to_spl_vec = _spl_pos - _pos
        _interv_vec = np.diff(_pos, axis=0, append=np.nan)
        _interv_proj_norm = _norm(_cancel_component(_interv_vec, _mole_to_spl_vec))
        _rise_cross = np.cross(_interv_proj_norm, _spl_vec_norm, axis=1)
        _inner = _dot(_rise_cross, _mole_to_spl_vec)
        _rise_sin = np.linalg.norm(_rise_cross, axis=1) * np.sign(_inner)
        rise_angles = 90 - np.rad2deg(np.arcsin(_rise_sin))
        if sub.features[Mole.pf][-1] == new_pf_id:
            rise_angles = rise_angles[:-1]
            sub = sub.subset(slice(None, -1))
        subsets.append(
            sub.with_features(
                pl.Series(Mole.rise, rise_angles).fill_nan(-np.inf).cast(pl.Float32)
            )
        )
    return _concat_groups(subsets).features[Mole.rise]


def calc_lateral_interval(
    mole: Molecules, spl: CylSpline, projective: bool = True
) -> pl.DataFrame:
    _spl_len = spl.length()
    subsets = list[Molecules]()
    _nrise = int(round(spl.props.get_glob(H.start))) * spl.config.rise_sign
    new_pf_id = mole.features[Mole.pf].max() + 1
    nth_dtype = mole.features[Mole.nth].dtype
    pf_dtype = mole.features[Mole.pf].dtype
    mole_ext = (
        mole.filter(pl.col(Mole.pf) == 0)
        .with_features(
            pl.int_range(0, pl.count(), dtype=nth_dtype).alias(Mole.nth) - _nrise,
            pl.repeat(new_pf_id, pl.count(), dtype=pf_dtype).alias(Mole.pf),
        )
        .filter(pl.col(Mole.nth).is_between(0, mole.count() - 1))
    )
    for _, sub in _groupby_with_index(mole.concat_with(mole_ext), Mole.nth):
        sub = sub.sort(Mole.pf)
        _pos = sub.pos
        _u = sub.features[Mole.position] / _spl_len
        _spl_pos = spl.map(_u, der=0)

        _mole_to_spl_vec = _spl_pos - _pos
        _interv_vec = np.diff(_pos, axis=0, append=np.nan)
        if projective:
            _interv_proj = _cancel_component(_interv_vec, _mole_to_spl_vec)
            _interv_abs = np.sqrt(_dot(_interv_proj, _interv_proj))
        else:
            _interv_abs = np.sqrt(_dot(_interv_vec, _interv_vec))

        if sub.features[Mole.pf][-1] == new_pf_id:
            _interv_abs = _interv_abs[:-1]
            sub = sub.subset(slice(None, -1))
        subsets.append(
            sub.with_features(
                pl.Series(Mole.lateral_interval, _interv_abs)
                .fill_nan(-np.inf)
                .cast(pl.Float32)
            )
        )
    return _concat_groups(subsets).features[Mole.lateral_interval]


class LatticeParameters(Enum):
    interv: LatticeParameters = "interv"
    elev_angle = "elev_angle"
    skew = "skew"
    skew_tilt = "skew_tilt"
    radius = "radius"
    rise = "rise"
    lat_interv = "lat_interv"

    def calculate(self, mole: Molecules, spl: CylSpline) -> pl.Series:
        if self is LatticeParameters.interv:
            return calc_interval(mole, spl)
        elif self is LatticeParameters.elev_angle:
            return calc_elevation_angle(mole, spl)
        elif self is LatticeParameters.skew:
            return calc_skew(mole, spl)
        elif self is LatticeParameters.skew_tilt:
            return calc_skew_tilt(mole, spl)
        elif self is LatticeParameters.radius:
            return calc_radius(mole, spl)
        elif self is LatticeParameters.rise:
            return calc_rise_angle(mole, spl)
        elif self is LatticeParameters.lat_interv:
            return calc_lateral_interval(mole, spl)
        else:
            raise ValueError(f"Unknown lattice parameter {self!r}.")

    @classmethod
    def choices(cls) -> list[str]:
        return [v.name for v in cls]


def _norm(vec: NDArray[np.float32]) -> NDArray[np.float32]:
    vec_len = np.linalg.norm(vec, axis=1)
    return vec / vec_len[:, np.newaxis]


def _dot(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    """Vectorized dot product."""
    return np.sum(a * b, axis=1)


def _cancel_component(
    vec: NDArray[np.float32], other: NDArray[np.float32]
) -> NDArray[np.float32]:
    other_norm = _norm(other)
    to_cancel = _dot(vec, other_norm)[:, np.newaxis] * other_norm
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
    mole0 = mole.with_features([pl.int_range(0, pl.count()).alias(_INDEX_KEY)])
    yield from mole0.groupby(by)


def _concat_groups(subsets: list[Molecules]) -> Molecules:
    """Concatenate groups generated by `_groupby_with_index`."""
    return Molecules.concat(subsets).sort(_INDEX_KEY).drop_features(_INDEX_KEY)
