from __future__ import annotations
from enum import Enum

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import polars as pl
from acryo import Molecules

from cylindra.const import MoleculesHeader as Mole, PropertyNames as H

if TYPE_CHECKING:
    from cylindra.components import CylSpline


def calc_interval(
    mole: Molecules, spl: CylSpline, projective: bool = True
) -> pl.Series:
    """Calculate the interval of each molecule to the next one."""
    subsets = list[Molecules]()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        surf = CylinderSurface(spl)
        _interv_vec = np.diff(sub.pos, axis=0, append=0)
        if projective:
            _start = _mole_to_coords(sub)
            _interv_vec = surf.project_vector(_interv_vec, _start)
        _y_interv = np.sqrt(np.sum(_interv_vec**2, axis=1))
        _y_interv[-1] = -np.inf  # fill invalid values with 0
        new_feat = pl.Series(Mole.interval, _y_interv).cast(pl.Float32)
        subsets.append(sub.with_features(new_feat))
    return _concat_groups(subsets).features[Mole.interval]


def calc_elevation_angle(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the elevation angle of the longitudinal neighbors."""
    subsets = list[Molecules]()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        surf = CylinderSurface(spl)
        _spl_vec_norm = surf.spline_vec_norm(sub.features[Mole.position])
        _interv_vec_norm = _norm(np.diff(sub.pos, axis=0, append=0))
        _cos: NDArray[np.float32] = _dot(_interv_vec_norm, _spl_vec_norm)
        _deg = np.rad2deg(np.arccos(_cos.clip(-1, 1)))
        _deg[-1] = -np.inf  # fill invalid values with 0
        subsets.append(
            sub.with_features(pl.Series(Mole.elev_angle, _deg).cast(pl.Float32))
        )
    return _concat_groups(subsets).features[Mole.elev_angle]


def calc_skew(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the skew of each molecule to the next one."""
    subsets = list[Molecules]()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        surf = CylinderSurface(spl)
        _interv_vec = np.diff(sub.pos, axis=0, append=0)
        _start = _mole_to_coords(sub)
        _interv_proj = surf.project_vector(_interv_vec, _start)
        _ang = surf.long_angle(_interv_proj, _start)
        _ang[-1] = -np.inf
        new_feat = pl.Series(Mole.skew, _ang, dtype=pl.Float32)
        subsets.append(sub.with_features(new_feat))
    return _concat_groups(subsets).features[Mole.skew]


def calc_dimer_twist(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the dimer twist of each molecule to the next one."""
    subsets = list[Molecules]()
    spacing = spl.props.get_glob(H.spacing)
    radius = spl.props.get_glob(H.radius)
    for _, sub in _groupby_with_index(mole, Mole.pf):
        surf = CylinderSurface(spl)
        _interv_vec = np.diff(sub.pos, axis=0, append=0)
        _start = _mole_to_coords(sub)
        _interv_proj = surf.project_vector(_interv_vec, _start)
        _twist_sin = _arcsin(spacing * surf.long_sin(_interv_proj, _start) / radius)
        _twist = np.rad2deg(2 * _arcsin(_twist_sin))
        _twist[-1] = -np.inf
        new_feat = pl.Series(Mole.dimer_twist, _twist, dtype=pl.Float32)
        subsets.append(sub.with_features(new_feat))

    return _concat_groups(subsets).features[Mole.dimer_twist]


def calc_radius(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the radius of each molecule."""
    _u = spl.y_to_position(mole.features[Mole.position])
    _spl_pos = spl.map(_u, der=0)
    _spl_vec = spl.map(_u, der=1)
    _spl_vec_norm = _norm(_spl_vec)
    _radius_vec = _spl_pos - mole.pos
    result = np.sqrt(_dot(_radius_vec, _radius_vec) - _dot(_radius_vec, _spl_vec_norm))
    return pl.Series(Mole.radius, result).cast(pl.Float32)


def calc_rise_angle(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Add a column of rise angles of each molecule."""
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
        surf = CylinderSurface(spl)
        _interv_vec = np.diff(sub.pos, axis=0, append=np.nan)
        _start = _mole_to_coords(sub)
        interv_proj = surf.project_vector(_interv_vec, _start)
        rise_angles = 90 - surf.long_angle(interv_proj, _start)
        if sub.features[Mole.pf][-1] == new_pf_id:
            rise_angles = rise_angles[:-1]
            sub = sub.subset(slice(None, -1))
        new_feat = pl.Series(Mole.rise, rise_angles).fill_nan(-np.inf).cast(pl.Float32)
        subsets.append(sub.with_features(new_feat))
    return _concat_groups(subsets).features[Mole.rise]


def calc_lateral_interval(
    mole: Molecules, spl: CylSpline, projective: bool = True
) -> pl.DataFrame:
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
        _interv_vec = np.diff(sub.pos, axis=0, append=np.nan)
        if projective:
            surf = CylinderSurface(spl)
            _interv_vec = surf.project_vector(_interv_vec, _mole_to_coords(sub))
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


class CylinderSurface:
    """Class to define the surface of a spline cylinder."""

    def __init__(self, spl: CylSpline):
        self._spl = spl

    def surface_norm(
        self,
        coords: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Get the surface normal vector at the given coordinates.

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
        _mole_to_spl_vec = _spl_coords - zyx
        return _norm(_mole_to_spl_vec)

    def spline_vec_norm(self, pos: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalized spline tangent vector for given positions (nm)."""
        u = self._spl.y_to_position(pos)
        _spl_vec = self._spl.map(u, der=1)
        return _norm(_spl_vec)

    def project_vector(
        self,
        vec: NDArray[np.float32],  # (N, 3)
        start: NDArray[np.float32],  # (N, 4)
    ) -> NDArray[np.float32]:
        """Project vector(s) to the cylinder surface."""
        norm = self._get_vector_surface_norm(vec, start)
        return _cancel_component(vec, norm)

    def _get_vector_surface_norm(
        self,
        vec: NDArray[np.float32],
        start: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        start = np.atleast_2d(start)
        start_zyx = start[:, :3]
        start_pos = start[:, 3]
        end_zyx = start_zyx + vec
        dpos = _dot(self.spline_vec_norm(start_pos), vec)
        end_pos = start_pos + dpos
        surf_norm_start = self.surface_norm(_concat(start_zyx, start_pos))
        surf_norm_end = self.surface_norm(_concat(end_zyx, end_pos))
        return _norm(surf_norm_start + surf_norm_end)

    def _parallel_to_norm_sign(
        self,
        vec: NDArray[np.float32],
        start: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        return np.sign(_dot(vec, self._get_vector_surface_norm(vec, start)))

    def long_sin(
        self,
        vec: NDArray[np.float32],
        start: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Sine of the longitudinal angle between the vector and the spline."""
        start = np.atleast_2d(start)
        vec_norm = _norm(vec)
        spl_vec_norm = self.spline_vec_norm(start[:, 3])
        _cross = np.cross(vec_norm, spl_vec_norm, axis=1)
        return np.linalg.norm(_cross, axis=1) * self._parallel_to_norm_sign(
            _cross, start
        )

    def long_angle(
        self,
        vec: NDArray[np.float32],
        start: NDArray[np.float32],
        degree: bool = True,
    ) -> NDArray[np.float32]:
        """Longitudinal angle between the vector and the spline."""
        angs = _arcsin(self.long_sin(vec, start))
        if degree:
            angs = np.rad2deg(angs)
        return angs


def _arcsin(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """arcsin with clipping."""
    return np.arcsin(x.clip(-1, 1))


class LatticeParameters(Enum):
    interv: LatticeParameters = "interv"
    elev_angle = "elev_angle"
    dimer_twist = "dimer_twist"
    skew = "skew"
    radius = "radius"
    rise = "rise"
    lat_interv = "lat_interv"

    def calculate(self, mole: Molecules, spl: CylSpline) -> pl.Series:
        """Calculate this lattice parameter for the given molecule."""
        if self is LatticeParameters.interv:
            return calc_interval(mole, spl)
        elif self is LatticeParameters.elev_angle:
            return calc_elevation_angle(mole, spl)
        elif self is LatticeParameters.dimer_twist:
            return calc_dimer_twist(mole, spl)
        elif self is LatticeParameters.skew:
            return calc_skew(mole, spl)
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
    mole0 = mole.with_features([pl.int_range(0, pl.count()).alias(_INDEX_KEY)])
    yield from mole0.groupby(by)


def _concat_groups(subsets: list[Molecules]) -> Molecules:
    """Concatenate groups generated by `_groupby_with_index`."""
    return Molecules.concat(subsets).sort(_INDEX_KEY).drop_features(_INDEX_KEY)
