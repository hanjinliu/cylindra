from __future__ import annotations
from enum import Enum

from typing import TYPE_CHECKING, Iterable
import numpy as np
from numpy.typing import NDArray
import polars as pl
from acryo import Molecules

from cylindra.const import MoleculesHeader as Mole, PropertyNames as H
from cylindra.utils import assert_column_exists
from cylindra._cylindra_ext import RegionProfiler as _RegionProfiler

if TYPE_CHECKING:
    from cylindra.components import CylSpline


def calc_spacing(mole: Molecules, spl: CylSpline, projective: bool = True) -> pl.Series:
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
        new_feat = pl.Series(Mole.spacing, _y_interv).cast(pl.Float32)
        subsets.append(sub.with_features(new_feat))
    return _concat_groups(subsets).features[Mole.spacing]


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


def calc_twist(mole: Molecules, spl: CylSpline) -> pl.Series:
    """Calculate the twist of each molecule to the next one."""
    subsets = list[Molecules]()
    radius = calc_radius(mole, spl).mean()
    for _, sub in _groupby_with_index(mole, Mole.pf):
        surf = CylinderSurface(spl)
        _interv_vec = np.diff(sub.pos, axis=0, append=0)
        _start = _mole_to_coords(sub)
        _interv_proj = surf.project_vector(_interv_vec, _start)
        _spacing = np.sqrt(np.sum(_interv_proj**2, axis=1))
        _twist_sin = _spacing * surf.long_sin(_interv_proj, _start) / radius
        _twist = np.rad2deg(_arcsin(_twist_sin))
        _twist[-1] = -np.inf
        new_feat = pl.Series(Mole.twist, _twist, dtype=pl.Float32)
        subsets.append(sub.with_features(new_feat))

    return _concat_groups(subsets).features[Mole.twist]


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
    subsets = list[Molecules]()
    mole_ext, new_pf_id = _seam_molecules(mole, spl)
    for i, sub in _groupby_with_index(mole.concat_with(mole_ext), Mole.nth):
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
) -> pl.Series:
    # NOTE: molecules must be in the canonical arrangement.
    subsets = list[Molecules]()
    mole_ext, new_pf_id = _seam_molecules(mole, spl)
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


def _seam_molecules(mole: Molecules, spl: CylSpline) -> tuple[Molecules, int]:
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
            pl.repeat(new_pf_id, pl.count(), dtype=pf_dtype).alias(Mole.pf),
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
        _spl_vec_norm = _norm(self._spl.map(u, der=1))
        _mole_to_spl_vec = _spl_coords - zyx
        return _norm(_cancel_component(_mole_to_spl_vec, _spl_vec_norm))

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
        dpos = _dot(self.spline_vec_norm(start_pos), vec)
        return self.surface_norm(_concat(start_zyx + vec / 2, start_pos + dpos / 2))

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

        start_pos = start[:, 3]
        dpos = _dot(self.spline_vec_norm(start_pos), vec)
        spl_vec_norm = self.spline_vec_norm(start_pos + dpos / 2)
        _cross = np.cross(vec_norm, spl_vec_norm, axis=1)
        _cross_len = np.linalg.norm(_cross, axis=1)
        return _cross_len * self._parallel_to_norm_sign(_cross, start)

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

        if (dtype := feat_label.dtype) not in pl.INTEGER_DTYPES:
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

    def calculate(self, mole: Molecules, spl: CylSpline) -> pl.Series:
        """Calculate this lattice parameter for the given molecule."""
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
    mole0 = mole.with_features([pl.int_range(0, pl.count()).alias(_INDEX_KEY)])
    yield from mole0.groupby(by)


def _concat_groups(subsets: list[Molecules]) -> Molecules:
    """Concatenate groups generated by `_groupby_with_index`."""
    return Molecules.concat(subsets).sort(_INDEX_KEY).drop_features(_INDEX_KEY)