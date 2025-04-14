from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from acryo import Molecules
from acryo.molecules import axes_to_rotator
from numpy.typing import NDArray

from cylindra import utils

if TYPE_CHECKING:
    from typing import Self

    from cylindra.components import Spline


class InterMoleculeNet:
    """Interaction network between molecules and molecules."""

    def __init__(
        self,
        molecules_origin: Molecules,
        molecules_target: Molecules,
        indices_origin: NDArray[np.intp],
        indices_target: NDArray[np.intp],
        features: pl.DataFrame | None = None,
    ):
        # type and shape check
        if indices_origin.shape != indices_target.shape:
            raise ValueError("origin and target indices must have the same shape.")
        if indices_origin.ndim != 1:
            raise ValueError("indices must be 1D array.")
        if indices_origin.dtype != np.intp:
            raise ValueError("indices must be integer array.")
        if indices_target.dtype != np.intp:
            raise ValueError("indices must be integer array.")
        if not isinstance(features, pl.DataFrame):
            features = pl.DataFrame(features)
        if features.shape[1] > 0 and features.shape[0] != indices_origin.shape[0]:
            raise ValueError(
                "features must have the same length as the number of interactions."
            )
        self.molecules_origin = molecules_origin
        self.molecules_target = molecules_target
        self.indices_origin = indices_origin
        self.indices_target = indices_target
        self._features = features

    @property
    def features(self) -> pl.DataFrame:
        return self._features

    @features.setter
    def features(self, df):
        self._features = pl.DataFrame(df)

    @property
    def origin(self) -> NDArray[np.float32]:
        """The origin coordinates."""
        return self.molecules_origin.pos[self.indices_origin]

    @property
    def target(self) -> NDArray[np.float32]:
        """The target coordinates."""
        return self.molecules_target.pos[self.indices_target]

    def count(self) -> int:
        """Number of interactions."""
        return self.indices_origin.shape[0]

    def distances(self) -> NDArray[np.float32]:
        """Return distances between origin and target."""
        return np.linalg.norm(self.vectors(), axis=-1)

    def vectors(self) -> NDArray[np.float32]:
        """Return vectors from origin to target."""
        return self.target - self.origin

    @classmethod
    def from_molecules(
        cls,
        mole_from: Molecules,
        mole_to: Molecules,
        min_distance: float = 0.0,
        max_distance: float = 10.0,
    ) -> InterMoleculeNet:
        """Create interaction network from molecules."""
        _check_low_high(min_distance, max_distance)
        dist = utils.distance_matrix(mole_from.pos, mole_to.pos)
        orig_indices = []
        targ_indices = []
        orig_indices, targ_indices = np.where(
            (dist <= max_distance) & (dist >= min_distance)
        )
        return InterMoleculeNet(
            mole_from,
            mole_to,
            np.array(orig_indices),
            np.array(targ_indices),
        )

    @classmethod
    def from_molecules_closest(
        cls,
        mole_from: Molecules,
        mole_to: Molecules,
    ) -> InterMoleculeNet:
        dist = utils.distance_matrix(mole_from.pos, mole_to.pos)
        indices_targ = np.argmin(dist, axis=1)
        indices_orig = np.arange(mole_from.count())
        return InterMoleculeNet(
            mole_from,
            mole_to,
            indices_orig,
            indices_targ,
        )

    def with_indices(
        self, origin: NDArray[np.intp], target: NDArray[np.intp]
    ) -> InterMoleculeNet:
        """Create interaction network with new indices."""
        if origin.shape != target.shape:
            raise ValueError("origin and target indices must have the same shape.")
        return InterMoleculeNet(
            self.molecules_origin,
            self.molecules_target,
            origin,
            target,
        )

    def with_features(self, expr, *more_expr, **named_expr) -> InterMoleculeNet:
        """Create interaction network with features."""
        df = self.features.with_columns(expr, *more_expr, **named_expr)
        return InterMoleculeNet(
            self.molecules_origin,
            self.molecules_target,
            self.indices_origin,
            self.indices_target,
            features=df,
        )

    def with_standard_features(self) -> InterMoleculeNet:
        """Add standard features to this net."""
        dist = self.distances()
        dot_orig = self.dot_product_origin()
        dot_targ = self.dot_product_target()
        df = {
            "distance": dist,
            "projection-origin-z": _safe_div(dot_orig[:, 0], dist),
            "projection-origin-y": _safe_div(dot_orig[:, 1], dist),
            "projection-origin-x": _safe_div(dot_orig[:, 2], dist),
            "projection-target-z": _safe_div(dot_targ[:, 0], dist),
            "projection-target-y": _safe_div(dot_targ[:, 1], dist),
            "projection-target-x": _safe_div(dot_targ[:, 2], dist),
        }
        return self.with_features(pl.DataFrame(df))

    def filter(self, predicate) -> InterMoleculeNet:
        """Filter interactions by feature."""
        index_column = pl.Series("._index", np.arange(self.count()))

        df = self.features.with_columns(index_column).filter(predicate)
        index_filt = df[index_column.name].to_numpy()
        return self.with_indices(
            self.indices_origin[index_filt],
            self.indices_target[index_filt],
        ).with_features(df.drop(index_column.name))

    #### IO ####

    @classmethod
    def from_dir(cls, path: str | Path) -> Self:
        """Load interaction network from a directory."""
        path = Path(path)
        mole_from = Molecules.from_parquet(path / "molecules_origin.parquet")
        mole_to = Molecules.from_parquet(path / "molecules_target.parquet")
        indices_from = np.load(path / "indices_origin.npy")
        indices_to = np.load(path / "indices_target.npy")
        features = pl.read_parquet(path / "features.parquet")
        return cls(mole_from, mole_to, indices_from, indices_to, features)

    def save(self, path: str | Path):
        """Save interaction network to a directory."""
        path = Path(path)
        if path.suffix != "":
            raise ValueError(f"Must be a directory, got {path}")
        path.mkdir(exist_ok=False)
        self.molecules_origin.to_parquet(path / "molecules_origin.parquet")
        self.molecules_target.to_parquet(path / "molecules_target.parquet")
        np.save(path / "indices_origin.npy", self.indices_origin)
        np.save(path / "indices_target.npy", self.indices_target)
        self.features.write_parquet(path / "features.parquet")

    #### Vector operations ####

    def dot_product_origin(self) -> NDArray[np.float32]:
        """Return a matrix of dot products between origin molecules and interaction.

        This method returns a 2D array of

        ```
        [[e0z*v, e0y*v, e0x*v],
         [e1z*v, e1y*v, e1x*v],
         ...]
        ```

        where `eiz` is the z-axis of the `i`-th molecule.
        """
        return self._dot_product_impl(self.molecules_origin, self.indices_origin)

    def dot_product_target(self) -> NDArray[np.float32]:
        """Return a matrix of dot products between target molecules and interaction.

        This method returns a 2D array of

        ```
        [[e0z*v, e0y*v, e0x*v],
         [e1z*v, e1y*v, e1x*v],
         ...]
        ```

        where `eiz` is the z-axis of the `i`-th molecule.
        """
        return self._dot_product_impl(self.molecules_target, self.indices_target)

    def _dot_product_impl(self, mole: Molecules, indices) -> NDArray[np.float32]:
        mx = mole.x
        my = mole.y
        mz = mole.z
        vec = self.vectors()
        dot_x = np.sum(mx[indices] * vec, axis=-1)
        dot_y = np.sum(my[indices] * vec, axis=-1)
        dot_z = np.sum(mz[indices] * vec, axis=-1)
        return np.stack([dot_z, dot_y, dot_x], axis=-1)


def _check_low_high(low, high):
    if low >= high:
        raise ValueError("lower bound must be less than higher bound.")
    if low < 0:
        raise ValueError("lower bound must be greater than or equal to 0.")
    return low, high


def closest_positions(
    mole: Molecules,
    spl: Spline,
    precision: float = 1.0,
) -> NDArray[np.float32]:
    """Calculate the closest positions of a set of points to a spline."""

    npartitions = utils.ceilint(spl.length() / precision)
    u = np.linspace(0, 1, npartitions)
    sample_points = spl.map(u)
    dist = utils.distance_matrix(mole.pos, sample_points)
    u_opt = np.argmin(dist, axis=1) / npartitions
    return u_opt


def align_molecules_to_spline(mole: Molecules, spl: Spline) -> Molecules:
    """Align molecule rotation to the spline.

    The y-axis of the molecule is aligned to the tangent of the spline, while z-axis
    will be aligned to the normal of the spline.
    """
    u_opt = closest_positions(mole, spl)
    tangent = spl.map(u_opt, der=1)
    normal = spl.map(u_opt) - mole.pos
    rotator = axes_to_rotator(_vec_normed(normal), _vec_normed(tangent))
    return Molecules(mole.pos, rotator, features=mole.features)


def _vec_normed(vec: NDArray[np.floating]) -> NDArray[np.floating]:
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
