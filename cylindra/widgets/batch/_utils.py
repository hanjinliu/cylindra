from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
import polars as pl
from acryo import Molecules, BatchLoader

from cylindra.project import CylindraProject
from cylindra.const import MoleculesHeader as Mole, PropertyNames as H

if TYPE_CHECKING:
    from cylindra.components import CylSpline

_SPLINE_FEATURES = [
    H.spacing,
    H.twist,
    H.npf,
    H.start,
    H.rise,
    H.radius,
    H.orientation,
]


class TempFeatures:
    """Class to handle temporary features of molecules."""

    def __init__(self, enabled: bool = True):
        self._temp_features = set[str]()
        self._enabled = enabled  # just for compatibility

    @property
    def to_drop(self):
        return self._temp_features

    def read_molecules(self, prj: CylindraProject, mole_abs_path: Path) -> Molecules:
        mole = Molecules.from_file(mole_abs_path)
        spl = _find_source(prj, mole_abs_path)
        features = [pl.repeat(mole_abs_path.stem, pl.count()).alias(Mole.id)]
        if spl is not None and self._enabled:
            for propname in _SPLINE_FEATURES:
                prop = spl.props.get_glob(propname, None)
                if prop is None:
                    continue
                propname_glob = propname + "_glob"
                features.append(pl.repeat(prop, pl.count()).alias(propname_glob))
                self._temp_features.add(propname_glob)
        mole = mole.with_features(features)
        return mole


class LoaderInfo(NamedTuple):
    """Tuple that represents a subtomogram loader."""

    loader: BatchLoader
    name: str
    image_paths: dict[int, Path]

    def rename(self, name: str):
        return LoaderInfo(self.loader, name, self.image_paths)


class PathInfo(NamedTuple):
    """Tuple that represents a child project path."""

    image: Path
    molecules: list[str]
    project: Path


def _find_source(prj: CylindraProject, mole_abs_path: Path) -> CylSpline | None:
    dir = mole_abs_path.parent
    mole_path = mole_abs_path.name
    for info in prj.molecules_info:
        if info.name == mole_path:
            source = info.source
            if source is None:
                return None
            return prj.load_spline(dir, source)
    return None
