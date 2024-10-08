from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, NamedTuple

import impy as ip
import polars as pl
from acryo import BatchLoader, Molecules

from cylindra._config import get_config
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.project import CylindraProject

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

    def read_molecules(
        self,
        mole_abs_path: Path,
        project: CylindraProject | None = None,
    ) -> Molecules:
        mole = Molecules.from_file(mole_abs_path)
        spl = _find_source(mole_abs_path, project)
        features = [pl.repeat(mole_abs_path.stem, pl.len()).alias(Mole.id)]
        if spl is not None and self._enabled:
            for propname in _SPLINE_FEATURES:
                prop = spl.props.get_glob(propname, None)
                if prop is None:
                    continue
                propname_glob = propname + "_glob"
                features.append(pl.repeat(prop, pl.len()).alias(propname_glob))
                self._temp_features.add(propname_glob)
        mole = mole.with_features(features)
        return mole


class LoaderInfo(NamedTuple):
    """Tuple that represents a subtomogram loader."""

    loader: BatchLoader
    name: str
    image_paths: dict[int, Path]
    invert: dict[int, bool]

    def rename(self, name: str):
        return LoaderInfo(self.loader, name, self.image_paths, self.invert)


class PathInfo:
    """Tuple that represents a child project path."""

    def __init__(
        self, image: Path, molecules: list[str | Path], project: Path | None = None
    ):
        self.image = image
        self.molecules = molecules
        self.project = project
        self._project_instance = None  # for caching

    def lazy_imread(self) -> ip.LazyImgArray:
        """Get the lazy image array."""
        img = ip.lazy.imread(self.image, chunks=get_config().dask_chunk)
        if self.need_invert:
            img = -img
        return img

    def iter_molecules(
        self, temp_features: TempFeatures, target_scale: float
    ) -> Iterator[Molecules]:
        """Iterate over all molecules."""
        prj = self.project_instance()
        if prj is None:
            for mole_path in self.molecules:
                mole = temp_features.read_molecules(mole_path)
                yield mole
        else:
            with prj.open_project() as dir:
                scale_factor = target_scale / prj.scale
                need_rescale = abs(scale_factor - 1) > 2e-6
                for mole_path in self.molecules:
                    mole = temp_features.read_molecules(dir / mole_path, prj)
                    if need_rescale:
                        mole._pos = mole._pos * scale_factor
                    yield mole

    def project_instance(self) -> CylindraProject | None:
        if self.project is None:
            return None
        if self._project_instance is None:
            self._project_instance = CylindraProject.from_file(self.project)
        return self._project_instance

    @property
    def need_invert(self) -> bool:
        """Whether the image needs inversion."""
        if self.project is not None:
            return self.project_instance().invert
        return False


def _find_source(
    mole_abs_path: Path,
    project: CylindraProject | None = None,
) -> CylSpline | None:
    if project is None:
        return None
    dir = mole_abs_path.parent
    mole_path = mole_abs_path.name
    for info in project.molecules_info:
        if info.name == mole_path:
            source = info.source
            if source is None:
                return None
            return project.load_spline(source, dir=dir)
    return None
