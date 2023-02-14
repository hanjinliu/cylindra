from __future__ import annotations
from contextlib import suppress
from pathlib import Path

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, Iterator, MutableSequence, SupportsIndex, TypeVar, overload

from acryo import TomogramCollection, Molecules
import numpy as np
import impy as ip
import polars as pl

from cylindra.const import GlobalVariables, MoleculesHeader as Mole
from .single import CylindraProject

if TYPE_CHECKING:
    from typing_extensions import Self

_V = TypeVar("_V")
_Null = object()

class Validator(ABC, Generic[_V]):
    def __init__(self, check: bool = True):
        self._value = _Null
        self._check = check
    
    @property
    def value(self) -> _V:
        if self._value is _Null:
            raise AttributeError("Value cannot be determined yet.")
        return self._value
    
    @value.setter
    def value(self, val: _V):
        if self._value is _Null:
            self._value = val
        else:
            if self._check:
                val = self.check_value(val)
            self._value = val
    
    def initialize(self):
        self._value = _Null
    
    @abstractmethod
    def check_value(self, val: Any) -> _V:
        """Assert input has the same value. Raise an error otherwise."""
    
class ScaleValidator(Validator[float]):
    def check_value(self, val: Any) -> float:
        val = float(val)
        if 1 - val / self.value > 0.01:
            raise ValueError(f"Existing scale is {self.value}, tried to set {val}.")
        return val
        
class ProjectSequence(MutableSequence[CylindraProject]):
    """Collection of Cylindra projects."""

    def __init__(self, *, check_scale: bool = True):
        self._projects: list[CylindraProject] = []
        self._scale_validator = ScaleValidator(check_scale)
    
    def __repr__(self) -> str:
        if len(self) > 1:
            return f"{type(self).__name__} with {len(self)} projects such as {self[0]!r}"
        return f"{type(self).__name__} (empty)"
    
    @classmethod
    def glob(cls, path: str | Path, check_scale: bool = True) -> Self:
        """
        Glob a path and add all projects found.
        
        >>> ProjectCollection.glob("path/to/projects/*.json")
        """
        from glob import glob
        self = cls(check_scale=check_scale)
        for path in glob(str(path)):
            self.add(path)
        return self
    
    @overload
    def __getitem__(self, key: int) -> CylindraProject: ...
    @overload
    def __getitem__(self, key: slice) -> ProjectSequence: ...
    
    def __getitem__(self, key: int):
        out = self._projects[key]
        if isinstance(key, slice):
            out = ProjectSequence(check_scale=True)
            out._projects = self._projects[key]
        return out
    
    def __setitem__(self, key: int, value: CylindraProject) -> None:
        if not isinstance(value, CylindraProject):
            raise TypeError(f"Expected CylindraProject, got {type(value)}.")
        if not isinstance(key, SupportsIndex):
            raise TypeError(f"Expected int, got {type(key)}.")
        self._projects[key] = value
    
    def __delitem__(self, key: int) -> None:
        del self._projects[key]
        if len(self) == 0:
            self._scale_validator.initialize()
    
    def __len__(self) -> int:
        return len(self._projects)

    def __iter__(self) -> Iterator[CylindraProject]:
        return iter(self._projects)
    
    def insert(self, index: int, value: CylindraProject) -> None:
        if not isinstance(value, CylindraProject):
            raise TypeError(f"Expected CylindraProject, got {type(value)}.")
        self._projects.insert(index, value)
    
    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], *, check_scale: bool = True, skip_exc: bool = False) -> Self:
        """Add all the projects of the given paths."""
        self = cls(check_scale=check_scale)
        if skip_exc:
            for path in paths:
                with suppress(Exception):
                    self.add(path)    
        else:
            for path in paths:
                self.add(path)
        return self
    
    def add(self, path: str | Path) -> Self:
        """Add a project from path."""
        prj = CylindraProject.from_json(path)
        self._scale_validator.value = prj.scale
        self._projects.append(prj)
        return self
    
    def sta_loader(self) -> TomogramCollection:
        """Construct a STA loader from all the projects."""
        col = TomogramCollection(scale=self._scale_validator.value)
        for idx, prj in enumerate(self._projects):
            tomo = ip.lazy_imread(prj.image, chunks=GlobalVariables.daskChunk)
            for fp in prj.molecules:
                fp = Path(fp)
                mole = Molecules.from_csv(fp)
                mole.features = mole.features.with_columns(
                    [pl.repeat(fp.stem, pl.count()).alias(Mole.id)]
                )
                col.add_tomogram(tomo.value, molecules=mole, image_id=idx)
        return col

    def localprops(self, allow_none: bool = True) -> pl.DataFrame:
        """Collect all localprops into a single dataframe."""
        dataframes: list[pl.DataFrame] = []
        for idx, prj in enumerate(self._projects):
            path = prj.localprops
            if path is None:
                if not allow_none:
                    raise ValueError(f"Localprops not found in project at {prj.project_path}.")
            else:
                df = pl.read_csv(path)
                dataframes.append(
                    df.with_columns(pl.repeat(idx, pl.count()).cast(pl.UInt16).alias(Mole.image))
                )
        return pl.concat(dataframes, how="diagonal")

    def globalprops(self, allow_none: bool = True) -> pl.DataFrame:
        """Collect all globalprops into a single dataframe."""
        dataframes: list[pl.DataFrame] = []
        for idx, prj in enumerate(self._projects):
            path = prj.globalprops
            if path is None and not allow_none:
                raise ValueError(f"Globalprops not found in project at {prj.project_path}.")
            imagespec = pl.Series(Mole.image, np.array([idx])).cast(pl.UInt16)
            df = pl.read_csv(path).with_columns(imagespec)
            dataframes.append(df)
        return pl.concat(dataframes, how="diagonal")
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert project information to a dataframe."""
        dataframes: list[pl.DataFrame] = []
        for prj in self._projects:
            df = pl.DataFrame(prj.dict())
            df = df.with_columns(
                [pl.Series("project-path", np.array([prj.project_path]))]
            )
            dataframes.append(df)
        return pl.concat(dataframes, how="vertical")

    def filter(self, predicate):
        # TODO: 
        new_loader = self.sta_loader().filter(predicate)
        mole = new_loader.molecules
        for idx, prj in enumerate(self):
            ...