from __future__ import annotations
from pathlib import Path

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

from acryo import TomogramCollection, Molecules
import numpy as np
import impy as ip
import polars as pl

from cylindra.const import GlobalVariables
from cylindra.project import CylindraProject

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
    
    @abstractmethod
    def check_value(self, val: Any) -> _V:
        """Assert input has the same value. Raise an error otherwise."""
    
class ScaleValidator(Validator[float]):
    def check_value(self, val: Any) -> float:
        val = float(val)
        if 1 - val / self.value > 0.01:
            raise ValueError(f"Existing scale is {self.value}, tried to set {val}.")
        return val
        
class ProjectCollection:
    """Collection of Cylindra projects."""
    def __init__(self, check_scale: bool = True):
        self._projects: list[CylindraProject] = []
        self._scale_validator = ScaleValidator(check_scale)
    
    @classmethod
    def glob(cls, path: str | Path, check_scale: bool = True) -> Self:
        """
        Glob a path and add all projects found.
        
        >>> ProjectCollection.glob("path/to/projects/*.json")
        """
        from glob import glob
        self = cls(check_scale)
        for path in glob(str(path)):
            self.add(path)
        return self
    
    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], check_scale: bool = True) -> Self:
        """Add all the projects of the given paths."""
        self = cls(check_scale)
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
                filespec = pl.Series("file-name", np.full(len(mole), fp.stem))
                mole.features = mole.features.with_columns(filespec)
                col.add_tomogram(tomo.value, molecules=mole, image_id=idx)
        return col

    def localprops(self, allow_none: bool = True) -> pl.DataFrame:
        """Collect all localprops into a single dataframe."""
        dataframes: list[pl.DataFrame] = []
        for idx, prj in enumerate(self._projects):
            path = prj.localprops
            if path is None and not allow_none:
                raise ValueError(f"Localprops not found in project at {prj.project_path}.")
            df = pl.read_csv(path)
            imagespec = pl.Series("image-id", np.full(len(df), idx))
            dataframes.append(df.with_columns(imagespec))
        return pl.concat(dataframes, how="vertical")

    def globalprops(self, allow_none: bool = True) -> pl.DataFrame:
        """Collect all globalprops into a single dataframe."""
        dataframes: list[pl.DataFrame] = []
        for idx, prj in enumerate(self._projects):
            path = prj.globalprops
            if path is None and not allow_none:
                raise ValueError(f"Globalprops not found in project at {prj.project_path}.")
            imagespec = pl.Series("image-id", np.array([idx]))
            df = pl.read_csv(path).with_columns(imagespec)
            dataframes.append(df)
        return pl.concat(dataframes, how="vertical")
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert project information to a dataframe."""
        dataframes: list[pl.DataFrame] = []
        for idx, prj in enumerate(self._projects):
            projectspec = idx
            df = pl.DataFrame(prj.dict())
            df = df.with_columns(
                [
                    pl.Series("project-id", np.array([projectspec])),
                    pl.Series("project-path", np.array([prj.project_path])),
                ]
            )
            dataframes.append(df)
        return pl.concat(dataframes, how="vertical")
