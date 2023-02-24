import os
import json
from typing import Any, Union, TYPE_CHECKING
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel

def json_encoder(obj):    
    """An enhanced encoder."""
    
    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pl.DataFrame):
        return obj.to_dict(as_series=False)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        if obj.is_absolute():
            return str(obj)
        else:
            return os.path.join(".", str(obj))
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")


PathLike = Union[Path, str, bytes]

class BaseProject(BaseModel):
    """The basic project class."""

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    macro: PathLike
    project_path: Union[Path, None] = None

    def dict(self, **kwargs) -> dict[str, Any]:
        """Return a dict."""
        d = super().dict(**kwargs)
        d.pop("project_path")
        return d

    def to_json(self, path: str) -> None:
        """Save project as a json file."""
        with open(path, mode="w") as f:
            json.dump(self.dict(), f, indent=4, separators=(",", ": "), default=json_encoder)
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.project_path})"
    
    @classmethod
    def from_json(cls, path: str):
        """Construct a project from a json file."""
        path = str(path)
    
        with open(path, mode="r") as f:
            js: dict = json.load(f)
        self = cls(**js, project_path=Path(path))
        file_dir = Path(path).parent
        self.resolve_path(file_dir)
        return self
    
    def resolve_path(self, file_dir: PathLike) -> None:
        """Resolve paths."""
        self.macro = Path(self.macro).resolve(file_dir)
        return None

def resolve_path(path: Union[str, Path, None], root: Path) -> Union[Path, None]:
    """Resolve a relative path to an absolute path."""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    path_joined = root / path
    if path_joined.exists():
        return path_joined
    raise ValueError(f"Path {path} could not be resolved under root path {root}.")
