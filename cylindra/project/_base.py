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
