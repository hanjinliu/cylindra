import os
import json
from typing import List, Dict, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel


def json_encoder(obj):    
    """An enhanced encoder."""
    
    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
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


PathLike = Union[Path, str]

class CylindraProject(BaseModel):
    """A project of cylindra."""
    
    datetime: str
    version: str
    dependency_versions: Dict[str, str]
    image: PathLike
    scale: float
    multiscales: List[int]
    current_ft_size: float
    splines: List[PathLike]
    localprops: Union[PathLike, None]
    globalprops: Union[PathLike, None]
    molecules: List[PathLike]
    global_variables: PathLike
    template_image: Union[PathLike, None]
    mask_parameters: Union[None, Tuple[float, float], PathLike]
    tilt_range: Union[Tuple[float, float], None]
    macro: PathLike

    @classmethod
    def from_json(cls, path: str):
        """Construct a project from a json file."""
        path = str(path)
    
        with open(path, mode="r") as f:
            js: dict = json.load(f)
        return cls(**js)
    
    def to_json(self, path: str) -> None:
        """Save project as a json file."""
        with open(path, mode="w") as f:
            json.dump(self.dict(), f, indent=4, separators=(",", ": "), default=json_encoder)
        return None
