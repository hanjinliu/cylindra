import json
from typing import Any, Dict, List, Tuple, Type, Union, get_args, get_origin
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

class ProjectDescriptor:
    def __init__(self, **kwargs):
        self._fields_ = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __init_subclass__(cls) -> None:
        for annot, tp in cls.__annotations__.items():
            prop = cls._annotation_to_property(annot, tp)
            setattr(cls, annot, prop)
        cls.__init__.__annotations__ = cls.__annotations__
    
    @classmethod
    def _annotation_to_property(cls, annot: str, tp: Type):
        _validator = _type_to_validator(tp)
        @property
        def prop(self: cls) -> tp:
            return self._fields_.get(annot, None)
        
        @prop.setter
        def prop(self: cls, v: tp) -> None:
            if not _validator(v):
                raise TypeError(
                    f"{annot} must be {tp} but got unexpected type {type(v)}."
                )
            self._fields_[annot] = v
            
        return prop
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}\n" + 
            "\n".join(f"{k} = {_short_repr(v)}" for k, v in self._fields_.items())
        )

    @classmethod
    def from_json(cls, path: str):
        path = str(path)
    
        with open(path, mode="r") as f:
            js: dict = json.load(f)
        return cls(**js)
    
    def to_json(self, path: str) -> None:
        with open(path, mode="w") as f:
            json.dump(self._fields_, f, indent=4, separators=(",", ": "), default=json_encoder)
        return None

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
        return str(obj)
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")


_list_like = (list, tuple, set)

def _type_to_validator(tp: Type):
    origin = get_origin(tp)
    if origin is None and tp in (list, tuple, set, dict):
        origin = tp
    
    if origin is Union:
        _types = get_args(tp)
        _validators = [_type_to_validator(t) for t in _types]
        def _validator(v):
            return any(_val(v) for _val in _validators)
        
    elif origin in _list_like:
        _types = get_args(tp)
        if len(_types) == 0:
            def _validator(v):
                return isinstance(v, _list_like)
        else:
            _inner_validator = _type_to_validator(_types[0])
            def _validator(v):
                return (
                    isinstance(v, _list_like) and
                    all(_inner_validator(v0) for v0 in v)
                )
    elif origin is dict:
        _types = get_args(tp)
        if len(_types) == 0:
            def _validator(v):
                return isinstance(v, origin)
        elif len(_types) == 1:
            _inner_validator = _type_to_validator(_types[0])
            def _validator(v):
                return (
                    isinstance(v, origin) and
                    all(_inner_validator(v0) for v0 in v.keys())
                )
        else:
            _key_validator = _type_to_validator(_types[0])
            _value_validator = _type_to_validator(_types[1])
            def _validator(v):
                return (
                    isinstance(v, origin) and
                    all(_key_validator(k0) and _value_validator(v0)
                        for k0, v0 in v.items()
                    )
                )
    elif tp is None:
        def _validator(v):
            return v is None
    else:
        def _validator(v):
            return isinstance(v, tp)
        
    return _validator

def _short_repr(v):
    r = repr(v)
    if len(r) < 90:
        return r
    return r[:87] + "..."


PathLike = Union[Path, str]

class MTPropsProject(ProjectDescriptor):
    """A project of MTProps."""
    
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
    chunksize: int
    macro: PathLike


class SubtomogramAveragingProject(ProjectDescriptor):
    """A project of subtomogram averaging using multiple tomograms."""
    
    datetime: str
    version: str
    dependency_versions: Dict[str, str]
    datasets: Dict[PathLike, List[PathLike]]  # {subproject-path: [molecules-path-0, molecules-path-1, ...]}
    shape: Tuple[float, float ,float]
    chunksize: int
    