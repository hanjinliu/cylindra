import json
from typing import Any, Dict, List, Tuple, Type, Union, get_args, get_origin
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

class ProjectDescriptor:
    def __init__(self, desc: Dict[str, Any] = None, **kwargs):
        self._fields_ = {}
        if desc is None:
            desc = kwargs
        elif kwargs:
            raise TypeError("Cannot set both desc and **kwargs.")
        for k, v in desc.items():
            setattr(self, k, v)

    def __init_subclass__(cls) -> None:
        for annot, tp in cls.__annotations__.items():
            prop = cls._annotation_to_property(annot, tp)
            setattr(cls, annot, prop)
    
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
        return cls(js)
    
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


def _type_to_validator(tp: Type):
    origin = get_origin(tp)
    if origin is None and tp in (list, tuple, set, dict):
        origin = tp
    
    if origin is Union:
        _types = get_args(tp)
        _validators = [_type_to_validator(t) for t in _types]
        def _validator(v):
            return any(_val(v) for _val in _validators)
        
    elif origin in (list, tuple, set):
        _types = get_args(tp)
        if len(_types) == 0:
            def _validator(v):
                return isinstance(v, origin)
        else:
            _inner_validator = _type_to_validator(_types[0])
            def _validator(v):
                return (
                    isinstance(v, origin) and
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
    
    version: str
    image: PathLike
    scale: float
    multiscales: List[int]
    current_ft_size: float
    splines: List[dict]
    localprops: PathLike
    globalprops: PathLike
    molecules: List[PathLike]
    template_image: PathLike
    mask_parameters: Union[None, Tuple[float, float], List[float], PathLike]
    