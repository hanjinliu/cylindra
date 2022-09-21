from __future__ import annotations
from enum import Enum
import numpy as np
from types import SimpleNamespace
from typing import Any

nm = float  # type alias for nanometer
pf = float  # type alias for protofilament numbering

# Constant strings for layer names and metadata keys
WORKING_LAYER_NAME = "Working Layer"
SELECTION_LAYER_NAME = "Splines"
MOLECULES = "molecules"
ALN_SUFFIX = "ALN"


class strEnum(Enum):
    """Enum with string values"""

    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class Ori(strEnum):
    """Microtubule orientation values."""
    
    none = "none"
    PlusToMinus = "PlusToMinus"
    MinusToPlus = "MinusToPlus"
    
    @classmethod
    def invert(cls, ori: Ori, allow_none: bool = True) -> Ori:
        """
        Invert orientation.
        
        Parameters
        ----------
        ori : Ori
            Ori object to be inverted.
        allow_none : bool, default is True
            If true, convert ``Ori.none`` into ``Ori.none``. Raise an error
            otherwise.
        
        Returns
        -------
        Ori
            Inverted ``Ori`` object.
        """
        if ori == Ori.PlusToMinus:
            out = Ori.MinusToPlus
        elif ori == Ori.MinusToPlus:
            out = Ori.PlusToMinus
        else:
            if allow_none:
                out = Ori.none
            else:
                raise ValueError(f"{ori} cannot be inverted.")
        return out

class H(SimpleNamespace):
    """Header names for result table of local properties."""
    
    splDistance = "splDistance"
    splPosition = "splPosition"
    riseAngle = "riseAngle"
    yPitch = "yPitch"
    skewAngle = "skewAngle"
    nPF = "nPF"  
    start = "start"


class K(SimpleNamespace):
    """Keys of spline attributes."""
    
    radius = "radius"
    orientation = "orientation"
    localprops = "localprops"
    globalprops = "globalprops"


class Mode(SimpleNamespace):
    """Padding mode used in scipy.ndimage."""
    
    grid_wrap = "grid-wrap"
    reflect = "reflect"
    mirror = "mirror"
    constant = "constant"
    nearest = "nearest"

class Mole(SimpleNamespace):
    """Feature header names for Molecules."""
    
    pf = "molecules-pf"
    isotype = "molecules-isotype"
    zncc = "molecules-zncc"
    pcc = "molecules-pcc"
    interval = "molecules-interval"

class Align(SimpleNamespace):
    zShift = "shift-z"
    yShift = "shift-y"
    xShift = "shift-x"
    zRotvec = "rotvec-z"
    yRotvec = "rotvec-y"
    xRotvec = "rotvec-x"

    
class EulerAxes(strEnum):
    """Sequence of Euler angles."""
    
    xyz = "xyz"
    yzx = "yzx"
    zxy = "zxy"
    xzy = "xzy"
    yxz = "yxz"
    zyx = "zyx"
    xyx = "xyx"
    xzx = "xzx"
    yxy = "yxy"
    yzy = "yzy"
    zxz = "zxz"
    zyz = "zyz"
    XYZ = "XYZ"
    YZX = "YZX"
    ZXY = "ZXY"
    XZY = "XZY"
    YXZ = "YXZ"
    ZYX = "ZYX"
    XYX = "XYX"
    XZX = "XZX"
    YXY = "YXY"
    YZY = "YZY"
    ZXZ = "ZXZ"
    ZYZ = "ZYZ"


class GVar:
    """Global variables"""
        
    nPFmin: int = 11
    nPFmax: int = 17
    splOrder: int = 3
    yPitchMin: nm = 3.9
    yPitchMax: nm = 4.3
    minSkew: float = -1.0
    maxSkew: float = 1.0
    minCurvatureRadius: nm = 400.0
    inner: float = 0.8
    outer: float = 1.3
    fitLength: nm = 48.0
    fitWidth: nm = 44.0
    daskChunk: int = (256, 256, 256)
    GPU: bool = True
    
    @classmethod
    def get_value(cls) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in cls.__annotations__.keys():
            out[k] = getattr(cls, k)
        return out
    
    @classmethod
    def set_value(cls, **kwargs):
        if kwargs.get("yPitchMin", -np.inf) >= kwargs.get("yPitchMax", np.inf):
            raise ValueError("'yPitchMin' must be smaller than 'yPitchMax'.")
        if kwargs.get("minSkew", -np.inf) >= kwargs.get("maxSkew", np.inf):
            raise ValueError("'minSkew' must be smaller than 'maxSkew'.")
        for k, v in kwargs.items():
            if not hasattr(cls, k):
                pass
            setattr(cls, k, v)
