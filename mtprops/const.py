from enum import Enum, auto
import numpy as np


class Ori(Enum):
    """Microtubule orientation values."""
    none = "none"
    PlusToMinus = "PlusToMinus"
    MinusToPlus = "MinusToPlus"
    
    def __str__(self):
        return self.value

class CacheKey(Enum):
    """Caching keys of straightening operations."""
    cart_straight = auto()
    cyl_straight = auto()
    
nm = float # type alias

class H:
    """Header names for result table of local properties."""
    splDistance = "splDistance"
    splPosition = "splPosition"
    riseAngle = "riseAngle"
    yPitch = "yPitch"
    skewAngle = "skewAngle"
    nPF = "nPF"  
    start = "start"

class K:
    """Keys of spline attributes."""
    radius = "radius"
    orientation = "orientation"
    localprops = "localprops"
    globalprops = "globalprops"

class Mode:
    """Padding mode used in scipy.ndimage."""
    grid_wrap = "grid-wrap"
    reflect = "reflect"
    mirror = "mirror"
    constant = "constant"
    nearest = "nearest"


class Sep(Enum):
    """Separator character."""
    Comma = ","
    Tab = "\t"
    Space = " "
    
    def __str__(self):
        return self.value

class Unit(Enum):
    """Unit of length."""
    pixel = "pixel"
    nm = "nm"
    angstrom = "angstrom"
    
    def __str__(self):
        return self.value

class EulerAxes(Enum):
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
    
    def __str__(self):
        return self.value

class GVar:
    """
    Global variables
    """    
    nPFmin: int = 11
    nPFmax: int = 17
    splOrder: int = 3
    yPitchMin: nm = 3.9
    yPitchMax: nm = 4.3
    minSkew: float = -1.0
    maxSkew: float = 1.0
    splError: nm = 0.8
    inner: float = 0.8
    outer: float = 1.3
    daskChunk: int = (128, 256, 256)
    
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