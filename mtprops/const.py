from enum import Enum, auto
import numpy as np

class Ori(Enum):
    none = "none"
    PlusToMinus = "PlusToMinus"
    MinusToPlus = "MinusToPlus"

class CacheKey(Enum):
    cart_straight = auto()
    cyl_straight = auto()
    
nm = float # type alias

class H:
    """
    Header names for result table of local properties.
    """
    splDistance = "splDistance"
    splPosition = "splPosition"
    riseAngle = "riseAngle"
    yPitch = "yPitch"
    skewAngle = "skewAngle"
    nPF = "nPF"  
    start = "start"

class GVar:
    """
    Global variables
    """    
    nPFmin: int = 11
    nPFmax: int = 17
    splOrder: int = 3
    yPitchMin: nm = 3.9
    yPitchMax: nm = 4.5
    minSkew: float = -1.0
    maxSkew: float = 1.0
    splError: nm = 0.8
    inner: float = 0.8
    outer: float = 1.5
    daskChunk: int = (64, 512, 512)
    
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