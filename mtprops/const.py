from enum import Enum, auto

class Ori(Enum):
    none = "none"
    PlusToMinus = "PlusToMinus"
    MinusToPlus = "MinusToPlus"

class CacheKey(Enum):
    subtomograms = auto()
    cart_straight = auto()
    cyl_straight = auto()
    
nm = float

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
    yPitchAvg: nm = 4.16
    splError: nm = 0.8
    rMax: nm = 17
    inner: float = 0.7
    outer: float = 1.6
    
    @classmethod
    def set_value(cls, *args, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(cls, k):
                pass
            setattr(cls, k, v)