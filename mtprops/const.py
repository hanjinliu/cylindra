from enum import Enum, auto

class Ori(Enum):
    none = "none"
    PlusToMinus = "PlusToMinus"
    MinusToPlus = "MinusToPlus"

class CacheKey(Enum):
    subtomograms = auto()
    straight = auto()
    
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
    
nm = float

INNER = 0.7
OUTER = 1.6