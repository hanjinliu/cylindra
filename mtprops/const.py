from typing import NewType

class H:
    """
    Header names for result table of local properties.
    """
    splDistance = "splDistance"
    splPosition = "splPosition"
    skew = "skew"
    yPitch = "yPitch"
    nPF = "nPF"  

    
nm = float
pixel = NewType("pixel", int)


INNER = 0.7
OUTER = 1.6