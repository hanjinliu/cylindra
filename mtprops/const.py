from typing import NewType

class H:
    splDistance = "splDistance"
    splPosition = "splPosition"
    skew = "skew"
    yPitch = "yPitch"
    nPF = "nPF"  

    
nm = NewType("nm", float)
pixel = NewType("pixel", int)


INNER = 0.7
OUTER = 1.6