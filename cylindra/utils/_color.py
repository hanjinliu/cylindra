from __future__ import annotations

from typing import Iterable


def str_color(color: Iterable[float] | str) -> str:
    """Convert any color input as a string."""
    if isinstance(color, str):
        return color
    _col = "#" + "".join(hex(int(c * 255))[2:].upper().zfill(2) for c in color)
    if _col.endswith("FF"):
        _col = _col[:-2]
    return _col
