from __future__ import annotations

from dataclasses import dataclass
from cylindra.components.spline import SplineConfig


@dataclass
class TomogramConfig:
    spline_config: SplineConfig = SplineConfig()
    spline_order: int = 3
    extrapolate: str = "linear"
