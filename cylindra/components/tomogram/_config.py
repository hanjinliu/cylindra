from __future__ import annotations

from dataclasses import dataclass, field
from cylindra.components.spline import SplineConfig


@dataclass
class TomogramConfig:
    spline_config: SplineConfig = field(default_factory=SplineConfig)
    spline_order: int = 3
    spline_extrapolate: str = "linear"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__annotations__:
                raise ValueError(f"Unknown config parameter: {k}")
            setattr(self, k, v)
