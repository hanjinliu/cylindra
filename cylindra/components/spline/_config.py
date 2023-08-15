from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar
from dataclasses import dataclass
from cylindra.const import nm

_T = TypeVar("_T")


@dataclass(frozen=True)
class Range(Generic[_T]):
    """Class for range."""

    min: _T
    max: _T

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min must be less than or equal to max")

    def __contains__(self, value: _T) -> bool:
        return self.min <= value <= self.max

    def copy(self) -> Range[_T]:
        return Range(self.min, self.max)

    def astuple(self) -> tuple[_T, _T]:
        return (self.min, self.max)

    def aslist(self) -> list[_T]:
        return [self.min, self.max]

    def asslice(self) -> slice:
        return slice(self.min, self.max + 1)


@dataclass(frozen=True)
class WeightRamp:
    """Spline weight ramping parameters."""

    ramp_length: nm = 50.0
    tip_ratio: float = 0.5

    def astuple(self) -> tuple[_T, _T]:
        return (self.min, self.max)


@dataclass(frozen=True)
class SplineConfig:
    """Class for spline configuration."""

    std: nm = 0.1
    npf_range: Range[int] = Range(11, 17)
    spacing_range: Range[nm] = Range(3.9, 4.3)
    skew_range: Range[float] = Range(-1.0, 1.0)
    rise_range: Range[float] = Range(0.0, 45.0)
    rise_sign: Literal[-1, 1] = -1
    clockwise: Literal["PlusToMinus", "MinusToPlus"] = "MinusToPlus"
    thickness_inner: nm = 2.0
    thickness_outer: nm = 3.0
    fit_depth: nm = 48.0
    fit_width: nm = 44.0
    weight_ramp: WeightRamp = WeightRamp()

    def copy(self) -> SplineConfig:
        return SplineConfig(
            std=self.std,
            npf_range=self.npf_range.copy(),
            spacing_range=self.spacing_range.copy(),
            skew_range=self.skew_range.copy(),
            rise_range=self.rise_range.copy(),
            rise_sign=self.rise_sign,
            clockwise=self.clockwise,
            thickness_inner=self.thickness_inner,
            thickness_outer=self.thickness_outer,
            fit_depth=self.fit_depth,
            fit_width=self.fit_width,
            weight_ramp=self.weight_ramp,
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "std": self.std,
            "npf_range": self.npf_range.astuple(),
            "spacing_range": self.spacing_range.astuple(),
            "skew_range": self.skew_range.astuple(),
            "rise_range": self.rise_range.astuple(),
            "rise_sign": self.rise_sign,
            "clockwise": self.clockwise,
            "thickness_inner": self.thickness_inner,
            "thickness_outer": self.thickness_outer,
            "fit_depth": self.fit_depth,
            "fit_width": self.fit_width,
            "weight_ramp": self.weight_ramp.astuple(),
        }
