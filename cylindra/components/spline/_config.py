from __future__ import annotations
import json

from typing import Any, Generic, Literal, Sequence, TypeVar
from dataclasses import dataclass
import warnings
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

    def astuple_rounded(self, ndigits: int = 2):
        return (round(self.min, ndigits), round(self.max, ndigits))

    @property
    def center(self) -> _T:
        return (self.min + self.max) / 2

    def aslist(self) -> list[_T]:
        return [self.min, self.max]

    def asslice(self) -> slice:
        return slice(self.min, self.max + 1)


@dataclass(frozen=True)
class WeightRamp:
    """Spline weight ramping parameters."""

    ramp_length: nm = 50.0
    tip_ratio: float = 0.5

    def __post_init__(self):
        if self.ramp_length < 0:
            raise ValueError("ramp_length must be non-negative")
        if not 0 <= self.tip_ratio <= 1:
            raise ValueError("tip_ratio must be between 0 and 1")

    def astuple(self) -> tuple[nm, float]:
        return (self.ramp_length, self.tip_ratio)

    def astuple_rounded(self):
        return (round(self.ramp_length, 1), round(self.tip_ratio, 3))


def _norm_range(x: Range[_T] | Sequence[_T] | tuple[_T, _T]) -> Range[_T]:
    if isinstance(x, Range):
        return x
    return Range(*x)


def _norm_weight_ramp(x: WeightRamp | Sequence[float] | tuple[nm, float]) -> WeightRamp:
    if isinstance(x, WeightRamp):
        return x
    return WeightRamp(*x)


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
            "spacing_range": self.spacing_range.astuple_rounded(),
            "skew_range": self.skew_range.astuple_rounded(),
            "rise_range": self.rise_range.astuple_rounded(),
            "rise_sign": self.rise_sign,
            "clockwise": self.clockwise,
            "thickness_inner": self.thickness_inner,
            "thickness_outer": self.thickness_outer,
            "fit_depth": self.fit_depth,
            "fit_width": self.fit_width,
            "weight_ramp": self.weight_ramp.astuple_rounded(),
        }

    @classmethod
    def construct(cls, **kwargs) -> SplineConfig:
        """Construct a SplineConfig with argument check."""
        return SplineConfig().updated(**kwargs)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        unknown: Literal["warn", "error", "ignore"] = "warn",
    ) -> SplineConfig:
        # for version compatibility
        _undef = {}
        cfg_input = {}
        for k, v in cfg.items():
            if k not in SplineConfig.__annotations__:
                _undef[k] = v
            else:
                cfg_input[k] = v

        if _undef:
            msg = f"Unknown keys, maybe due to version incompatibility: {_undef!r}"
            if unknown == "error":
                raise ValueError(msg)
            elif unknown == "warn":
                warnings.warn(msg, RuntimeWarning)
            else:
                pass
        return SplineConfig().updated(**cfg_input)

    @classmethod
    def from_file(
        cls,
        path: str,
        unknown: Literal["warn", "error", "ignore"] = "warn",
    ) -> SplineConfig:
        with open(path) as f:
            cfg: dict = json.load(f)
        return SplineConfig.from_dict(cfg, unknown=unknown)

    def to_file(self, path: str):
        with open(path, mode="w") as f:
            json.dump(self.asdict(), f)
        return None

    def updated(
        self,
        std: nm | None = None,
        npf_range: tuple[int, int] | None = None,
        spacing_range: tuple[nm, nm] | None = None,
        skew_range: tuple[float, float] | None = None,
        rise_range: tuple[float, float] | None = None,
        rise_sign: Literal[-1, 1] | None = None,
        clockwise: Literal["PlusToMinus", "MinusToPlus"] | None = None,
        thickness_inner: nm | None = None,
        thickness_outer: nm | None = None,
        fit_depth: nm | None = None,
        fit_width: nm | None = None,
        weight_ramp: tuple[float, float] | None = None,
    ) -> SplineConfig:
        kwargs = locals()
        kwargs.pop("self")
        for k, v in kwargs.items():
            if v is None:
                kwargs[k] = getattr(self, k)
        for rng in ["npf_range", "spacing_range", "skew_range", "rise_range"]:
            kwargs[rng] = _norm_range(kwargs[rng])
        kwargs["weight_ramp"] = _norm_weight_ramp(kwargs["weight_ramp"])
        if kwargs["rise_sign"] not in [-1, 1]:
            raise ValueError("rise_sign must be -1 or 1")
        if kwargs["clockwise"] not in ["PlusToMinus", "MinusToPlus"]:
            raise ValueError("clockwise must be PlusToMinus or MinusToPlus")
        for n in [
            "thickness_inner",
            "thickness_outer",
            "fit_depth",
            "fit_width",
            "std",
        ]:
            if kwargs[n] < 0:
                raise ValueError("thickness_inner must be non-negative")
        return SplineConfig(**kwargs)
