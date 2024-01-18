from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, Generic, Literal, Sequence, TypeVar

import numpy as np

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

    def astuple_rounded(self, ndigits: int = 4):
        return (round(self.min, ndigits), round(self.max, ndigits))

    @property
    def center(self) -> _T:
        return (self.min + self.max) / 2

    def aslist(self) -> list[_T]:
        return [self.min, self.max]

    def asarray(self):
        return np.array([self.min, self.max])

    def asslice(self) -> slice:
        return slice(self.min, self.max + 1)


def _norm_range(x: Range[_T] | Sequence[_T] | tuple[_T, _T]) -> Range[_T]:
    if isinstance(x, Range):
        return x
    return Range(*x)


@dataclass(frozen=True)
class SplineConfig:
    """Class for spline configuration."""

    npf_range: Range[int] = Range(11, 17)
    spacing_range: Range[nm] = Range(3.9, 4.3)
    twist_range: Range[float] = Range(-1.0, 1.0)
    rise_range: Range[float] = Range(0.0, 45.0)
    rise_sign: Literal[-1, 1] = -1
    clockwise: Literal["PlusToMinus", "MinusToPlus"] = "MinusToPlus"
    thickness_inner: nm = 2.0
    thickness_outer: nm = 3.0
    fit_depth: nm = 48.0
    fit_width: nm = 44.0

    def _repr_pretty_(self, p, cycle: bool):
        if cycle:
            p.text(repr(self))
        parts = list[str]()
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            parts.append(f"{k}={v!r}")
        cont = ",\n\t".join(parts)
        p.text(f"SplineConfig(\n\t{cont}\n)")

    def copy(self) -> SplineConfig:
        return SplineConfig(
            npf_range=self.npf_range.copy(),
            spacing_range=self.spacing_range.copy(),
            twist_range=self.twist_range.copy(),
            rise_range=self.rise_range.copy(),
            rise_sign=self.rise_sign,
            clockwise=self.clockwise,
            thickness_inner=self.thickness_inner,
            thickness_outer=self.thickness_outer,
            fit_depth=self.fit_depth,
            fit_width=self.fit_width,
        )

    def asdict(self) -> dict[str, Any]:
        return {
            "npf_range": self.npf_range.astuple(),
            "spacing_range": self.spacing_range.astuple_rounded(),
            "twist_range": self.twist_range.astuple_rounded(),
            "rise_range": self.rise_range.astuple_rounded(),
            "rise_sign": self.rise_sign,
            "clockwise": self.clockwise,
            "thickness_inner": self.thickness_inner,
            "thickness_outer": self.thickness_outer,
            "fit_depth": self.fit_depth,
            "fit_width": self.fit_width,
        }

    def json_dumps(self) -> str:
        strings = list[str]()
        for k, v in self.asdict().items():
            if isinstance(v, tuple):
                strings.append(f'"{k}": {list(v)!r}'.replace("'", '"'))
            else:
                strings.append(f'"{k}": {v!r}'.replace("'", '"'))
        return "{\n    " + ",\n    ".join(strings) + "\n}"

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
            match unknown:
                case "error":
                    raise ValueError(msg)
                case "warn":
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                case "ignore":
                    pass
                case other:  # pragma: no cover
                    raise ValueError(f"Got invalid case {other!r}")
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
        npf_range: tuple[int, int] | None = None,
        spacing_range: tuple[nm, nm] | None = None,
        twist_range: tuple[float, float] | None = None,
        rise_range: tuple[float, float] | None = None,
        rise_sign: Literal[-1, 1] | None = None,
        clockwise: Literal["PlusToMinus", "MinusToPlus"] | None = None,
        thickness_inner: nm | None = None,
        thickness_outer: nm | None = None,
        fit_depth: nm | None = None,
        fit_width: nm | None = None,
    ) -> SplineConfig:
        kwargs = locals()
        kwargs.pop("self")
        for k, v in kwargs.items():
            if v is None:
                kwargs[k] = getattr(self, k)
        for rng in ["npf_range", "spacing_range", "twist_range", "rise_range"]:
            kwargs[rng] = _norm_range(kwargs[rng])
        if kwargs["rise_sign"] not in [-1, 1]:
            raise ValueError("rise_sign must be -1 or 1")
        if kwargs["clockwise"] not in ["PlusToMinus", "MinusToPlus"]:
            raise ValueError("clockwise must be PlusToMinus or MinusToPlus")
        for n in ["thickness_inner", "thickness_outer", "fit_depth", "fit_width"]:
            if kwargs[n] < 0:
                raise ValueError(f"{n} must be non-negative")
        return SplineConfig(**kwargs)
