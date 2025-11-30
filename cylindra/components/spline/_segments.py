from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter

    from cylindra.components import Spline


@dataclass(frozen=True)
class SplineSegment:
    """A class representing a segment of a spline."""

    start: float
    end: float
    value: Any | None = None  # must be JSON serializable

    def __repr__(self) -> str:
        return f"SplineSegment(start={self.start:.4f}, end={self.end:.4f}, value={self.value!r})"

    def with_borders(self, start: float, end: float) -> SplineSegment:
        return type(self)(start, end, copy(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplineSegment:
        return cls(
            start=data["start"],
            end=data["end"],
            value=data.get("value", None),
        )

    def length(self, spl: Spline, nknots: int = 512) -> float:
        return spl.length(self.start, self.end, nknots=nknots)

    def sample(self, spl: Spline, num: int) -> NDArray[np.float32]:
        if num > 1:
            return spl.map(np.linspace(self.start, self.end, num))
        else:
            mid = (self.start + self.end) / 2
            return spl.map([mid])


class SplineSegments(Sequence[SplineSegment]):
    """A class representing a collection of spline segments."""

    def __init__(self, segments: Iterable[SplineSegment] = ()):
        self._segments = list(segments)

    def __repr__(self) -> str:
        return f"SplineSegments({self._segments})"

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool):
        if cycle:
            p.text("SplineSegments(...)")
        else:
            with p.group(14, "SplineSegments([", "])"):
                for i, seg in enumerate(self._segments):
                    if i > 0:
                        p.text(",")
                        p.breakable()
                    p.pretty(seg)

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> SplineSegments:
        segments = [SplineSegment.from_dict(d) for d in data]
        return cls(segments)

    def __getitem__(self, index: int) -> SplineSegment:
        return self._segments[index]

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self):
        return iter(self._segments)

    def enumerate(self) -> Iterable[tuple[int, SplineSegment]]:
        return enumerate(self._segments)

    def copy(self) -> SplineSegments:
        return SplineSegments(self._segments)

    def _append(self, start: float, end: float, value):
        json.dumps(value)  # Check if value is JSON serializable
        self._segments.append(SplineSegment(start, end, value))

    def _remove(self, indices: list[int]):
        appeared = set()
        for idx in indices:
            if idx in appeared:
                raise ValueError(f"Index {idx} appears multiple times.")
            if idx < 0 or idx >= len(self._segments):
                raise IndexError(f"Index {idx} out of range.")
            appeared.add(idx)
        for idx in sorted(appeared, reverse=True):
            self._segments.pop(idx)
