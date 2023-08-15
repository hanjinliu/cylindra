from __future__ import annotations

from typing import Callable, MutableSequence, overload, Iterable, Iterator
from cylindra.components.spline import CylSpline


class SplineList(MutableSequence[CylSpline]):
    """Container of splines."""

    def __init__(self, iterable: Iterable[CylSpline] = ()) -> None:
        self._list = list(iterable)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._list!r})"

    @overload
    def __getitem__(self, i: int) -> CylSpline:
        ...

    @overload
    def __getitem__(self, i: slice) -> list[CylSpline]:
        ...

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list(self._list[i])
        return self._list[i]

    def __setitem__(self, i: int, spl: CylSpline) -> None:
        if not isinstance(spl, CylSpline):
            raise TypeError(f"Cannot add {type(spl)} to SplineList")
        self._list[i] = spl

    def __delitem__(self, i: int) -> None:
        del self._list[i]

    def __len__(self) -> int:
        return len(self._list)

    def insert(self, i: int, spl: CylSpline) -> None:
        if not isinstance(spl, CylSpline):
            raise TypeError(f"Cannot add {type(spl)} to SplineList")
        self._list.insert(i, spl)

    def __iter__(self) -> Iterator[CylSpline]:
        return iter(self._list)

    def index(self, value: CylSpline, start: int = 0, stop: int = 9999999) -> int:
        for i, spl in enumerate(self._list):
            if i < start:
                continue
            if spl is value:
                return i
            if i >= stop:
                break
        raise ValueError(f"{value} is not in list")

    def remove(self, value: CylSpline) -> None:
        i = self.index(value)
        del self[i]

    def copy(self) -> SplineList:
        return SplineList(self._list)

    def filter(self, predicate: Callable[[CylSpline], bool]) -> SplineList:
        return SplineList(filter(predicate, self._list))
