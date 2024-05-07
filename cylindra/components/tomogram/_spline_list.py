from __future__ import annotations

from typing import Iterable, Iterator, MutableSequence, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from cylindra.components.spline import CylSpline
from cylindra.const import PropertyNames as H


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
        """Copy the spline list."""
        return SplineList(self._list)

    def count(self) -> int:
        """Number of splines in the list."""
        return len(self)

    def filter(self, predicate: pl.Expr) -> SplineList:
        """Filter the list by its global properties."""

        df = pl.concat([spl.props.glob for spl in self])
        indices = np.where(df.select(predicate).to_series().to_numpy())[0]
        return SplineList([self._list[i] for i in indices])

    def sort(self, by: pl.Expr | str) -> SplineList:
        """Sort the list by its global properties."""

        def fn(spl: CylSpline):
            return spl.props.glob.select(by).to_series()[0]

        return SplineList(sorted(self._list, key=fn))

    def iter(self) -> Iterator[CylSpline]:
        """Iterate over splines."""
        return iter(self)

    def enumerate(self) -> Iterator[tuple[int, CylSpline]]:
        """Iterate over spline ID and splines."""
        return enumerate(self)

    def iter_anchor_coords(self) -> Iterable[NDArray[np.float32]]:
        """Iterate over anchor coordinates of all splines."""
        for i in range(len(self)):
            coords = self[i].map()
            yield from coords

    def collect_localprops(
        self, i: int | Iterable[int] = None, allow_none: bool = True
    ) -> pl.DataFrame | None:
        """
        Collect all the local properties into a single polars.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        pl.DataFrame
            Concatenated data frame.
        """
        if i is None:
            i = range(len(self))
        elif isinstance(i, int):
            i = [i]
        props = list[pl.DataFrame]()
        for i_ in i:
            spl = self[i_]
            prop = spl.props.loc
            if len(prop) == 0:
                if not allow_none:
                    raise ValueError(f"Local properties of spline {i_} is missing.")
                continue
            props.append(
                prop.with_columns(
                    pl.repeat(i_, pl.len()).cast(pl.UInt16).alias(H.spline_id),
                    pl.int_range(0, pl.len()).cast(pl.UInt16).alias(H.pos_id),
                    pl.Series(H.spl_pos, spl.anchors, dtype=pl.Float32),
                    pl.Series(H.spl_dist, spl.distances(), dtype=pl.Float32),
                )
            )

        if len(props) == 0:
            return None
        how = "diagonal" if allow_none else "vertical"
        return pl.concat(props, how=how)

    def collect_globalprops(
        self, i: int | Iterable[int] = None, allow_none: bool = True
    ) -> pl.DataFrame | None:
        """
        Collect all the global properties into a single polars.DataFrame.

        Parameters
        ----------
        i : int or iterable of int, optional
            Spline ID that you want to collect.

        Returns
        -------
        pl.DataFrame
            Concatenated data frame.
        """
        if i is None:
            i = range(len(self))
        elif isinstance(i, int):
            i = [i]
        props = list[pl.DataFrame]()
        for i_ in i:
            prop = self[i_].props.glob
            if len(prop) == 0:
                if not allow_none:
                    raise ValueError(f"Global properties of spline {i_} is missing.")
                continue
            props.append(prop.with_columns(pl.Series(H.spline_id, [i_])))

        if len(props) == 0:
            return None
        how = "diagonal" if allow_none else "vertical"
        return pl.concat(props, how=how)
