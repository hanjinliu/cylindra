"""Filtering functions for cylindric structure, with `scipy.ndimage`-like API."""
from __future__ import annotations

import operator
from typing import Any, Callable

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from cylindra._cylindra_ext import CylindricArray as _CylindricArray
from cylindra.const import MoleculesHeader as Mole


class CylindricArray:
    def __init__(self, rust_obj: _CylindricArray):
        self._rust_obj = rust_obj

    def __repr__(self) -> str:
        return f"CylindricArray({self._rust_obj.asarray()}, nrise={self.nrise})"

    @property
    def nrise(self) -> int:
        """The number of rise of the cylindric structure."""
        return self._rust_obj.nrise()

    def asarray(self, dtype=None) -> NDArray[np.float32]:
        """As a 2D numpy array."""
        out = self._rust_obj.asarray()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def as1d(self, dtype=None) -> NDArray[np.float32]:
        out = self._rust_obj.as1d()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def as_series(self, name: str = "", dtype=None) -> pl.Series:
        if type(dtype) is type and dtype in pl.DataType:
            pl_dtype = dtype
            np_dtype = None
        else:
            pl_dtype = None
            np_dtype = dtype
        return pl.Series(name, self.as1d(np_dtype), dtype=pl_dtype)

    def with_values(self, values: ArrayLike) -> Self:
        return CylindricArray(
            self._rust_obj.with_values(np.asarray(values, dtype=np.float32))
        )

    __array__ = asarray

    @classmethod
    def from_sequences(
        cls, nth: ArrayLike, npf: ArrayLike, value: ArrayLike, nrise: int
    ) -> Self:
        nth = np.asarray(nth, dtype=np.int32)
        npf = np.asarray(npf, dtype=np.int32)
        value = np.asarray(value, dtype=np.float32)
        nrise = int(nrise)
        return cls(_CylindricArray(nth, npf, value, nrise))

    @classmethod
    def zeros_like(self, df: pl.DataFrame, nrise: int) -> Self:
        nth = df[Mole.nth].to_numpy()
        npf = df[Mole.pf].to_numpy()
        value = np.zeros(len(df), dtype=np.float32)
        return CylindricArray.from_sequences(nth, npf, value, nrise)

    @classmethod
    def from_dataframe(self, df: pl.DataFrame, target: str, nrise: int) -> Self:
        nth = df[Mole.nth].to_numpy()
        npf = df[Mole.pf].to_numpy()
        value = df[target].to_numpy()
        return CylindricArray.from_sequences(nth, npf, value, nrise)

    def convolve(self, kernel: ArrayLike) -> Self:
        ker = np.asarray(kernel, dtype=np.float32)
        return CylindricArray(self._rust_obj.convolve(ker))

    def mean_filter(self, kernel: ArrayLike) -> Self:
        ker = np.asarray(kernel, dtype=np.bool_)
        return CylindricArray(self._rust_obj.mean_filter(ker))

    def count_neighbors(self, kernel: ArrayLike) -> Self:
        ker = np.asarray(kernel, dtype=np.bool_)
        return CylindricArray(self._rust_obj.count_neighbors(ker))

    def max_filter(self, kernel: ArrayLike) -> Self:
        ker = np.asarray(kernel, dtype=np.bool_)
        return CylindricArray(self._rust_obj.max_filter(ker))

    def min_filter(self, kernel: ArrayLike) -> Self:
        ker = np.asarray(kernel, dtype=np.bool_)
        return CylindricArray(self._rust_obj.min_filter(ker))

    def median_filter(self, kernel: ArrayLike) -> Self:
        ker = np.asarray(kernel, dtype=np.bool_)
        return CylindricArray(self._rust_obj.median_filter(ker))

    def binarize(self, threshold: float) -> Self:
        value = self.as1d()
        new_value = value >= threshold
        return self.with_values(new_value)

    def label(self) -> Self:
        return CylindricArray(self._rust_obj.label())

    def __neg__(self) -> Self:
        return self.with_values(-self.as1d())

    @staticmethod
    def _make_operator(op) -> Callable[[CylindricArray, Any], CylindricArray]:
        def _method(self: CylindricArray, value) -> CylindricArray:
            if isinstance(value, CylindricArray):
                value = value.as1d()
            return self.with_values(op(self.as1d(), value))

        _method.__name__ = op.__name__
        _method.__qualname__ = f"CylindricArray.{op.__name__}"
        return _method

    __eq__ = _make_operator(operator.__eq__)
    __ne__ = _make_operator(operator.__ne__)
    __lt__ = _make_operator(operator.__lt__)
    __le__ = _make_operator(operator.__le__)
    __gt__ = _make_operator(operator.__gt__)
    __ge__ = _make_operator(operator.__ge__)
    __add__ = _make_operator(operator.__add__)
    __sub__ = _make_operator(operator.__sub__)
    __mul__ = _make_operator(operator.__mul__)
    __truediv__ = _make_operator(operator.__truediv__)
    __pow__ = _make_operator(operator.__pow__)
    __and__ = _make_operator(operator.__and__)
    __or__ = _make_operator(operator.__or__)
    __xor__ = _make_operator(operator.__xor__)


def convolve(df: pl.DataFrame, kernel: ArrayLike, target: str, nrise: int) -> pl.Series:
    return pl.Series(
        target, CylindricArray.from_dataframe(df, target, nrise).convolve(kernel).as1d()
    )


def mean_filter(
    df: pl.DataFrame, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, nrise)
        .mean_filter(kernel)
        .as_series(target)
    )


def max_filter(
    df: pl.DataFrame, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, nrise)
        .max_filter(kernel)
        .as_series(target)
    )


def min_filter(
    df: pl.DataFrame, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, nrise)
        .min_filter(kernel)
        .as_series(target)
    )


def median_filter(
    df: pl.DataFrame, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, nrise)
        .median_filter(kernel)
        .as_series(target)
    )


def run_filter(
    df: pl.DataFrame,
    kernel: ArrayLike,
    target: str,
    nrise: int,
    method: str,
) -> pl.Series:
    match method:
        case "mean":
            _filter_func = mean_filter
        case "max":
            _filter_func = max_filter
        case "min":
            _filter_func = min_filter
        case "median":
            _filter_func = median_filter
        case _:  # pragma: no cover
            raise ValueError(f"Unknown method: {method!r}")
    return _filter_func(df, kernel, target, nrise)


def count_neighbors(
    df: pl.DataFrame,
    kernel: ArrayLike,
    nrise: int,
) -> pl.Series:
    return (
        CylindricArray.zeros_like(df, nrise)
        .count_neighbors(kernel)
        .as_series(name="neighbor")
        .round()
        .cast(pl.UInt32)
    )


def binarize(df: pl.DataFrame, threshold: float, target: str) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, 0)
        .binarize(threshold)
        .as_series(name=target)
    )


def label(df: pl.DataFrame, target: str, nrise: int) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, nrise)
        .label()
        .as_series(target)
        .round()
        .cast(pl.UInt32)
    )
