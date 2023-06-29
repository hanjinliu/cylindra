"""Filtering functions for cylindric structure, with `scipy.ndimage`-like API."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
from cylindra._cylindra_ext import CylindricArray as _CylindricArray
from cylindra.const import MoleculesHeader as Mole
from typing_extensions import Self


class CylindricArray:
    def __init__(self, rust_obj: _CylindricArray):
        self._rust_obj = rust_obj

    def __repr__(self) -> str:
        return f"CylindricArray({self._rust_obj.asarray()}, nrise={self.nrise})"

    @property
    def nrise(self) -> int:
        """The number of rise of the cylindric structure."""
        return self._rust_obj.nrise()

    def asarray(self) -> NDArray[np.float32]:
        """As a 2D numpy array."""
        return self._rust_obj.asarray()

    def as1d(self) -> NDArray[np.float32]:
        return self._rust_obj.as1d()

    def as_series(self, name: str = "") -> pl.Series:
        return pl.Series(name, self.as1d())

    def with_values(self, values: ArrayLike) -> Self:
        return CylindricArray(
            self._rust_obj.with_values(np.asarray(values, dtype=np.float32))
        )

    def __array__(self, dtype=None) -> NDArray[np.float32]:
        out = self.asarray()
        if dtype is not None:
            out = out.astype(dtype)
        return out

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

    def __neg__(self) -> Self:
        return self.with_values(-self.as1d())

    def label(self) -> Self:
        return CylindricArray(self._rust_obj.label())


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
    if method == "mean":
        _filter_func = mean_filter
    elif method == "max":
        _filter_func = max_filter
    elif method == "min":
        _filter_func = min_filter
    elif method == "median":
        _filter_func = median_filter
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return _filter_func(df, kernel, target, nrise)


def binarize(df: pl.DataFrame, threshold: float, target: str) -> pl.Series:
    return (
        CylindricArray.from_dataframe(df, target, 0)
        .binarize(threshold)
        .as_series(target)
    )


def label(df: pl.DataFrame, target: str, nrise: int) -> pl.Series:
    return CylindricArray.from_dataframe(df, target, nrise).label().as_series(target)
