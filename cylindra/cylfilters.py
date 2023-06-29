"""Filtering functions for cylindric structure, with `scipy.ndimage`-like API."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
from acryo import Molecules
from cylindra._cylindra_ext import CylindricArray
from cylindra.const import MoleculesHeader as Mole


def convolve(mole: Molecules, kernel: ArrayLike, target: str, nrise: int) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ar = CylindricArray(nth, pf, value, nrise)
    out = ar.convolve(np.asarray(kernel, dtype=np.float32))
    return pl.Series(target, _as_series(out, nth, pf))


def max_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ar = CylindricArray(nth, pf, value, nrise)
    out = ar.max_filter(np.asarray(kernel, dtype=np.bool_))
    return pl.Series(target, _as_series(out, nth, pf))


def min_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ar = CylindricArray(nth, pf, value, nrise)
    out = ar.min_filter(np.asarray(kernel, dtype=np.bool_))
    return pl.Series(target, _as_series(out, nth, pf))


def median_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ar = CylindricArray(nth, pf, value, nrise)
    out = ar.median_filter(np.asarray(kernel, dtype=np.bool_))
    return pl.Series(target, _as_series(out, nth, pf))


def _get_input(mole: Molecules, target: str):
    df = mole.features
    value = df[target].to_numpy().astype(np.float32)
    nth = df[Mole.nth].to_numpy().astype(np.int32)
    pf = df[Mole.pf].to_numpy().astype(np.int32)
    return nth, pf, value


def _as_series(out: CylindricArray, nth: NDArray[np.int32], pf: NDArray[np.int32]):
    new_series = np.zeros(nth.size, dtype=np.float32)
    out_array = out.asarray()
    for i, j, k in zip(range(nth.size), nth, pf):
        new_series[i] = out_array[j, k]
    return new_series
