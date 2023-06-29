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
    ker = np.asarray(kernel, dtype=np.float32)
    return pl.Series(target, CylindricArray(nth, pf, value, nrise).convolve(ker).as1d())


def mean_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ker = np.asarray(kernel, dtype=np.bool_)
    return pl.Series(
        target, CylindricArray(nth, pf, value, nrise).mean_filter(ker).as1d()
    )


def max_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ker = np.asarray(kernel, dtype=np.bool_)
    return pl.Series(
        target, CylindricArray(nth, pf, value, nrise).max_filter(ker).as1d()
    )


def min_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ker = np.asarray(kernel, dtype=np.bool_)
    return pl.Series(
        target, CylindricArray(nth, pf, value, nrise).min_filter(ker).as1d()
    )


def median_filter(
    mole: Molecules, kernel: ArrayLike, target: str, nrise: int
) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    ker = np.asarray(kernel, dtype=np.bool_)
    return pl.Series(
        target, CylindricArray(nth, pf, value, nrise).median_filter(ker).as1d()
    )


def label(mole: Molecules, target: str, nrise: int) -> pl.Series:
    nth, pf, value = _get_input(mole, target)
    return pl.Series(target, CylindricArray(nth, pf, value, nrise).label().as1d()).cast(
        pl.UInt32
    )


def _get_input(mole: Molecules, target: str):
    df = mole.features
    value = df[target].to_numpy().astype(np.float32)
    nth = df[Mole.nth].to_numpy().astype(np.int32)
    pf = df[Mole.pf].to_numpy().astype(np.int32)
    return nth, pf, value
