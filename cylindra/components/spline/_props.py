from __future__ import annotations

from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl
from typing_extensions import Self

from cylindra.const import nm

_DataFrameLike = pl.DataFrame | Mapping[str, Any] | Sequence[pl.Series | pl.Expr]
_void = object()


class SplineProps:
    """Class for spline properties."""

    def __init__(self) -> None:
        self._loc = pl.DataFrame([])
        self._glob = pl.DataFrame([])
        self._window_size = dict[str, nm]()
        self._binsize_loc = dict[str, int]()
        self._binsize_glob = dict[str, int]()

    def __repr__(self) -> str:
        loc = self.loc
        ws = self.window_size
        mapping = dict[str, str]()
        for k in loc.columns:
            if k in ws:
                mapping[k] = f"{k}\n({ws[k]:.2f} nm)"
            else:
                mapping[k] = f"{k}\n(-- nm)"
        loc = loc.rename(mapping)
        return f"SplineProps(\nlocal=\n{loc!r}\nglobal=\n{self.glob!r}\n)"

    @property
    def loc(self) -> pl.DataFrame:
        """Return the local properties"""
        return self._loc

    @loc.setter
    def loc(self, df: pl.DataFrame):
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        self._loc = df

    @property
    def glob(self) -> pl.DataFrame:
        """Return the global properties"""
        return self._glob

    @glob.setter
    def glob(self, df: pl.DataFrame):
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        if df.shape[0] > 1:
            raise ValueError("Global properties must be a single row.")
        self._glob = df

    @property
    def window_size(self) -> MappingProxyType[str, nm]:
        """Return the window size dict of the local properties"""
        return MappingProxyType(self._window_size)

    @property
    def binsize_loc(self) -> MappingProxyType[str, int]:
        """Return the bin size dict of the local properties"""
        return MappingProxyType(self._binsize_loc)

    @property
    def binsize_glob(self) -> MappingProxyType[str, int]:
        """Return the bin size dict of the global properties"""
        return MappingProxyType(self._binsize_glob)

    def copy(self) -> Self:
        """Copy this object"""
        new = self.__class__()
        new._loc = self._loc.clone()
        new._glob = self._glob.clone()
        new._window_size = self._window_size.copy()
        new._binsize_loc = self._binsize_loc.copy()
        new._binsize_glob = self._binsize_glob.copy()
        return new

    def __getitem__(self, key) -> Self:
        new = SplineProps()
        new._loc = self._loc[key]
        new._glob = self._glob[key]
        new._window_size = {key: self._window_size[key]}
        new._binsize_loc = self._binsize_loc[key]
        new._binsize_glob = self._binsize_glob[key]
        return new

    def select(self, keys: str | Iterable[str]) -> Self:
        """Select local properties."""
        if isinstance(keys, str):
            keys = [keys]
        new = SplineProps()
        new._loc = self._loc.select(keys)
        new._glob = self._glob.select(keys)
        new._window_size = {k: self._window_size[k] for k in keys}
        new._binsize_loc = {k: self._binsize_loc[k] for k in keys}
        new._binsize_glob = {k: self._binsize_glob[k] for k in keys}
        return new

    def update_loc(
        self,
        props: _DataFrameLike,
        window_size: nm | Mapping[str, nm],
        bin_size: int | Mapping[str, int] | None = None,
    ) -> Self:
        """
        Set local properties of given window size.

        Parameters
        ----------
        props : DataFrame-like object
            Local properties.
        window_size : nm, optional
            Window size of local properties in nm.
        """
        if not isinstance(props, pl.DataFrame):
            df = pl.DataFrame(props)
        else:
            df = props

        self._loc = self._loc.with_columns(df)
        if isinstance(window_size, Mapping):
            self._window_size.update(
                {c: _pos_float(window_size[c]) for c in df.columns}
            )
        else:
            ws = _pos_float(window_size)
            self._window_size.update({c: ws for c in df.columns})
        if isinstance(bin_size, (int, np.integer)):
            for key in df.columns:
                self._binsize_loc[key] = bin_size
        elif isinstance(bin_size, Mapping):
            self._binsize_loc.update(bin_size)
        return self

    def update_glob(
        self,
        props: _DataFrameLike | None = None,
        bin_size: int | Mapping[str, int] | None = None,
        **kwargs,
    ) -> Self:
        """Update the global properties."""
        if kwargs:
            if props is not None:
                raise ValueError("Cannot specify both props and kwargs.")
            props = pl.DataFrame(kwargs)
        if not isinstance(props, pl.DataFrame):
            df = pl.DataFrame(props)
        else:
            df = props
        if df.shape[0] > 1:
            raise ValueError("Global properties must be a single row.")
        self._glob = self._glob.with_columns(df)
        if isinstance(bin_size, (int, np.integer)):
            for key in df.columns:
                self._binsize_glob[key] = bin_size
        elif isinstance(bin_size, Mapping):
            self._binsize_glob.update(bin_size)
        return self

    @contextmanager
    def temp_glob(self, props: _DataFrameLike | None = None, **kwargs):
        """Temporarily update the global properties."""
        if kwargs:
            if props is not None:
                raise ValueError("Cannot specify both props and kwargs.")
            props = pl.DataFrame(kwargs)
        old_df = self._glob
        self._glob = old_df.with_columns(props)
        try:
            yield
        finally:
            self._glob = old_df

    def drop_loc(self, keys: str | Iterable[str]) -> Self:
        """Drop local properties."""
        if isinstance(keys, str):
            keys = [keys]
        self._loc = self._loc.drop(keys)
        for key in keys:
            self._window_size.pop(key, None)
        return self

    def drop_glob(self, keys: str | Iterable[str]) -> Self:
        """Drop global propperties."""
        if isinstance(keys, str):
            keys = [keys]
        self._glob = self._glob.drop(keys)
        return self

    def clear_loc(self) -> Self:
        """Clear local properties."""
        self._loc = pl.DataFrame([])
        self._window_size.clear()
        return self

    def clear_glob(self) -> Self:
        """Clear global properties."""
        self._glob = pl.DataFrame([])
        return self

    def get_loc(self, key: str, default=_void) -> pl.Series:
        """
        Get a local property of the spline, similar to ``dict.get`` method.

        Parameters
        ----------
        key : str
            Local property key.
        default : any, optional
            Default value to return if key is not found, raise error by default.
        """
        if isinstance(key, str):
            if key in self.loc.columns:
                return self.loc[key]
            elif default is _void:
                raise KeyError(f"Key {key!r} not found in localprops.")
            return default
        elif isinstance(key, pl.Expr):
            if default is not _void:
                raise ValueError("Cannot specify default value for polars.Expr input.")
            return self.loc.select(key).to_series()
        else:
            raise TypeError("Key must be either str or polars.Expr.")

    def get_glob(self, key: str | pl.Expr, default=_void) -> Any:
        """
        Get a global property of the spline, similar to ``dict.get`` method.

        Parameters
        ----------
        key : str or Expr
            Global property key.
        default : any, optional
            Default value to return if key is not found, raise error by default.
        """
        if isinstance(key, str):
            if key in self.glob.columns:
                return self.glob[key][0]
            elif default is _void:
                raise KeyError(f"Key {key!r} not found in globalprops.")
            return default
        elif isinstance(key, pl.Expr):
            if default is not _void:
                raise ValueError("Cannot specify default value for polars.Expr input.")
            return self.glob.select(key).to_series()[0]
        else:
            raise TypeError("Key must be either str or polars.Expr.")

    def has_loc(self, keys: str | Iterable[str]) -> bool:
        """Check if *all* the keys are in local properties."""
        if isinstance(keys, str):
            keys = [keys]
        return all(key in self.loc.columns for key in keys)

    def has_glob(self, keys: str | Iterable[str]) -> bool:
        """Check if *all* the keys are in global properties."""
        if isinstance(keys, str):
            keys = [keys]
        return all(key in self.glob.columns for key in keys)


def _pos_float(x: Any) -> float:
    out = float(x)
    if out < 0:
        raise ValueError("Value must be positive.")
    return out
