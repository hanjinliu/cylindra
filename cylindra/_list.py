from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence, SupportsIndex, TypeVar
from pathlib import Path
import numpy as np
from cylindra.const import IDName, PropertyNames as H
import polars as pl

if TYPE_CHECKING:
    import pandas as pd

_R = TypeVar("_R")
COLUMNS = [H.splDistance, H.splPosition, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]

def nansize(arr: np.ndarray, axis=0):
    """Return the number of non-NaN elements in the array along a given axis."""
    return np.sum(~np.isnan(arr), axis=axis)

def nansem(arr: np.ndarray, axis=0):
    """Return the standard error of the mean of the array along a given axis."""
    return np.nanstd(arr, axis=axis) / np.sqrt(nansize(arr, axis=axis))

_AGG_FUNCS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "sem": nansem,
    "var": np.nanvar,
    "size": nansize,
    "sum": np.nansum,
}

_INHERIT_DTYPE = {"min", "max", "size", "sum"}

class DataFrameList(Sequence["pd.DataFrame"]):
    """
    List of DataFrames for analysis of multiple local properties.
    
    >>> dl = DataFrameList.from_csv("path/to/localprops.csv")
    >>> dl.agg_id("mean")  # aggregate over all splines
    >>> dl.agg_pos("mean")  # aggregate over all spline positions
    """
    def __init__(self, df_list: Sequence[pd.DataFrame]):
        self._list = list(df_list)
    
    @classmethod
    def from_localprops(cls, df: pl.DataFrame) -> DataFrameList:
        """Construct a dataframe list from a localprops attribute of a Spline object."""
        input_data = [df_sub[COLUMNS].to_pandas() for _, df_sub in df.groupby(by=IDName.spline)]
        return cls(input_data)
    
    @classmethod
    def from_csv(cls, path) -> DataFrameList:
        """Construct a dataframe list from a csv file."""
        df = pl.read_csv(path)
        if IDName.spline in df.columns and IDName.pos in df.columns:
            df = df.with_columns(pl.repeat(Path(path).parent.stem, pl.count()).alias("source"))
            return cls.from_localprops(df)
        raise ValueError(
            f"The csv file does not contain {IDName.spline!r} and {IDName.pos!r} columns."
        )
    
    @classmethod
    def glob_csv(cls, dir: str | Path, filename: str = "localprops.csv"):
        dir = Path(dir)
        if not dir.is_dir():
            raise ValueError(f"Input path must be a directory.")
        instances: list[cls] = []
        for p in dir.glob(f"**/{filename}"):
            instances.append(cls.from_csv(p)._list)
        return cls(sum(instances, start=[]))
    
    def __getitem__(self, index: Any) -> pd.DataFrame:
        return self._list[index]
    
    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter(self._list)
    
    def __len__(self) -> int:
        return len(self._list)
    
    def __repr__(self) -> str:
        if len(self) == 0:
            return f"DataFrameList()"
        strs: list[str] = []
        for k, v in enumerate(self._list):
            strs.append(f"{k}: DataFrame of shape={v.shape!r}")
            if k > 12:
                strs.append("...")
                strs.append(f"{len(self._list) - 1}: DataFrame of shape={self._list[-1].shape!r}")
                break
        joined = ",\n\t".join(strs)
        return f"DataFrameList(\n\t{joined}\n)"

    def agg_pos(self, fn):
        """Aggregate over all spline positions."""
        return [df.agg(fn) for df in self._list]
    
    def agg_id(self, fname: str, align: Literal["top", "bottom", "none"] = "top") -> pd.DataFrame:
        """
        Aggregate over all spline.
        
        Parameters
        ----------
        fname : str
            Name of the function to use for aggregation. Supported functions are: "mean", "median", 
            "min", "max", "std", "sem", "var", "size" and "sum".
        align : {"top", "bottom", "none"}, optional
            How to align the data frames before aggregation if lengths are different. If "top", they
            will be aligned at the index 0. If "bottom", they will be aligned at the last index. If
            "none", exception will be raised if lengths are different.
        """
        import pandas as pd
        
        fn = _AGG_FUNCS[fname]
        max_len = max(len(df) for df in self._list)
        all_df = []
        for df in self._list:
            nr, nc = df.shape
            if nr < max_len:
                if align == "none":
                    raise ValueError("DataFrames must be of equal length.")
                
                nan_arr = np.full((max_len - nr, nc), np.nan)
                if align == "top":
                    df = pd.concat(
                        [df, pd.DataFrame(nan_arr, columns=df.columns)],
                        axis=0,
                        ignore_index=True,
                    )
                elif align == "bottom":
                    df = pd.concat(
                        [pd.DataFrame(nan_arr, columns=df.columns), df],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    raise ValueError(f"Invalid align value: {align!r}")

            all_df.append(df)

            columns = df.columns
        
        df_out_dict = {}
        for col in columns:
            stacked = np.stack([df[col] for df in all_df], axis=1)
            df_out_dict[col] = fn(stacked, axis=1)
        
        df_out = pd.DataFrame(df_out_dict)
        if fname in _INHERIT_DTYPE:
            for col in columns:
                df_out[col] = df_out[col].astype(self._list[0][col].dtype)
        return df_out
            
    def collect(self) -> pd.DataFrame:
        """Collect all the child data frames into a single data frame."""
        import pandas as pd

        return pd.concat(self._list, axis=0, ignore_index=True)
    
    @property
    def iloc(self) -> ILocIndexer:
        """Iterative iloc indexer."""
        return ILocIndexer(self)
    
    def select(self, col) -> DataFrameList:
        """Select certain columns of each data frame."""
        cls = type(self)
        if isinstance(col, str):
            col = [col]
        return cls([df[col] for df in self])

    def subset(self, spec: Sequence[int]) -> DataFrameList:
        """Return a subset of the data frame list."""
        cls = type(self)
        spec: np.ndarray = np.asarray(spec)
        if spec.dtype.kind == "b":
            spec = np.where(spec)[0]
        spec = set(spec)
        return cls([df for i, df in enumerate(self) if i in spec])

    def apply(self, fn: Callable[[pd.DataFrame], _R]) -> list[_R]:
        """Apply a function to each data frame."""
        return [fn(df) for df in self]
    
    def build_gui(self, show: bool = True):
        """
        Build a GUI for the data frame list.
        
        Parameters
        ----------
        show : bool, default is True
            Whether to show the GUI.
        """
        from cylindra.widgets.dataframe_list import DataFrameListWidget
        
        ui = DataFrameListWidget()
        ui.set_data(self)
        if show:
            ui.show()
        return ui


class ILocIndexer:
    """Vectorized iloc indexer."""
    def __init__(self, dfl: DataFrameList):
        self._dfl = dfl
    
    def __getitem__(self, key) -> pd.DataFrame:
        cls = type(self._dfl)
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], SupportsIndex):
            # dataframe list must return dataframe, not series.
            k0, k1 = key
            key = (k0, slice(k1, k1 + 1))
        return cls([df.iloc[key] for df in self._dfl])
