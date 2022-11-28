from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence
from pathlib import Path
import numpy as np
from cylindra.const import IDName, PropertyNames as H

if TYPE_CHECKING:
    import pandas as pd

COLUMNS = [H.splDistance, H.splPosition, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]

_AGG_FUNCS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "sum": np.nansum,
}

class DataFrameList(Sequence["pd.DataFrame"]):
    def __init__(self, df_list: Sequence[pd.DataFrame]):
        self._list = list(df_list)
    
    @classmethod
    def from_localprops(cls, df: pd.DataFrame):
        input_data = [df_sub[COLUMNS] for _, df_sub in df.groupby(by=IDName.spline)]
        return cls(input_data)
    
    @classmethod
    def from_csv(cls, path):
        import pandas as pd
        df = pd.read_csv(path)
        return cls.from_localprops(df)
    
    @classmethod
    def glob_csv(cls, dir: str | Path):
        dir = Path(dir)
        if not dir.is_dir():
            raise ValueError(f"Input path must be a directory.")
        instances: list[cls] = []
        for p in dir.glob("**/localprops.csv"):
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
        strs = []
        for k, v in enumerate(self._list):
            strs.append(f"{k}: DataFrame of shape={v.shape!r}")
            if k > 12:
                strs.append("...")
                strs.append(f"{len(self._list) - 1}: DataFrame of shape={self._list[-1].shape!r}")
                break
        joined = ",\n\t".join(strs)
        return f"DataFrameList(\n\t{joined}\n)"

    def _agg_pos(self, fn):
        return [df.agg(fn) for df in self._list]
    
    def _agg_id(self, fn):
        import pandas as pd
        max_len = max(len(df) for df in self._list)
        all_df = []
        for df in self._list:
            nr, nc = df.shape
            if nr < max_len:
                df = pd.concat([df, np.full((max_len - nr, nc), np.nan)])
            all_df.append(df)

            columns = df.columns
        
        df_out_dict = {}
        for col in columns:
            stacked = np.stack([df[col] for df in all_df], axis=1)
            df_out_dict[col] = fn(stacked, axis=1)
        return pd.DataFrame(df_out_dict)
            
    def collect(self) -> pd.DataFrame:
        """Collect all the child data frames into a single data frame."""
        import pandas as pd

        return pd.concat(self._list, axis=0)
    
    @property
    def loc(self) -> LocIndexer:
        """Iterative loc indexer."""
        return LocIndexer(self)
    
    @property
    def iloc(self) -> ILocIndexer:
        """Iterative iloc indexer."""
        return ILocIndexer(self)
    
    def select(self, col):
        cls = type(self)
        return cls({k: df[col] for k, df in self.items()})

    def agg_id(self, fname: str):
        fn = _AGG_FUNCS[fname]
        return self._agg_id(fn)
    
    def agg_pos(self, fn):
        return self._agg_pos(fn)      


class LocIndexer:
    def __init__(self, dd: DataFrameList):
        self._dd = dd
    
    def __getitem__(self, key) -> pd.DataFrame:
        cls = type(self._dd)
        return cls([df.loc[key] for df in self._dd])
    

class ILocIndexer:
    def __init__(self, dd: DataFrameList):
        self._dd = dd
    
    def __getitem__(self, key) -> pd.DataFrame:
        cls = type(self._dd)
        return cls([df.iloc[key] for df in self._dd])
