from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping
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

class DataFrameDict(Mapping[int, "pd.DataFrame"]):
    def __init__(self, df_dict: dict[int, pd.DataFrame]):
        self._dict = df_dict
    
    @classmethod
    def from_localprops(cls, df: pd.DataFrame):
        input_data: dict[int, pd.DataFrame] = {
            id_: df_sub[COLUMNS] for id_, df_sub in df.groupby(by=IDName.spline)
        }
        return cls(input_data)
    
    @classmethod
    def from_csv(cls, path):
        import pandas as pd
        df = pd.read_csv(path)
        return cls.from_localprops(df)
    
    def __getitem__(self, key: Any) -> pd.DataFrame:
        return self._dict[key]
    
    def __iter__(self) -> Iterator[pd.DataFrame]:
        return iter(self._dict)
    
    def __len__(self) -> int:
        return len(self._dict)
    
    def __repr__(self) -> str:
        strs = []
        for k, v in self._dict.items():
            strs.append(f"{k}: DataFrame of shape={v.shape!r}")
        joined = ",\n\t".join(strs)
        return f"DataFrameDict(\n\t{joined}\n)"

    def _agg_pos(self, fn):
        return {k: df.agg(fn) for k, df in self._dict.items()}
    
    def _agg_id(self, fn):
        import pandas as pd
        max_len = max(len(df) for df in self._dict.values())
        all_df = []
        for df in self._dict.values():
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
            
    def flatten(self) -> pd.DataFrame:
        import pandas as pd
        return pd.concat(list(self._dict.values()), axis=0)
    
    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)
    
    @property
    def iloc(self) -> ILocIndexer:
        return ILocIndexer(self)

    def agg_id(self, fname: str):
        fn = _AGG_FUNCS[fname]
        return self._agg_id(fn)
    
    def agg_pos(self, fn):
        return self._agg_pos(fn)

class LocIndexer:
    def __init__(self, dd: DataFrameDict):
        self._dd = dd
    
    def __getitem__(self, key) -> pd.DataFrame:
        cls = type(self._dd)
        return cls({k: df.loc[key] for k, df in self._dd.items()})
    

class ILocIndexer:
    def __init__(self, dd: DataFrameDict):
        self._dd = dd
    
    def __getitem__(self, key) -> pd.DataFrame:
        cls = type(self._dd)
        return cls({k: df.iloc[key] for k, df in self._dd.items()})
        
