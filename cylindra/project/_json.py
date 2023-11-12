from enum import Enum
from pathlib import Path


def project_json_encoder(obj):
    """An enhanced encoder."""
    import numpy as np
    import pandas as pd
    import polars as pl

    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pl.DataFrame):
        return obj.to_dict(as_series=False)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        # return as a POSIX path
        if obj.is_absolute():
            return obj.as_posix()
        else:
            return "./" + obj.as_posix()
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")
