from __future__ import annotations
from .._utils import translate_command
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

class IMOD(SimpleNamespace):
    """IMOD commands."""
    model2point = translate_command("model2point")
    point2model = translate_command("point2model")
    _3dmod = translate_command("3dmod")
    


def read_mod(path: str, order="zyx") -> pd.DataFrame:
    path = str(path)
    with tempfile.NamedTemporaryFile(mode="r+") as fh:
        output_path = fh.name
        IMOD.model2point(input=path, output=output_path)
        df: pd.DataFrame = pd.read_csv(output_path, sep="\s+", header=None)
        df.columns = ["x", "y", "z"]
    if order == "zyx":
        df = df[["z", "y", "x"]]
    elif order == "xyz":
        pass
    else:
        raise ValueError(f"order={order} is not allowed.")
    return df


def save_mod(path: str, data):
    path = str(path)
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Input data must be (N, 3) array, got {data.shape}.")
    with tempfile.NamedTemporaryFile(mode="r+") as fh:
        input_path = fh.name
        np.savetxt(input_path, data, fmt="%.5f", delimiter="\t")
        IMOD.point2model(input=input_path, output=path)
    return None