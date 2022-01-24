from __future__ import annotations

from black import os
from .._utils import translate_command
from ...const import Order
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

class IMOD(SimpleNamespace):
    """IMOD commands."""
    model2point = translate_command("model2point")
    point2model = translate_command("point2model")
    _3dmod = translate_command("3dmod")
    


def read_mod(path: str, order: str | Order = "zyx") -> pd.DataFrame:
    """
    
    Read a mod file.
    
    A mod file stores 3D coordinates in xyz-order. This function read it as a data
    frame in either xyz- or zyx-order.

    Parameters
    ----------
    path : str
        Path to mod file.
    order : str or Order, default is "zyx"
        The order of dimension of output data frame.

    Returns
    -------
    pd.DataFrame
        (N, 3) data frame with coordinates.
    """    
    path = str(path)
    order = Order(order)
    with tempfile.NamedTemporaryFile(mode="r+") as fh:
        output_path = fh.name
        IMOD.model2point(input=path, output=output_path)
        df: pd.DataFrame = pd.read_csv(output_path, sep="\s+", header=None)
        df.columns = ["x", "y", "z"]
    if Order.zyx:
        df = df[["z", "y", "x"]]
    return df


def save_mod(path: str, data):
    """
    Save array data as a mod file that can be used in IMOD.

    Parameters
    ----------
    path : str
        Saving path.
    data : array-like
        Data that will be saved.
    """    
    # Unlike read_mod, permission error usually occurs in this function.
    # Here we don't use tempfile to avoid it.
    path = str(path)
    if not path.endswith(".mod"):
        raise ValueError("File path must end with '.mod'.")
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Input data must be (N, 3) array, got {data.shape}.")
    
    input_path = os.path.splitext(path)[0] + ".csv"
    i = 0
    while os.path.exists(input_path):
        input_path = os.path.splitext(path)[0] + f"-{i}.csv"
        i += 1
    np.savetxt(input_path, data, fmt="%.5f", delimiter="\t")
    IMOD.point2model(input=input_path, output=path)
    os.remove(input_path)
    return None