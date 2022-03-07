from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

from .._utils import translate_command

class IMOD(SimpleNamespace):
    """IMOD commands."""
    model2point = translate_command("model2point")
    point2model = translate_command("point2model")
    _3dmod = translate_command("3dmod")
    

def read_mod(path: str, order: str = "zyx") -> pd.DataFrame:
    """
    
    Read a mod file.
    
    A mod file stores 3D coordinates in xyz-order. This function read it as a data
    frame in either xyz- or zyx-order.

    Parameters
    ----------
    path : str
        Path to mod file.
    order : str, default is "zyx"
        The order of dimension of output data frame.

    Returns
    -------
    pd.DataFrame
        (N, 3) data frame with coordinates.
    """    
    path = str(path)
    if order not in ("xyz", "zyx"):
        raise ValueError("order must be either 'zyx' or 'xyz'.")
    with tempfile.NamedTemporaryFile(mode="r+") as fh:
        output_path = fh.name
        IMOD.model2point(input=path, output=output_path)
        df: pd.DataFrame = pd.read_csv(output_path, sep="\s+", header=None)
        df.columns = ["x", "y", "z"]
    if order == "zyx":
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


def save_angles(path: str, euler_angle: np.ndarray = None):
    """Save angle data for PEET subtomogram averaging."""
    euler_angle = np.asarray(euler_angle)
    size = euler_angle.shape[0]
    z1 = -euler_angle[:, 0]
    x2 = -euler_angle[:, 1]
    z3 = -euler_angle[:, 2]
    
    columns = ["CCC", "reserved", "reserved", "pIndex", "wedgeWT", "NA", "NA", "NA", "NA",
               "NA", "xOffset", "yOffset", "zOffset", "NA", "NA", "reserved", "EulerZ(1)", 
               "EulerZ(3)", "EulerX(2)", "reserved"]
    
    data = {"CCC": np.ones(size, dtype=np.float32),
            "reserved1": np.zeros(size, dtype=np.uint8),
            "reserved2": np.zeros(size, dtype=np.uint8),
            "pIndex": np.arange(1, size+1, dtype=np.uint16),
            "wedgeWT": np.zeros(size, dtype=np.uint8),
            "NA1": np.zeros(size, dtype=np.uint8),
            "NA2": np.zeros(size, dtype=np.uint8),
            "NA3": np.zeros(size, dtype=np.uint8),
            "NA4": np.zeros(size, dtype=np.uint8),
            "NA5": np.zeros(size, dtype=np.uint8),
            "xOffset": np.zeros(size, dtype=np.float32),
            "yOffset": np.zeros(size, dtype=np.float32),
            "zOffset": np.zeros(size, dtype=np.float32),
            "NA6": np.zeros(size, dtype=np.uint8),
            "NA7": np.zeros(size, dtype=np.uint8),
            "reserved": np.zeros(size, dtype=np.uint8),
            "EulerZ(1)": z1,
            "EulerZ(3)": z3,
            "EulerX(2)": x2,
            "reserved3": np.zeros(size, dtype=np.uint8),
            }
    df = pd.DataFrame(data)
    df.columns = columns
    df.to_csv(path, float_format="%.3f", index=False)
    return None
