from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def read_mod(path: str) -> pl.DataFrame:
    """Read a mod file.

    A mod file stores 3D coordinates in xyz-order. This function read it as a data
    frame in either xyz- or zyx-order.

    Parameters
    ----------
    path : str
        Path to mod file.

    Returns
    -------
    pd.DataFrame
        (N, 3) data frame with coordinates.
    """
    import imodmodel

    df = imodmodel.read(path)
    return pl.DataFrame(df)


def save_mod(path: str, data: pl.DataFrame):
    """Save array data as a mod file that can be used in IMOD.

    Parameters
    ----------
    path : str
        Saving path.
    data : array-like
        Data that will be saved.
    """
    import imodmodel

    path = str(path)
    if not path.endswith(".mod"):
        raise ValueError("File path must end with '.mod'.")
    return imodmodel.write(data.to_pandas(), path)


def save_angles(path: str, euler_angle: np.ndarray = None):
    """Save angle data for PEET subtomogram averaging."""
    euler_angle = np.asarray(euler_angle)
    size = euler_angle.shape[0]
    z1 = -euler_angle[:, 0]
    x2 = -euler_angle[:, 1]
    z3 = -euler_angle[:, 2]

    data = {
        "CCC": np.ones(size, dtype=np.float32),
        "reserved_1": np.zeros(size, dtype=np.uint8),
        "reserved_2": np.zeros(size, dtype=np.uint8),
        "pIndex": np.arange(1, size + 1, dtype=np.uint16),
        "wedgeWT": np.zeros(size, dtype=np.uint8),
        "NA_1": np.zeros(size, dtype=np.uint8),
        "NA_2": np.zeros(size, dtype=np.uint8),
        "NA_3": np.zeros(size, dtype=np.uint8),
        "NA_4": np.zeros(size, dtype=np.uint8),
        "NA_5": np.zeros(size, dtype=np.uint8),
        "xOffset": np.zeros(size, dtype=np.float32),
        "yOffset": np.zeros(size, dtype=np.float32),
        "zOffset": np.zeros(size, dtype=np.float32),
        "NA_6": np.zeros(size, dtype=np.uint8),
        "NA_7": np.zeros(size, dtype=np.uint8),
        "reserved": np.zeros(size, dtype=np.uint8),
        "EulerZ(1)": z1,
        "EulerZ(3)": z3,
        "EulerX(2)": x2,
        "reserved_3": np.zeros(size, dtype=np.uint8),
    }
    df = pl.DataFrame(data)
    text = df.write_csv(float_precision=3, include_header=False)
    header_text = ",".join(s.split("_")[0] for s in df.columns)
    with open(path, "w") as fh:
        fh.write(header_text + "\n" + text)


def read_edf(path: str) -> dict[str, Any]:
    """Read the IMOD project .edf file."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):  # comment line
            continue
        field_name, field_value = line.split("=", 1)
        out[field_name.strip()] = field_value.strip()
    return out
