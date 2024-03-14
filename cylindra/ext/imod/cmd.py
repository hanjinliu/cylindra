from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# NOTE: output of model2point has separator "\s+". This is only supported in pandas.
import pandas as pd
import polars as pl

from cylindra.ext._utils import CommandNotFound, translate_command


class IMODCommand(SimpleNamespace):
    """IMOD commands."""

    model2point = translate_command("model2point")
    point2model = translate_command("point2model")
    _3dmod = translate_command("3dmod")


def read_mod(path: str) -> pl.DataFrame:
    """

    Read a mod file.

    A mod file stores 3D coordinates in xyz-order. This function read it as a data
    frame in either xyz- or zyx-order.

    Parameters
    ----------
    path : str
        Path to mod file.
    order : str, default "zyx"
        The order of dimension of output data frame.

    Returns
    -------
    pd.DataFrame
        (N, 3) data frame with coordinates.
    """
    path = str(path)
    if IMODCommand.model2point.available():
        with tempfile.NamedTemporaryFile(mode="r+") as fh:
            output_path = fh.name
            IMODCommand.model2point(
                input=path, output=output_path, object=True, contour=True
            )
            df = pd.read_csv(output_path, sep=r"\s+", header=None)
            df.columns = ["object_id", "contour_id", "x", "y", "z"]
    else:
        try:
            import imodmodel
        except ImportError:
            raise CommandNotFound(
                "To read mod file, either the `model2point` command of IMOD or the "
                "Python package `imodmodel` is required."
            ) from None
        df = imodmodel.read(path)
    return pl.DataFrame(df)


def save_mod(path: str, data: pl.DataFrame):
    """
    Save array data as a mod file that can be used in IMOD.

    Parameters
    ----------
    path : str
        Saving path.
    data : array-like
        Data that will be saved.
    """
    path = str(path)
    if not path.endswith(".mod"):
        raise ValueError("File path must end with '.mod'.")

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        input_path = tempdir / "input.txt"
        text = data.write_csv(separator=",", include_header=False, float_precision=5)
        bk = "     "
        text = (
            "\n".join(bk + line.replace(",", bk) for line in text.splitlines()) + "\n"
        )
        Path(input_path).write_text(text)
        IMODCommand.point2model(input=input_path, output=path)
    return None


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
    return None
