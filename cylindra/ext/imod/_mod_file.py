from __future__ import annotations

from struct import Struct

import numpy as np

MODEL_NAME = b"IMODV1.2IMOD-NewModel" + b" " * 115
OBJECT_BYTES = (
    b"OBJT\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x18\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x01\x00\x00\x00\x00?\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x01\x03\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
)
CONT = b"CONT"
IEOF = b"IEOF"

_I32 = Struct(">i")
_U32 = Struct(">I")
_F32 = Struct(">f")


def f32s(fs):
    n = len(fs)
    st = Struct(">" + "f" * n)
    return st.pack(*fs)


def write_array(arr: np.ndarray, path: str):
    npoints = arr.shape[0]
    xmax = int(arr[:, 0].max()) + 1
    ymax = int(arr[:, 1].max()) + 1
    zmax = int(arr[:, 2].max()) + 1
    val = (
        MODEL_NAME
        + _I32.pack(xmax)
        + _I32.pack(ymax)
        + _I32.pack(zmax)
        + _I32.pack(1)  # obj size
        + _U32.pack(15360)  # flags
        + _I32.pack(1)  # drawmode
        + _I32.pack(2)  # mousemode
        + _I32.pack(0)  # blacklevel
        + _I32.pack(255)  # whitelevel
        + _F32.pack(0.0) * 3  # offsets
        + _F32.pack(1.0) * 3  # scales
        + _I32.pack(0)  # current object
        + _I32.pack(0)  # current contour
        + _I32.pack(0)  # current point
        + _I32.pack(3)  # res
        + _I32.pack(128)  # threshold
        + _F32.pack(1.0)  # pixsize
        + _I32.pack(0)  # unit, 0 = pixels
        + _I32.pack(0)  # Checksum storage
        + _F32.pack(0.0) * 3  # alpha, beta, gamma
        + OBJECT_BYTES
        + CONT
        + _I32.pack(npoints)
        + _U32.pack(3)
        + _I32.pack(0)
        + _I32.pack(0)
        + f32s(arr.ravel().tolist())
        + IEOF
    )
    with open(path, "wb") as f:
        f.write(val)
