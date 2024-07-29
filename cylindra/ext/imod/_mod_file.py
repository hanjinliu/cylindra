from __future__ import annotations

import struct

import numpy as np

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


def i32(i: int):
    return i.to_bytes(4, signed=True)


def u32(i: int):
    return i.to_bytes(4, signed=False)


def f32(f: float):
    return struct.pack(">f", f)


def f32s(fs):
    n = len(fs)
    return struct.pack(">" + "f" * n, *fs)


def write_array(arr: np.ndarray, path: str):
    npoints = arr.shape[0]
    xmax = int(arr[:, 0].max()) + 1
    ymax = int(arr[:, 1].max()) + 1
    zmax = int(arr[:, 2].max()) + 1
    val = (
        b"IMODV1.2IMOD-NewModel"
        + b" " * 115  # model name
        + i32(xmax)
        + i32(ymax)
        + i32(zmax)
        + i32(1)  # obj size
        + u32(15360)  # flags
        + i32(1)  # drawmode
        + i32(2)  # mousemode
        + i32(0)  # blacklevel
        + i32(255)  # whitelevel
        + f32(0.0) * 3  # offsets
        + f32(1.0) * 3  # scales
        + i32(0)  # current object
        + i32(0)  # current contour
        + i32(0)  # current point
        + i32(3)  # res
        + i32(128)  # threshold
        + f32(1.0)  # pixsize
        + i32(0)  # unit, 0 = pixels
        + i32(0)  # Checksum storage
        + f32(0.0) * 3  # alpha, beta, gamma
        + OBJECT_BYTES
        + CONT
        + i32(npoints)
        + u32(3)
        + i32(0)
        + i32(0)
        + f32s(arr.ravel().tolist())
        + IEOF
    )
    with open(path, "wb") as f:
        f.write(val)
