from __future__ import annotations

from typing import Any

import impy as ip


def lazy_imread(path, chunks: Any = "auto") -> ip.LazyImgArray:
    return ip.lazy.imread(path, chunks=chunks)


def read_header(path):
    return ip.read_header(path)
