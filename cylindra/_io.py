from __future__ import annotations

import impy as ip


def lazy_imread(path, chunks) -> ip.LazyImgArray:
    return ip.lazy.imread(path, chunks=chunks)


def read_header(path):
    return ip.read_header(path)
