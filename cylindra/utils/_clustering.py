from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from cylindra._cylindra_ext import activate_lateral, activate_longitudinal
from cylindra.const import MoleculesHeader as Mole

if TYPE_CHECKING:
    from acryo import Molecules

Edge = tuple[int, int, int, int]  # i_from, j_from, i_to, j_to


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent: list[int] = list(range(n))
        self.rank: list[int] = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _build_edges(
    nth: pl.Series,
    pf: pl.Series,
    mole: Molecules,
    start: int,
    func_long: Callable[[float, float, float], bool],
    func_lat: Callable[[float, float, float], bool],
) -> tuple[list[Edge], list[Edge]]:
    npf = pf.n_unique()
    idx: dict[tuple[int, int], int] = {}
    for i, (n, p) in enumerate(zip(nth, pf, strict=False)):
        idx[(int(n), int(p))] = i

    long_edges: list[Edge] = []
    lat_edges: list[Edge] = []
    _x = mole.x
    _y = mole.y
    _z = mole.z
    pos = mole.pos
    for (n, p), i in idx.items():
        # get longitudinal neighbor
        j = idx.get((n + 1, p))

        if j is not None:
            _ri, _rj = _get_args(pos, _z, _y, _x, i, j)
            if func_long(*_ri) and func_long(*_rj):
                long_edges.append((p, n, i, j))

        # get lateral neighbor
        if p + 1 < npf:
            j = idx.get((n, p + 1))
        else:
            j = idx.get((n + start, 0))  # seam crossed

        if j is not None:
            _ri, _rj = _get_args(pos, _z, _y, _x, i, j)
            if func_lat(*_ri) and func_lat(*_rj):
                lat_edges.append((p, n, i, j))

    return long_edges, lat_edges


def _get_args(pos, zs, ys, xs, i, j):
    dr = pos[j] - pos[i]
    dr_inv = -dr
    _rx_i = float(xs[i].dot(dr))
    _ry_i = float(ys[i].dot(dr))
    _rz_i = float(zs[i].dot(dr))
    _rx_j = float(xs[j].dot(dr_inv))
    _ry_j = float(ys[j].dot(dr_inv))
    _rz_j = float(zs[j].dot(dr_inv))
    _norm = float(np.sqrt(dr.dot(dr)))
    return (_rz_i, _ry_i, _rx_i, _norm), (_rz_j, _ry_j, _rx_j, _norm)


_NAMESPACE = {
    "__builtins__": {},
    "np": np,
    "math": math,
    "bool": bool,
    "int": int,
    "float": float,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}


def _expr_to_zyxd_func(expr: str) -> Callable[[float, float, float], bool]:
    """Convert a string expression to a callable function."""

    func_code = (
        "def func(z: float, y: float, x: float, d: float) -> bool:\n"
        f"    return {expr}\n"
    )
    _locals = {}
    exec(func_code, _NAMESPACE, _locals)
    return _locals["func"]


def cluster_molecules(
    mole: Molecules,
    npf: int,
    start: int = -3,
    expr_long: str = "True",
    expr_lat: str = "True",
    min_long_connections: int = 1,
    min_lat_connections: int = 2,
) -> NDArray[np.int_]:
    long_edges, lat_edges = _build_edges(
        mole.features[Mole.nth],
        mole.features[Mole.pf],
        mole,
        start,
        _expr_to_zyxd_func(expr_long),
        _expr_to_zyxd_func(expr_lat),
    )
    n_mole = mole.count()

    uf = UnionFind(n_mole)
    for k in activate_longitudinal(long_edges, npf, min_long_connections):
        _, _, i, j = long_edges[k]
        uf.union(i, j)
    for k in activate_lateral(lat_edges, long_edges, npf, start, min_lat_connections):
        _, _, i, j = lat_edges[k]
        uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(n_mole)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels
