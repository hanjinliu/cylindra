from __future__ import annotations
from typing import Any, NamedTuple

from acryo import Molecules
import numpy as np

from .spline import Spline
from ..utils import ceilint, oblique_meshgrid
from ..const import Mole

class Patch(NamedTuple):
    """Patch info"""

    start: int
    center: float
    stop: int


class CylindricModel:
    """The model class that describes a cylindrical structure of molecules."""

    def __init__(
        self,
        shape: tuple[int, int],
        tilts: tuple[float, float],
        intervals: tuple[float, float] = (1., 1.),
        radius: float = 1.0,
        offsets: tuple[float, float] = (0., 0.),
    ):
        self._shape = shape
        self._tilts = tilts
        self._intervals = intervals
        self._offsets = offsets
        self._radius = radius
        self._mesh: np.ndarray | None = None  # normalized mesh
    
    def __repr__(self) -> str:
        _cls = type(self).__name__
        s = ", ".join(f"{k}: {v}" for k, v in self.to_params().items())
        return f"{_cls}({s})"
    
    @classmethod
    def from_params(
        cls, 
        shape: tuple[int, int],
        tilts: tuple[float, float],
        intervals: tuple[float, float] = (1., 1.),
        radius: float = 1.0,
        offsets: tuple[float, float] = (0., 0.),
    ):
        self = cls(shape, tilts, intervals, radius, offsets)
        self._create_mesh()
        return self
    
    def to_params(self) -> dict[str, Any]:
        return {
            "shape": self._shape,
            "tilts": self._tilts,
            "intervals": self._intervals,
            "radius": self._radius,
            "offsets": self._offsets,
        }
    
    def _create_mesh(self):
        mesh = oblique_meshgrid(
            self._shape, self._tilts, self._intervals, self._offsets
        )  # (Ny, Npf, 2)
        radius_arr = np.full(mesh.shape[:2] + (1,), self._radius, dtype=np.float32)
        self._mesh = np.concatenate([radius_arr, mesh], axis=2)  # (Ny, Npf, 3)
        return self
    
    def to_molecules(self, spl: Spline) -> Molecules:
        """Generate molecules from the coordinates and given spline."""
        mole = spl.cylindrical_to_molecules(self._mesh.reshape(-1, 3))
        mole.features = {Mole.pf: np.arange(len(mole), dtype=np.uint32) % self._shape[1]}
        return mole

    def expand(
        self,
        at: tuple[float, float],
        yshift: float,
        patch_size: tuple[int, int] = (3, 3),
    ) -> CylindricModel:
        _y, _a = _parse_range_args(at, patch_size)

        mesh = self._mesh .copy()
        for y in range(_y.start, _y.stop):
            dy = (y - _y.center) * yshift / 2
            mesh[y, _a.start:_a.stop, 1] += dy
        
        new = self.__class__(**self.to_params())
        new._mesh = mesh
        return new
    
    def skew(
        self,
        at: tuple[float, float],
        angle_shift: float,
        patch_size: tuple[int, int] = (3, 3),
    ) -> CylindricModel:
        _y, _a = _parse_range_args(at, patch_size)

        mesh = self._mesh .copy()
        for a in range(_a.start, _a.stop):
            da = (a - _a.center) * angle_shift / 2
            mesh[_y.start:_y.stop, a, 2] += da
        
        new = self.__class__(**self.to_params())
        new._mesh = mesh
        return new

def _parse_range_args(
    at: tuple[float, float],
    patch_size: tuple[int, int],
) -> tuple[Patch, Patch]:
    ysize, asize = patch_size
    ycenter, acenter = at
    ystart = ceilint(ycenter - ysize / 2)
    ystop = int(ycenter + ysize / 2)
    astart = ceilint(acenter - asize / 2)
    astop = int(acenter + asize / 2)
    return Patch(ystart, ycenter, ystop), Patch(astart, acenter, astop)
