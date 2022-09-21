from __future__ import annotations

from acryo import Molecules
import numpy as np

from .spline import Spline
from ..utils import roundint, oblique_meshgrid
from ..const import Mole

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
        rate: float = 0.03,
        patch_size: tuple[int, int] = (3, 3),
    ) -> CylindricModel:
        ysize, asize = patch_size
        ystart = at[0] - ysize / 2
        ystop = at[0] + ysize / 2
        astart = at[1] - asize / 2
        astop = at[1] + asize / 2
        
        