from __future__ import annotations
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from acryo import Molecules
import numpy as np

from .spline import Spline
from ..utils import oblique_meshgrid
from ..const import Mole

if TYPE_CHECKING:
    from typing_extensions import Self


class CylindricModel:
    """
    A model class that describes a heterogenic cylindrical structure.
    
    Parameters
    ----------
    shape : (int, int)
        Shape of the molecule grid in (axial, angular) shape.
    tilts : (float, float)
        Relative tilt tangent in (axial, angular) direction.
    intervals : (float, float)
        Intervals in (axial, angular) direction.
    radius : float
        Radius of the cylindrical structure.
    offsets : (float, float)
        Offsets in (axial, angular) direction.
    displace : (Ny, Npf, 3) array
        Displacement vector of each molecule.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        tilts: tuple[float, float],
        intervals: tuple[float, float] = (1., 1.),
        radius: float = 1.0,
        offsets: tuple[float, float] = (0., 0.),
        displace: np.ndarray | None = None
    ):
        self._shape = shape
        self._tilts = tilts
        self._intervals = intervals
        self._offsets = offsets
        self._radius = radius
        if displace is None:
            self._displace = np.zeros(shape + (3,), dtype=np.float32)
        else:
            if displace.shape != shape + (3,):
                raise ValueError("Shifts shape mismatch")
            self._displace = displace
    
    def replace(
        self,
        tilts: tuple[float, float] | None = None,
        intervals: tuple[float, float] | None = None,
        radius: float | None = None,
        offsets: tuple[float, float] | None = None,
        displace: np.ndarray | None = None,
    ) -> Self:
        """Create a new model with the same shape but different parameters."""

        if tilts is None:
            tilts = self._tilts
        if intervals is None:
            intervals = self._intervals
        if radius is None:
            radius = self._radius
        if offsets is None:
            offsets = self._offsets
        if displace is None:
            displace = self._displace.copy()
        return self.__class__(
            shape=self._shape,
            tilts=tilts,
            intervals=intervals,
            radius=radius,
            offsets=offsets,
            displace=displace,
        )

    def copy(self) -> Self:
        """Make a copy of the model object."""
        return self.replace()
    
    __copy__ = copy
    
    @property
    def shape(self) -> tuple[int, int]:
        return self._shape
    
    @property
    def radius(self) -> float:
        return self._radius
    
    @property
    def tilts(self) -> tuple[float, float]:
        return self._tilts

    @property
    def intervals(self) -> tuple[float, float]:
        return self._intervals

    @property
    def offsets(self) -> tuple[float, float]:
        return self._offsets

    @property
    def displace(self) -> np.ndarray:
        return self._displace
    
    def __repr__(self) -> str:
        _cls = type(self).__name__
        strs: list[str] = []
        for k, v in self.to_params().items():
            if isinstance(v, float):
                strs.append(f"{k}={v:.3g}")
            elif isinstance(v, tuple):
                vstr = ", ".join(f"{x:.3g}" for x in v)
                strs.append(f"{k}=({vstr})")
            else:
                strs.append(f"{k}={v}")
        strs = ", ".join(strs)
        return f"{_cls}({strs})"
    
    def to_params(self) -> dict[str, Any]:
        """Describe the model state as a dictionary."""
        return {
            "shape": self._shape,
            "tilts": self._tilts,
            "intervals": self._intervals,
            "radius": self._radius,
            "offsets": self._offsets,
        }
    
    def _create_mesh(self) -> np.ndarray:
        mesh = oblique_meshgrid(
            self._shape, self._tilts, self._intervals, self._offsets
        )  # (Ny, Npf, 2)
        radius_arr = np.full(mesh.shape[:2] + (1,), self._radius, dtype=np.float32)
        return np.concatenate([radius_arr, mesh], axis=2)  # (Ny, Npf, 3)
    
    def to_molecules(self, spl: Spline) -> Molecules:
        """Generate molecules from the coordinates and given spline."""
        mesh = oblique_meshgrid(
            self._shape, self._tilts, self._intervals, self._offsets
        )  # (Ny, Npf, 2)
        radius_arr = np.full(mesh.shape[:2] + (1,), self._radius, dtype=np.float32)
        mesh3d = np.concatenate([radius_arr, mesh], axis=2)  # (Ny, Npf, 3)
        shifted = mesh3d + self._displace
        mole = spl.cylindrical_to_molecules(shifted.reshape(-1, 3))
        mole.features = {Mole.pf: np.arange(len(mole), dtype=np.uint32) % self._shape[1]}
        return mole

    def add_offsets(self, offsets: tuple[float, float]) -> Self:
        """Increment offsets attribute of the model."""
        _offsets = tuple(x + y for x, y in zip(self._offsets, offsets))
        return replace(offsets=_offsets)
    
    def add_shift(self, shift: np.ndarray) -> Self:
        """Increment displace attribute of the model."""
        displace = self._displace + shift
        return replace(displace=displace)
    
    def add_radial_shift(self, shift: np.ndarray) -> Self:
        return self._add_directional_shift(shift, 0)

    def add_axial_shift(self, shift: np.ndarray) -> Self:
        return self._add_directional_shift(shift, 1)
    
    def add_skew_shift(self, shift: np.ndarray) -> Self:
        return self._add_directional_shift(shift, 2)
    
    def dilate(self, radius_shift: float, start: int, stop: int) -> Self:
        return self._add_local_uniform_directional_shift(radius_shift, start, stop, 0)
    
    def expand(self, yshift: float, start: int, stop: int) -> Self:
        return self._add_local_uniform_directional_shift(yshift, start, stop, 1)
        
    def screw(self, angle_shift: float, start: int, stop: int) -> Self:
        return self._add_local_uniform_directional_shift(angle_shift, start, stop, 2)
    
    def _add_directional_shift(self, displace: np.ndarray, axis: int) -> Self:
        displace = self._displace.copy()
        displace[:, :, axis] += displace
        return replace(displace=displace)
    
    def _add_local_uniform_directional_shift(
        self, shift: float, start: int, stop: int, axis: int
    ) -> Self:
        displace = self._displace.copy()
        for idx in range(start, stop):
            displace[idx:, :, axis] += shift
        return self.replace(displace=displace)
        