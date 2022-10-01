from __future__ import annotations
from typing import Any, TYPE_CHECKING

from acryo import Molecules
import numpy as np
from numpy.typing import ArrayLike

from .spline import Spline
from ..const import Mole

if TYPE_CHECKING:
    from typing_extensions import Self


class CylinderModel:
    """
    A model class that describes a heterogenic cylindrical structure.
    
    Parameters
    ----------
    shape : (int, int)
        Shape of the molecule grid in (axial, angular) shape.
    tilts : (float, float)
        Relative tilt tangent in (axial, angular) direction.
    interval : float
        Axial interval.
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
        interval: float = 1.0,
        radius: float = 1.0,
        offsets: tuple[float, float] = (0., 0.),
        displace: np.ndarray | None = None
    ):
        self._shape = shape
        self._tilts = tilts
        self._intervals = (interval, 2 * np.pi / shape[1])
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
        interval: float | None = None,
        radius: float | None = None,
        offsets: tuple[float, float] | None = None,
        displace: np.ndarray | None = None,
    ) -> Self:
        """Create a new model with the same shape but different parameters."""

        if tilts is None:
            tilts = self._tilts
        if interval is None:
            interval = self._intervals[0]
        if radius is None:
            radius = self._radius
        if offsets is None:
            offsets = self._offsets
        if displace is None:
            displace = self._displace.copy()
        return self.__class__(
            shape=self._shape,
            tilts=tilts,
            interval=interval,
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
    
    @property
    def nrise(self) -> int:
        return int(np.round(self.tilts[1] * self.shape[1]))
    
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
    
    def to_molecules(self, spl: Spline) -> Molecules:
        """Generate molecules from the coordinates and given spline."""
        shifted = self._get_shifted()
        mole = spl.cylindrical_to_molecules(shifted.reshape(-1, 3))
        mole.features = {Mole.pf: np.arange(len(mole), dtype=np.uint32) % self._shape[1]}
        return mole

    def add_offsets(self, offsets: tuple[float, float]) -> Self:
        """Increment offsets attribute of the model."""
        _offsets = tuple(x + y for x, y in zip(self._offsets, offsets))
        return self.replace(offsets=_offsets)
    
    def add_shift(self, shift: np.ndarray) -> Self:
        """Increment displace attribute of the model."""
        displace = self._displace + shift
        return self.replace(displace=displace)
    
    def add_radial_shift(self, shift: np.ndarray) -> Self:
        """Add shift to the radial (r-axis) direction."""
        return self._add_directional_shift(shift, 0)

    def add_axial_shift(self, shift: np.ndarray) -> Self:
        """Add shift to the axial (y-axis) direction."""
        return self._add_directional_shift(shift, 1)
    
    def add_skew_shift(self, shift: np.ndarray) -> Self:
        """Add shift to the skew (a-axis) direction."""
        return self._add_directional_shift(shift, 2)
    
    def dilate(self, radius_shift: float, start: int, stop: int) -> Self:
        """Locally add uniform shift to the radial (r-axis) direction."""
        return self._add_local_uniform_directional_shift(radius_shift, start, stop, 0)
    
    def expand(self, yshift: float, start: int, stop: int) -> Self:
        """Locally add uniform shift to the axial (y-axis) direction."""
        return self._add_local_uniform_directional_shift(yshift, start, stop, 1)
        
    def screw(self, angle_shift: float, start: int, stop: int) -> Self:
        """Locally add uniform shift to the skew (a-axis) direction."""
        return self._add_local_uniform_directional_shift(angle_shift, start, stop, 2)
    
    def alleviate(self, label: ArrayLike, niter: int = 1) -> Self:
        from .._cpp_ext import alleviate
        label = np.asarray(label, dtype=np.int32)
        if label.shape[1] != 2:
            raise ValueError("Label shape mismatch")
        mesh = oblique_meshgrid(
            self._shape, self._tilts, self._intervals, self._offsets
        )  # (Ny, Npf, 2)
        radius_arr = np.full(mesh.shape[:2] + (1,), self._radius, dtype=np.float32)
        mesh3d = np.concatenate([radius_arr, mesh], axis=2)  # (Ny, Npf, 3)
        shifted = mesh3d + self._displace
        shifted = alleviate(shifted, label, self.nrise, niter)
        displace = shifted - mesh3d
        return self.replace(displace=displace)
    
    def _add_directional_shift(self, displace: np.ndarray, axis: int) -> Self:
        displace = self._displace.copy()
        displace[:, :, axis] += displace
        return self.replace(displace=displace)
    
    def _add_local_uniform_directional_shift(
        self, shift: float, start: int, stop: int, axis: int
    ) -> Self:
        displace = self._displace.copy()
        for idx in range(start, stop):
            displace[idx:, :, axis] += shift
        return self.replace(displace=displace)
    
    def _get_shifted(self):
        mesh = oblique_meshgrid(
            self._shape, self._tilts, self._intervals, self._offsets
        )  # (Ny, Npf, 2)
        radius_arr = np.full(mesh.shape[:2] + (1,), self._radius, dtype=np.float32)
        mesh3d = np.concatenate([radius_arr, mesh], axis=2)  # (Ny, Npf, 3)
        return mesh3d + self._displace

def oblique_meshgrid(
    shape: tuple[int, int], 
    tilts: tuple[float, float] = (0., 0.),
    intervals: tuple[float, float] = (1., 1.),
    offsets: tuple[float, float] = (0., 0.),
) -> np.ndarray:
    """
    Construct 2-D meshgrid in oblique coordinate system.

    Parameters
    ----------
    shape : tuple[int, int]
        Output shape. If ``shape = (a, b)``, length of the output mesh will be ``a`` along
        the first axis, and will be ``b`` along the second one.
    tilts : tuple[float, float], optional
        Tilt tangents of each axis in world coordinate. Positive tangent means that the 
        corresponding axis tilt toward the line "y=x".
    intervals : tuple[float, float], optional
        The intervals (or scale) of new axes. 
    offsets : tuple[float, float], optional
        The origin of new coordinates.

    Returns
    -------
    np.ndarray
        World coordinates of lattice points of new coordinates.
    """
    tan0, tan1 = tilts
    d0, d1 = intervals
    c0, c1 = offsets
    n0, n1 = shape
    
    v0 = np.array([1, tan0], dtype=np.float32)
    v1 = np.array([tan1, 1], dtype=np.float32)

    out = np.empty((n0, n1, 2), dtype=np.float32)
    
    for i in range(n0):
        for j in range(n1):
            out[i, j, :] = v0 * i + v1 * j
    
    out[:, :, 0] = out[:, :, 0] * d0 + c0
    out[:, :, 1] = out[:, :, 1] * d1 + c1
    return out