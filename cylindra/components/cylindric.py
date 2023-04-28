from __future__ import annotations
from typing import Any, TYPE_CHECKING, NamedTuple

from acryo import Molecules
import numpy as np
from numpy.typing import ArrayLike, NDArray
import polars as pl
from .spline import Spline
from cylindra.const import MoleculesHeader as Mole

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
        tilts: tuple[float, float] = (0.0, 0.0),
        interval: float = 1.0,
        radius: float = 1.0,
        offsets: tuple[float, float] = (0.0, 0.0),
        displace: NDArray[np.floating] | None = None,
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
            self._displace = np.asarray(displace, dtype=np.float32)

    def with_nrise(self, nrise: int) -> Self:
        """Return an updated model with given rise number."""
        tilts_lat = nrise / self.shape[1]
        return self.replace(tilts=(self.tilts[0], tilts_lat))

    def replace(
        self,
        tilts: tuple[float, float] | None = None,
        interval: float | None = None,
        radius: float | None = None,
        offsets: tuple[float, float] | None = None,
        displace: NDArray[np.floating] | None = None,
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
        """Shape of the model in (axial, angular) orientation."""
        return self._shape

    @property
    def radius(self) -> float:
        """Radius of the model."""
        return self._radius

    @property
    def tilts(self) -> tuple[float, float]:
        """Normalized tilt of axial-to-angular direction and angular-to-axial one."""
        return self._tilts

    @property
    def intervals(self) -> tuple[float, float]:
        """Canonical interval between each molecules."""
        return self._intervals

    @property
    def offsets(self) -> tuple[float, float]:
        """Offset of the first molecule at the origin.."""
        return self._offsets

    @property
    def displace(self) -> NDArray[np.float32]:
        """Displacement vector of the molecules."""
        return self._displace

    @property
    def nrise(self) -> int:
        """
        Rise number of the model.

        Molecule at (Y, self.shape[1]) is same as (Y - nrise, 0).
        """
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
        """
        Generate molecules from the coordinates and given spline.

        Generated molecules will have following features.
        - "molecules-pf" ... The index of the molecule in the angular direction
          (Protofilament number).
        - "molecules-position" ... The position of the molecule in the axial direction.
          in nm. If the spline starts from the tip, position=0 is the tip.
        """
        shifted = self._get_shifted()
        shifted_2d = shifted.reshape(-1, 3)
        mole = spl.cylindrical_to_molecules(shifted_2d)
        if spl.inverted:
            pos = pl.Series(shifted_2d[::-1, 1])
        else:
            pos = pl.Series(shifted_2d[:, 1])
        pf = pl.Series(np.arange(len(mole), dtype=np.uint32)) % self._shape[1]
        mole.features = {Mole.pf: pf, Mole.position: pos}
        return mole

    def to_mesh(self, spl: Spline):
        nodes = spl.cylindrical_to_world(
            self.replace(tilts=(0, 0))._get_mesh().reshape(-1, 3)
        )
        vertices: list[tuple[int, int, int]] = []
        ny, npf = self.shape
        for y in range(ny):
            for a in range(npf):
                idx = y * npf + a
                if y > 0:
                    (idx, idx - npf, idx + 1)
                    if a > 0:
                        vertices.append((idx, idx - 1, idx - npf))
                    else:
                        vertices.append((idx, idx + npf - 1, idx - npf))
                if y < ny - 1:
                    (idx, idx + npf, idx - 1)
                    if a < npf - 1:
                        vertices.append((idx, idx + 1, idx + npf))
                    else:
                        vertices.append((idx, idx - npf + 1, idx + npf))
        return nodes, np.array(vertices)

    def add_offsets(self, offsets: tuple[float, float]) -> Self:
        """Increment offsets attribute of the model."""
        _offsets = tuple(x + y for x, y in zip(self._offsets, offsets))
        return self.replace(offsets=_offsets)

    def add_shift(self, shift: NDArray[np.floating]) -> Self:
        """Increment displace attribute of the model."""
        displace = self._displace + shift
        return self.replace(displace=displace)

    def add_radial_shift(self, shift: NDArray[np.floating]) -> Self:
        """Add shift to the radial (r-axis) direction."""
        return self._add_directional_shift(shift, 0)

    def add_axial_shift(self, shift: NDArray[np.floating]) -> Self:
        """Add shift to the axial (y-axis) direction."""
        return self._add_directional_shift(shift, 1)

    def add_skew_shift(self, shift: NDArray[np.floating]) -> Self:
        """Add shift to the skew (a-axis) direction."""
        return self._add_directional_shift(shift, 2)

    def dilate(
        self, radius_shift: float, sl: slice | tuple[slice, slice] | CylindricSlice
    ) -> Self:
        """Locally add uniform shift to the radial (r-axis) direction."""
        shift = np.zeros(self.shape, dtype=np.float32)
        if not isinstance(sl, CylindricSlice):
            sl = indexer[sl]
        sl.get_resolver(self.nrise).set_slice(shift, radius_shift)
        shift3d = np.stack([shift, np.zeros_like(shift), np.zeros_like(shift)], axis=2)
        return self.add_shift(shift3d)

    def expand(
        self, yshift: float, sl: slice | tuple[slice, slice] | CylindricSlice
    ) -> Self:
        """Locally add uniform shift to the axial (y-axis) direction."""
        if not isinstance(sl, CylindricSlice):
            sl = indexer[sl]

        displace = self._displace.copy()
        axis = 1
        for _y, _a in sl.resolve(self.shape, self.nrise):
            for start in range(_y.start, _y.stop):
                displace[start:, _a, axis] += yshift
        return self.replace(displace=displace)

    def screw(
        self, angle_shift: float, sl: slice | tuple[slice, slice] | CylindricSlice
    ) -> Self:
        """Locally add uniform shift to the skew (a-axis) direction."""
        if not isinstance(sl, CylindricSlice):
            sl = indexer[sl]

        displace = self._displace.copy()
        axis = 2
        for _y, _a in sl.resolve(self.shape, self.nrise):
            for start in range(_y.start, _y.stop):
                displace[start:, _a, axis] += angle_shift
        return self.replace(displace=displace)

    def alleviate(self, label: ArrayLike, niter: int = 1) -> Self:
        """
        Alleviate displacements by iterative local-averaging algorithm.

        This method should be called after e.g. `add_axial_shift`. Molecules adjacent to
        the shifted molecules will be shifted to match the center of the surrounding
        molecules.

        Parameters
        ----------
        label : array-like
            Label that specify the molecules to be fixed. Shape of this argument can be
            eigher (N, 2) or same as the shape of the model. In the former case, it is
            interpreted as N indices. In the latter case, True indices will be considered
            as the indices.
        niter : int, default is 1
            Number of iteration.

        Returns
        -------
        CylinderModel
            New model with updated parameters.
        """
        from cylindra._cylindra_ext import alleviate

        label = np.asarray(label, dtype=np.int32)
        if label.shape[1] != 2:
            if label.shape == self.shape:
                label = np.stack(np.where(label), axis=1)
            else:
                raise ValueError("Label shape mismatch")
        mesh = self._get_mesh()
        shifted = mesh + self._displace
        shifted = alleviate(shifted, label, self.nrise, niter)
        displace = shifted - mesh
        return self.replace(displace=displace)

    def _add_directional_shift(self, displace: NDArray[np.floating], axis: int) -> Self:
        _displace = self._displace.copy()
        _displace[:, :, axis] += displace
        return self.replace(displace=_displace)

    def _get_shifted(self) -> NDArray[np.float32]:
        """Get coordinate mesh with displacements applied."""
        mesh = self._get_mesh()
        return mesh + self._displace

    def _get_mesh(self) -> NDArray[np.float32]:
        """Get canonical coordinate mesh."""
        mesh2d = oblique_meshgrid(
            self._shape, self._tilts, self._intervals, self._offsets
        )  # (Ny, Npf, 2)
        radius_arr = np.full(mesh2d.shape[:2] + (1,), self._radius, dtype=np.float32)
        mesh3d = np.concatenate([radius_arr, mesh2d], axis=2)  # (Ny, Npf, 3)
        return mesh3d


def oblique_meshgrid(
    shape: tuple[int, int],
    tilts: tuple[float, float] = (0.0, 0.0),
    intervals: tuple[float, float] = (1.0, 1.0),
    offsets: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.floating]:
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


class CylindricSlice(NamedTuple):
    y: slice
    a: slice

    def __repr__(self) -> str:
        _y, _a = self
        if _y == slice(None):
            ysl = ":"
        else:
            ysl = f"{_y.start}:{_y.stop}"
        if _a == slice(None):
            asl = ":"
        else:
            asl = f"{_a.start}:{_a.stop}"
        return f"Idx[{ysl}, {asl}]"

    def get_resolver(self, rise: int) -> CylindricSliceResolver:
        return CylindricSliceResolver(*self, rise)

    def resolve(self, shape: tuple[int, int], rise: int):
        return self.get_resolver(rise).resolve_slices(shape)


class CylindricSliceConstructor:
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return CylindricSlice(*key)
        return CylindricSlice(key, slice(None))


indexer = CylindricSliceConstructor()


class CylindricSliceResolver(NamedTuple):
    y: slice
    a: slice
    rise: int

    def resolve_slices(self, shape: tuple[int, int]) -> list[CylindricSlice]:
        ny, na = shape
        _y, _a, rise = self
        astart = _a.start
        astop = _a.stop
        if astart is None:
            astart = 0
        if astop is None:
            astop = na
        if astart >= astop:
            raise ValueError("start must be larger than stop.")

        slices: list[CylindricSlice] = []
        npart_start, res_start = divmod(astart, na)
        npart_stop, res_stop = divmod(astop, na)

        i = npart_start
        s0 = res_start
        while i <= npart_stop:
            yoffset = i * rise
            s1 = na if i < npart_stop else res_stop
            slices.append(
                CylindricSlice(
                    slice(_y.start + yoffset, _y.stop + yoffset), slice(s0, s1)
                )
            )

            i += 1
            s0 = 0

        return slices

    def get_slice(self, arr: np.ndarray) -> np.ndarray:
        slices = self.resolve_slices(arr.shape)
        return np.concatenate([arr[sl] for sl in slices], axis=1)

    def set_slice(self, arr: np.ndarray, val: Any) -> None:
        slices = self.resolve_slices(arr.shape)
        start = 0
        if isinstance(val, np.ndarray):
            for sl in slices:
                asl = sl[1]
                size = asl.stop - asl.start
                stop = start + size
                arr[sl] = val[:, start:stop]
                start = stop
        elif np.isscalar(val):
            for sl in slices:
                asl = sl[1]
                size = asl.stop - asl.start
                stop = start + size
                arr[sl] = val
                start = stop
        else:
            raise TypeError(f"Cannot set {val!r}.")
        return None
