from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import polars as pl
from acryo import Molecules
from numpy.typing import ArrayLike, NDArray

from cylindra import utils
from cylindra._cylindra_ext import cylinder_faces
from cylindra.const import MoleculesHeader as Mole

if TYPE_CHECKING:
    from typing_extensions import Self

    from .spline import Spline


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

    def resolve(self, shape: tuple[int, int], rise: int) -> list[CylindricSlice]:
        """
        Resolve borders of cylindric slice.

        Parameters
        ----------
        shape : (int, int)
            Shape of the cylinder.
        rise : int
            Rise number of the cylinder.
        """
        return self.get_resolver(rise).resolve_slices(shape)


_Slicer = slice | tuple[slice, slice] | CylindricSlice


class CylinderModel:
    """
    A model class that describes a heterogenic cylindrical structure.

    Parameters
    ----------
    shape : (int, int)
        Shape of the molecule grid in (axial, angular) shape.
    tilts : (float, float)
        Relative tilt tangent in (axial, angular) direction.
    intervals : (float, float)
        Interval in (nm, radian).
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
        intervals: tuple[float, float] = (1.0, 1.0),
        radius: float = 1.0,
        offsets: tuple[float, float] = (0.0, 0.0),
        displace: NDArray[np.floating] | None = None,
    ):
        self._shape = shape
        self._tilts = tilts
        if intervals[0] <= 0 or intervals[1] <= 0:
            raise ValueError("Intervals must be positive")
        self._intervals = intervals
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
        intervals: tuple[float, float] | None = None,
        radius: float | None = None,
        offsets: tuple[float, float] | None = None,
        displace: NDArray[np.floating] | None = None,
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
        strs = list[str]()
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

    def to_molecules(
        self, spl: Spline, features: pl.DataFrame | None = None
    ) -> Molecules:
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
        pos = pl.Series(shifted_2d[:, 1])
        arange = pl.Series(np.arange(len(mole), dtype=np.int32))
        nth = arange // self._shape[1]
        pf = arange % self._shape[1]
        mole.features = pl.DataFrame({Mole.nth: nth, Mole.pf: pf, Mole.position: pos})
        if features is not None:
            for col in [Mole.nth, Mole.pf, Mole.position]:
                if col in features.columns:
                    features.drop(col)
            mole.features = mole.features.with_columns(features)
        return mole

    def locate_molecules(self, spl: Spline, coords: NDArray[np.int32]) -> Molecules:
        """Locate molecules at given integer coordinates around the spline."""
        mesh = self._get_mesh(coords)
        mole = spl.cylindrical_to_molecules(mesh)
        pos = pl.Series(mesh[:, 1])
        nth = coords[:, 0]
        pf = coords[:, 1]
        mole.features = {Mole.nth: nth, Mole.pf: pf, Mole.position: pos}
        return mole

    def to_mesh(
        self,
        spl: Spline,
        shape: tuple[int, int] | None = None,
        value_by: str | None = None,
        order: int = 0,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]]:
        """
        Create a mesh data for cylinder visualization.

        Returned mesh is a tuple of (nodes, vertices). Nodes is a (N, 3) array.
        """
        if shape is None:
            shape = self.shape
        ycoords = np.linspace(0, self.shape[0] - 1, shape[0])
        acoords = np.linspace(0, self.shape[1] - 1, shape[1])
        yy, aa = np.meshgrid(ycoords, acoords, indexing="ij")
        coords = np.stack([yy.ravel(), aa.ravel()], axis=1, dtype=np.float32)
        mesh2d = self.replace(tilts=(0, 0))._get_mesh(coords)
        verts = spl.cylindrical_to_world(mesh2d.reshape(-1, 3))
        faces = cylinder_faces(*shape)
        if value_by is not None:
            loc = spl.props.get_loc(value_by)
            anc = spl.anchors
            xinterp = np.repeat(np.linspace(0, 1, shape[0]), shape[1]).clip(
                anc[0], anc[-1]
            )
            values = utils.interp(anc, loc, order=order)(xinterp)
        else:
            values = np.full(verts.shape[0], 0.5, dtype=np.float32)
        return verts, faces, values

    def add_shift(self, shift: NDArray[np.floating]) -> Self:
        """Increment displace attribute of the model."""
        displace = self._displace + shift
        return self.replace(displace=displace)

    def dilate(self, by: float, sl: _Slicer) -> Self:
        """Locally add uniform shift to the radial (r-axis) direction."""
        shift = np.zeros(self.shape, dtype=np.float32)
        if not isinstance(sl, CylindricSlice):
            sl = indexer[sl]
        sl.get_resolver(self.nrise).set_slice(shift, by)
        shift3d = np.stack([shift, np.zeros_like(shift), np.zeros_like(shift)], axis=2)
        return self.add_shift(shift3d)

    def expand(self, by: float, sl: _Slicer) -> Self:
        """
        Locally add uniform displacement to the axial (y-axis) direction.

                              o o
            o o o o         o o o o
            o o o o         o     o
            o o o o  ---->  o o o o
            o o o o         o     o
            o o o o         o o o o
                              o o
        """
        return self._in_plane_displace(by, sl, axis=1)

    def twist(self, by: float, sl: _Slicer) -> Self:
        """
        Locally add uniform displacement to the skew (a-axis) direction.

            o o o o         o  o oo
            o o o o         o  o oo
            o o o o  ---->  o o o o
            o o o o         oo o  o
            o o o o         oo o  o
        """
        return self._in_plane_displace(by, sl, axis=2)

    def _in_plane_displace(self, by: float, sl: _Slicer, axis: int):
        if not isinstance(sl, CylindricSlice):
            sl = indexer[sl]

        # check out-of-bound
        if (
            (sl.y.start is not None and sl.y.start < 0)
            or (sl.y.stop is not None and sl.y.stop > self.shape[0])
            or (sl.a.start is not None and sl.a.start < 0)
            or (sl.a.stop is not None and sl.a.stop > self.shape[1])
        ):
            raise ValueError(f"{sl} is out of bound of cylinder of shape {self.shape}.")
        displace = self._displace.copy()
        for _y, _a in sl.resolve(self.shape, self.nrise):
            _ny = _y.stop - _y.start
            if _ny == 0:
                continue
            _ar = by * np.arange(1, _ny + 1)
            displace[:, _a, axis] -= _ny / 2 * by
            displace[_y.start : _y.stop, _a, axis] += _ar[:, np.newaxis]
            displace[_y.stop :, _a, axis] += by * _ny
        return self.replace(displace=displace)

    def alleviate(self, label: ArrayLike) -> Self:
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
        mesh = self._get_regular_mesh()
        shifted = mesh + self._displace
        shifted = alleviate(shifted, label, self.nrise)
        displace = shifted - mesh
        return self.replace(displace=displace)

    def _add_directional_shift(self, displace: NDArray[np.floating], axis: int) -> Self:
        _displace = self._displace.copy()
        _displace[:, :, axis] += displace
        return self.replace(displace=_displace)

    def _get_shifted(self) -> NDArray[np.float32]:
        """Get coordinate mesh with displacements applied."""
        mesh = self._get_regular_mesh()
        return mesh + self._displace

    def _get_regular_mesh(self) -> NDArray[np.float32]:
        yy, aa = np.indices(self._shape, dtype=np.int32)
        coords = np.stack([yy.ravel(), aa.ravel()], axis=1)
        mesh2d = self._get_mesh(coords)
        return mesh2d.reshape(self._shape + (3,))

    def _get_mesh(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert indices to oblique coordinates in (r, y, a)."""
        from cylindra._cylindra_ext import oblique_coordinates

        y_incr = -self._tilts[1] * self._shape[1] * self._intervals[0] / 2
        yoffset = y_incr + self._offsets[0]
        aoffset = self._offsets[1]
        mesh2d = oblique_coordinates(
            coords.astype(np.float32, copy=False),
            self._tilts,
            self._intervals,
            (yoffset, aoffset),
        )
        r_arr = np.full(mesh2d.shape[:1] + (1,), self._radius, dtype=np.float32)
        return np.concatenate([r_arr, mesh2d], axis=1)


class CylindricSliceConstructor:
    """The indexer class for a cylindric boundary."""

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

        slices = list[CylindricSlice]()
        npart_start, res_start = divmod(astart, na)
        npart_stop, res_stop = divmod(astop, na)
        if res_stop == 0:
            npart_stop -= 1
            res_stop = na

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
