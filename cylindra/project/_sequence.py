from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    MutableSequence,
    NamedTuple,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)

import polars as pl

from cylindra._config import get_config
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.const import cast_dataframe
from cylindra.cylmeasure import LatticeParameters, calc_localvec_lat, calc_localvec_long
from cylindra.project._single import CylindraProject

if TYPE_CHECKING:
    import numpy as np
    from acryo import BatchLoader, Molecules
    from numpy.typing import NDArray
    from typing_extensions import Self

    from cylindra.components import CylSpline

_V = TypeVar("_V")
_Null = object()
_IDTYPE = Literal["int", "path"]


class Validator(ABC, Generic[_V]):
    def __init__(self, check: bool = True):
        self._value = _Null
        self._check = check

    @property
    def value(self) -> _V:
        if self._value is _Null:
            raise AttributeError("Value cannot be determined yet.")
        return self._value

    @value.setter
    def value(self, val: _V):
        if self._value is _Null:
            self._value = val
        else:
            if self._check:
                val = self.check_value(val)
            self._value = val

    def initialize(self):
        self._value = _Null

    @abstractmethod
    def check_value(self, val: Any) -> _V:
        """Assert input has the same value. Raise an error otherwise."""


class ScaleValidator(Validator[float]):
    def check_value(self, val: Any) -> float:
        val = float(val)
        if 1 - val / self.value > 0.01:
            raise ValueError(f"Existing scale is {self.value}, tried to set {val}.")
        return val


class ProjectSequence(MutableSequence[CylindraProject]):
    """
    Collection of Cylindra projects.

    This object is just for project management. BatchLoader, DataFrame and local/global
    properties can be generated from this object.
    """

    def __init__(self, *, check_scale: bool = True):
        self._projects = list[CylindraProject]()
        self._scale_validator = ScaleValidator(check_scale)

    def __repr__(self) -> str:
        if len(self) > 1:
            return (
                f"{type(self).__name__} with {len(self)} projects such as {self[0]!r}"
            )
        return f"{type(self).__name__} (empty)"

    @overload
    def __getitem__(self, key: int) -> CylindraProject:
        ...

    @overload
    def __getitem__(self, key: slice) -> ProjectSequence:
        ...

    def __getitem__(self, key: int):
        out = self._projects[key]
        if isinstance(key, slice):
            out = ProjectSequence(check_scale=self._scale_validator._check)
            out._projects = self._projects[key]
        return out

    def __setitem__(self, key: int, value: CylindraProject) -> None:
        if not isinstance(value, CylindraProject):
            raise TypeError(f"Expected CylindraProject, got {type(value)}.")
        if not isinstance(key, SupportsIndex):
            raise TypeError(f"Expected int, got {type(key)}.")
        self._projects[key] = value

    def __delitem__(self, key: int) -> None:
        del self._projects[key]
        if len(self) == 0:
            self._scale_validator.initialize()

    def __len__(self) -> int:
        return len(self._projects)

    def __iter__(self) -> Iterator[CylindraProject]:
        return iter(self._projects)

    def insert(self, index: int, value: CylindraProject) -> None:
        """Insert a project at the given index."""
        if not isinstance(value, CylindraProject):
            raise TypeError(f"Expected CylindraProject, got {type(value)}.")
        return self._projects.insert(index, value)

    def __add__(self, other: ProjectSequence) -> ProjectSequence:
        """Concatenate two ProjectSequence objects."""
        if not isinstance(other, ProjectSequence):
            raise TypeError(f"Expected ProjectSequence, got {type(other)}.")
        new = ProjectSequence(check_scale=True)
        new._projects = self._projects + other._projects
        if len(self) > 0:
            new._scale_validator.value = self._scale_validator.value
        if len(other) > 0:
            new._scale_validator.value = other._scale_validator.value
        return new

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        check_scale: bool = True,
        skip_exc: bool = False,
    ) -> Self:
        """Add all the projects of the given paths."""
        self = cls(check_scale=check_scale)
        if skip_exc:
            for path in paths:
                with suppress(Exception):
                    self.append_file(path)
        else:
            for path in paths:
                self.append_file(path)
        return self

    def append_file(self, path: str | Path) -> Self:
        """Add a project from a file path."""
        prj = CylindraProject.from_file(path)
        self._scale_validator.value = prj.scale
        self._projects.append(prj)
        return self

    def sta_loader(
        self,
        name_filter: Callable[[str], bool] | None = None,
        *,
        curvature: bool = False,
        allow_no_image: bool = False,
    ) -> BatchLoader:
        """
        Construct a STA loader from all the projects.

        Parameters
        ----------
        name_filter : callable, default None
            Function that takes a molecule file name (without extension) and
            returns True if the molecule should be collected. Collect all the
            molecules by default.
        curvature : bool, default False
            If True, the spline curvature will be added to the molecule features.
        allow_no_image : bool, default False
            If True, this method will not raise an error when the image file is not
            found.
        """
        import impy as ip
        from acryo import BatchLoader

        col = BatchLoader(scale=self._scale_validator.value)
        if name_filter is None:

            def name_filter(_):
                return True

        for idx, prj in enumerate(self._projects):
            if prj.image is None or not prj.image.exists():
                if not allow_no_image:
                    raise ValueError(
                        f"Image file not found in project at {prj.project_path}."
                    )
                import numpy as np

                img = np.zeros((0, 0, 0), dtype=np.float32)  # dummy
            else:
                img = ip.lazy.imread(prj.image, chunks=get_config().dask_chunk).value
            with prj.open_project() as dir:
                for info, mole in prj.iter_load_molecules(dir):
                    if not name_filter(info.stem):
                        continue
                    if (
                        curvature
                        and (_spl_i := info.source) is not None
                        and Mole.position in mole.features.columns
                    ):
                        _spl = prj.load_spline(_spl_i, dir=dir, props=False)
                        _u = _spl.y_to_position(mole.features[Mole.position])
                        cv = _spl.curvature(_u)
                        mole.features = mole.features.with_columns(
                            pl.Series(cv, dtype=pl.Float32).alias("spline_curvature")
                        )
                    mole.features = mole.features.with_columns(
                        pl.repeat(info.stem, pl.len()).alias(Mole.id)
                    )
                    col.add_tomogram(img, molecules=mole, image_id=idx)
        return col

    def collect_localprops(
        self,
        allow_none: bool = True,
        id: _IDTYPE = "int",
        spline_details: bool = False,
    ) -> pl.DataFrame:
        """
        Collect all localprops into a single dataframe.

        Parameters
        ----------
        allow_none : bool, default True
            Continue data collection even if property table data file was not
            found in any project. Raise error otherwise.
        id : str, default "int"
            How to describe the source tomogram. If "int", each tomogram will
            be labeled with ascending integers. If "path", each tomogram will
            be labeled with the name of the project directory.
        spline_details : bool, default False
            If True, spline coordinates, its derivatives and the curvature
            will also be collected as well. This will take more memory and time.

        Returns
        -------
        pl.DataFrame
            Dataframe with all the properties.
        """
        dfs_prj = list[pl.DataFrame]()  # localprops of each project
        for idx, prj in enumerate(self._projects):
            with prj.open_project() as dir:
                if not prj._localprops_path(dir).exists():
                    if not allow_none:
                        raise ValueError(
                            f"Localprops not found in project at {prj.project_path}."
                        )
                    continue
                dfs_spl = list[pl.DataFrame]()
                for spl in prj.iter_load_splines(dir, drop_columns=False):
                    _df_spl = spl.props.loc
                    if spline_details:
                        if not spl.has_anchors:
                            raise ValueError(
                                f"Cannot collect spline details because spline {spl!r} "
                                "does not have anchors."
                            )
                        _crds = [spl.map(der=der) for der in [0, 1, 2]]
                        _cv = spl.curvature()
                        _df_spl = _df_spl.with_columns(
                            pl.Series("spline_z", _crds[0][:, 0], dtype=pl.Float32),
                            pl.Series("spline_y", _crds[0][:, 1], dtype=pl.Float32),
                            pl.Series("spline_x", _crds[0][:, 2], dtype=pl.Float32),
                            pl.Series("spline_dz", _crds[1][:, 0], dtype=pl.Float32),
                            pl.Series("spline_dy", _crds[1][:, 1], dtype=pl.Float32),
                            pl.Series("spline_dx", _crds[1][:, 2], dtype=pl.Float32),
                            pl.Series("spline_ddz", _crds[2][:, 0], dtype=pl.Float32),
                            pl.Series("spline_ddy", _crds[2][:, 1], dtype=pl.Float32),
                            pl.Series("spline_ddx", _crds[2][:, 2], dtype=pl.Float32),
                            pl.Series("spline_curvature", _cv, dtype=pl.Float32),
                        )
                    dfs_spl.append(_df_spl)
                _df_prj = pl.concat(dfs_spl, how="diagonal")
                columns = [pl.repeat(idx, pl.len()).cast(pl.UInt16).alias(Mole.image)]
                if H.spline_id in _df_prj.columns:
                    columns.append(pl.col(H.spline_id).cast(pl.UInt16))

            dfs_prj.append(_df_prj.with_columns(columns))
        out = cast_dataframe(pl.concat(dfs_prj, how="diagonal"))
        return self._normalize_id(out, id)

    def collect_globalprops(
        self,
        allow_none: bool = True,
        suffix: str = "",
        id: _IDTYPE = "int",
    ) -> pl.DataFrame:
        """
        Collect all globalprops into a single dataframe.

        Parameters
        ----------
        allow_none : bool, default True
            Continue data collection even if property table data file was not
            found in any project. Raise error otherwise.
        suffix : str, default ""
            Suffix to add to the column names that may be collide with the local
            properties.
        id : str, default "int"
            How to describe the source tomogram. If "int", each tomogram will
            be labeled with ascending integers. If "path", each tomogram will
            be labeled with the name of the project directory.

        Returns
        -------
        pl.DataFrame
            Dataframe with all the properties.
        """
        dataframes = list[pl.DataFrame]()
        for idx, prj in enumerate(self._projects):
            with prj.open_project() as dir:
                path = prj._globalprops_path(dir)
                if path is None:
                    if not allow_none:
                        raise ValueError(
                            f"Globalprops not found in project at {prj.project_path}."
                        )
                    continue
                imagespec = pl.Series(Mole.image, [idx]).cast(pl.UInt16)
                df = pl.read_csv(path).with_columns(imagespec)
            dataframes.append(df)
        out = cast_dataframe(pl.concat(dataframes, how="diagonal"))
        if suffix:
            need_rename = [
                H.spacing, H.twist, H.npf, H.rise, H.skew,
                H.rise_length, H.radius, H.start,
            ]  # fmt: skip
            nmap = {col: col + suffix for col in need_rename if col in out.columns}
            out = out.rename(nmap)
        return self._normalize_id(out, id)

    def collect_joinedprops(
        self,
        allow_none: bool = True,
        id: _IDTYPE = "int",
        spline_details: bool = False,
    ) -> pl.DataFrame:
        """
        Collect all the local and global properties into a single dataframe.

        The global properties are suffixed with "_glob". Note that these columns
        will repeat the same values for each spline. For instance, the "spacing"
        columns will look like following.

        >>> col.collect_joinedprops().select(["spacing", "spacing_glob"])

            shape: (12, 2)
            ┌───────────┬──────────────┐
            │ spacing   ┆ spacing_glob │
            │ ---       ┆ ---          │
            │ f32       ┆ f32          │
            ╞═══════════╪══════════════╡
            │ 4.093385  ┆ 4.1024575    │
            │ 4.0987015 ┆ 4.1024575    │
            │ 4.1013646 ┆ 4.1024575    │
            │ …         ┆ …            │
            │ 4.074887  ┆ 4.089436     │
            │ 4.0987015 ┆ 4.089436     │
            └───────────┴──────────────┘

        Parameters
        ----------
        allow_none : bool, default True
            Forwarded to `collect_localprops` and `collect_globalprops`.
        id : str, default "int"
            How to describe the source tomogram. If "int", each tomogram will
            be labeled with ascending integers. If "path", each tomogram will
            be labeled with the name of the project directory.
        spline_details : bool, default False
            Forwarded to `collect_localprops`.

        Returns
        -------
        pl.DataFrame
            Dataframe with all the properties.
        """
        props = self.collect_props(
            allow_none=allow_none, spline_details=spline_details, suffix="_glob"
        )
        on = [H.spline_id, Mole.image]
        out = props.loc.join(props.glob, on=on, suffix="_glob")
        return self._normalize_id(out, id)

    def collect_props(
        self,
        allow_none: bool = True,
        spline_details: bool = False,
        suffix="",
    ) -> CollectedProps:
        """
        Collect all the local and global properties.

        Parameters
        ----------
        allow_none : bool, default True
            Forwarded to `collect_localprops` and `collect_globalprops`.
        spline_details : bool, default False
            Forwarded to `collect_localprops`.

        Returns
        -------
        CollectedProps
            Tuple of the collected local and global properties.
        """
        loc = self.collect_localprops(
            allow_none=allow_none, id="int", spline_details=spline_details
        )
        glb = self.collect_globalprops(allow_none=allow_none, id="int", suffix=suffix)
        if spline_details:
            lengths = list[float]()
            for _, spl in self.iter_splines():
                lengths.append(spl.length())
            col = pl.Series("spline_length", lengths, dtype=pl.Float32)
            glb = glb.with_columns(col)
        return CollectedProps(loc, glb)

    def collect_molecules(
        self,
        name_filter: Callable[[str], bool] | None = None,
        *,
        curvature: bool = False,
    ) -> Molecules:
        """
        Collect all the molecules in this project sequence.

        Parameters
        ----------
        name_filter : callable, optional
            Function that takes a molecule file name (without extension) and
            returns True if the molecule should be collected. Collect all the
            molecules by default.
        curvature : bool, default False
            If True, the spline curvature will be added to the molecule features.
        """
        loader = self.sta_loader(name_filter, curvature=curvature, allow_no_image=True)
        return loader.molecules

    def iter_splines(self) -> Iterable[tuple[SplineKey, CylSpline]]:
        """Iterate over all the splines in all the projects."""
        for i_prj, prj in enumerate(self._projects):
            with prj.open_project() as dir:
                for i_spl, spl in enumerate(prj.iter_load_splines(dir)):
                    yield SplineKey(i_prj, i_spl), spl

    def iter_molecules(
        self,
        name_filter: Callable[[str], bool] | None = None,
    ) -> Iterable[tuple[MoleculesKey, Molecules]]:
        """
        Iterate over all the molecules in all the projects.

        Parameters
        ----------
        name_filter : callable, optional
            Function that takes a molecule file name (without extension) and
            returns True if the molecule should be collected. Collect all the
            molecules by default.
        """
        for sl, (mole, _) in self.iter_molecules_with_splines(
            name_filter, skip_no_spline=False
        ):
            yield sl, mole

    def iter_molecules_with_splines(
        self,
        name_filter: Callable[[str], bool] | None = None,
        *,
        skip_no_spline: bool = True,
    ) -> Iterator[MoleculesItem]:
        """
        Iterate over all the molecules and its source spline.

        Parameters
        ----------
        name_filter : callable, optional
            Function that takes a molecule file name (without extension) and
            returns True if the molecule should be collected. Collect all the
            molecules by default.
        skip_no_spline : bool, default True
            If True, molecules without a source spline will be skipped.
        """
        if name_filter is None:

            def name_filter(_):
                return True

        for i_prj, prj in enumerate(self._projects):
            with prj.open_project() as dir_:
                for info, mole in prj.iter_load_molecules():
                    if not name_filter(info.name):
                        continue
                    if (src := info.source) is None and skip_no_spline:
                        continue
                    spl = prj.load_spline(src, dir=dir_)
                    yield MoleculesItem(MoleculesKey(i_prj, info.stem), (mole, spl))

    def collect_spline_coords(self, ders: int | Iterable[int] = 0) -> pl.DataFrame:
        """
        Collect spline coordinates or its derivative(s) as a dataframe.

        Coordinates will be labeled as "z", "y", "x". The 1st derivative will be
        labeled as "dz", "dy", "dx", and so on.

        Parameters
        ----------
        ders : int or iterable of int, default 0
            Derivative order(s) to collect. If multiple values are given, all the
            derivatives will be concatenated in a single dataframe.
        """
        dfs = list[pl.DataFrame]()
        if not hasattr(ders, "__iter__"):
            ders = [ders]
        for (i, j), spl in self.iter_splines():
            nanc = spl.anchors.size
            df = pl.DataFrame(
                [
                    pl.repeat(i, nanc, eager=True, dtype=pl.UInt16).alias(Mole.image),
                    pl.repeat(j, nanc, eager=True, dtype=pl.UInt16).alias(H.spline_id),
                ]
            )
            for der in ders:
                d = "d" * der
                coords = spl.map(der=der)
                df = df.with_columns(
                    pl.Series(coords[:, 0], dtype=pl.Float32).alias(f"{d}z"),
                    pl.Series(coords[:, 1], dtype=pl.Float32).alias(f"{d}y"),
                    pl.Series(coords[:, 2], dtype=pl.Float32).alias(f"{d}x"),
                )
            dfs.append(df)
        return pl.concat(dfs, how="vertical")

    def _normalize_id(self, out: pl.DataFrame, id: _IDTYPE) -> pl.DataFrame:
        match id:
            case "int":
                pass
            case "path":
                _map = dict[int, str]()
                _appeared = set[str]()
                for i, prj in enumerate(self._projects):
                    path = prj.project_path
                    if path is None:
                        raise ValueError(
                            f"The {i}-th project {prj!r} does not have a path."
                        )
                    label = _make_unique_label(Path(path).parent.name, _appeared)
                    _map[i] = label
                    _appeared.add(label)
                _image_col = pl.col(Mole.image)
                if hasattr(_image_col, "replace_strict"):  # polars>=1.0.0
                    _image_col = _image_col.replace_strict(
                        _map, return_dtype=pl.Enum(list(_map.values()))
                    )
                else:
                    _image_col = _image_col.replace(
                        _map, return_dtype=pl.Enum(list(_map.values()))
                    )
                out = out.with_columns(_image_col)
            case _:
                raise ValueError(f"Invalid id type {id!r}.")
        return out


def _make_unique_label(label: str, appeared: set[str]) -> str:
    if label not in appeared:
        return label
    i = 0
    while True:
        new_label = f"{label}_{i}"
        if new_label not in appeared:
            return new_label
        i += 1


class MoleculesKey(NamedTuple):
    """Tuple of the project ID and the name of a molecules object."""

    project_id: int
    """Index of the project in the project sequence."""
    name: str
    """Name of the molecules object."""


class SplineKey(NamedTuple):
    """Tuple of the project ID and the spline ID."""

    project_id: int
    """Index of the project in the project sequence."""
    spline_id: int
    """Index of the spline in the project."""


class MoleculesItem(NamedTuple):
    key: MoleculesKey
    value: tuple[Molecules, CylSpline | None]

    def __repr__(self) -> str:
        return (
            f"MoleculesItem(key={self.key!r}, molecules={self.molecules!r}, "
            f"spline={self.spline!r})"
        )

    def _repr_pretty_(self, p, cycle):
        p.text(
            f"MoleculesItem(\n\tkey={self.key!r},\n\tmolecules={self.molecules!r},"
            f"\n\tspline={self.spline!r}\n)"
        )

    @property
    def molecules(self) -> Molecules:
        """The molecules object"""
        return self.value[0]

    @property
    def spline(self) -> CylSpline | None:
        """The source spline if exists."""
        return self.value[1]

    def lattice_structure(self, props: Sequence[str] = ("spacing",)) -> pl.DataFrame:
        return pl.DataFrame(
            [LatticeParameters(p).calculate(*self.value) for p in props]
        )

    def local_vectors_longitudinal(
        self, fill_value: float = 0.0
    ) -> NDArray[np.float32]:
        """Return the local vectors in the longitudinal direction."""
        df = calc_localvec_long(self.molecules, self.spline, fill=fill_value)
        return df.to_numpy()

    def local_vectors_lateral(self, fill_value: float = 0.0) -> NDArray[np.float32]:
        """Return the local vectors in the lateral direction."""
        df = calc_localvec_lat(self.molecules, self.spline, fill=fill_value)
        return df.to_numpy()


class CollectedProps(NamedTuple):
    """Tuple of the collected local and global properties."""

    loc: pl.DataFrame
    """Collected local properties."""
    glob: pl.DataFrame
    """Collected global properties."""
