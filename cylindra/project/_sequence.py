from __future__ import annotations
from contextlib import suppress
from pathlib import Path

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    MutableSequence,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
import polars as pl

from cylindra.const import (
    MoleculesHeader as Mole,
    IDName,
    PropertyNames as H,
    cast_dataframe,
)
from cylindra.project._single import CylindraProject
from cylindra._config import get_config

if TYPE_CHECKING:
    from typing_extensions import Self
    from cylindra.components import CylSpline
    from acryo import BatchLoader, Molecules

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
            out = ProjectSequence(check_scale=True)
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
                    self.add(path)
        else:
            for path in paths:
                self.add(path)
        return self

    def add(self, path: str | Path) -> Self:
        """Add a project from a path."""
        prj = CylindraProject.from_json(path)
        self._scale_validator.value = prj.scale
        self._projects.append(prj)
        return self

    def sta_loader(self) -> BatchLoader:
        """Construct a STA loader from all the projects."""
        import impy as ip
        from acryo import BatchLoader, Molecules

        col = BatchLoader(scale=self._scale_validator.value)
        for idx, prj in enumerate(self._projects):
            tomo = ip.lazy.imread(prj.image, chunks=get_config().dask_chunk)
            for fp in prj.molecules:
                fp = Path(fp)
                mole = Molecules.from_file(fp)
                mole.features = mole.features.with_columns(
                    pl.repeat(fp.stem, pl.count()).alias(Mole.id)
                )
                col.add_tomogram(tomo.value, molecules=mole, image_id=idx)
        return col

    def collect_localprops(
        self, allow_none: bool = True, id: _IDTYPE = "int"
    ) -> pl.DataFrame:
        """
        Collect all localprops into a single dataframe.

        Parameters
        ----------
        allow_none : bool, default is True
            Continue data collection even if property table data file was not
            found in any project. Raise error otherwise.
        id : str, default is "int"
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
            path = prj.localprops
            if path is None and not allow_none:
                raise ValueError(
                    f"Localprops not found in project at {prj.project_path}."
                )
            else:
                df = pl.read_csv(path)
                columns = [pl.repeat(idx, pl.count()).cast(pl.UInt16).alias(Mole.image)]
                if IDName.spline in df.columns:
                    columns.append(pl.col(IDName.spline).cast(pl.UInt16))
                dataframes.append(df.with_columns(columns))
        out = cast_dataframe(pl.concat(dataframes, how="diagonal"))
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
        allow_none : bool, default is True
            Continue data collection even if property table data file was not
            found in any project. Raise error otherwise.
        suffix : str, default is ""
            Suffix to add to the column names that may be collide with the local
            properties.
        id : str, default is "int"
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
            path = prj.globalprops
            if path is None and not allow_none:
                raise ValueError(
                    f"Globalprops not found in project at {prj.project_path}."
                )
            imagespec = pl.Series(Mole.image, [idx]).cast(pl.UInt16)
            df = pl.read_csv(path).with_columns(imagespec)
            dataframes.append(df)
        out = cast_dataframe(pl.concat(dataframes, how="diagonal"))
        if suffix:
            need_rename = [H.spacing, H.skew, H.npf, H.rise, H.radius]
            out = out.rename(
                {col: col + suffix for col in need_rename if col in out.columns}
            )
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
        allow_none : bool, default is True
            Continue data collection even if property table data file was not
            found in any project. Raise error otherwise.
        id : str, default is "int"
            How to describe the source tomogram. If "int", each tomogram will
            be labeled with ascending integers. If "path", each tomogram will
            be labeled with the name of the project directory.

        Returns
        -------
        pl.DataFrame
            Dataframe with all the properties.
        """
        loc = self.collect_localprops(allow_none=allow_none, id="int")
        if spline_details:
            _all_ders = list[np.ndarray]()
            for _, spl in self.iter_splines():
                arrs = [spl.map(der=d) for d in range(3)]
                ders = np.concatenate(arrs, axis=1)
                _all_ders.append(ders)
            _all_ders = np.concatenate(_all_ders, axis=0)
            _col_names = [
                f"spline_{s}"
                for s in ["z", "y", "x", "dz", "dy", "dx", "ddz", "ddy", "ddx"]
            ]
            loc = loc.with_columns(
                [
                    pl.Series(name, _all_ders[:, i], dtype=pl.Float32)
                    for i, name in enumerate(_col_names)
                ]
            )
        glb = self.collect_globalprops(allow_none=allow_none, id="int")
        if spline_details:
            lengths = list[float]()
            for _, spl in self.iter_splines():
                lengths.append(spl.length())
            col = pl.Series("spline_length", lengths, dtype=pl.Float32)
            glb = glb.with_columns(col)
        key = [IDName.spline, Mole.image]
        out = loc.join(glb, on=key, suffix="_glob")
        return self._normalize_id(out, id)

    def iter_splines(self) -> Iterable[tuple[tuple[int, int], CylSpline]]:
        """Iterate over all the splines in all the projects."""
        for i_prj, prj in enumerate(self._projects):
            for i_spl in range(prj.nsplines):
                spl = prj.load_spline(i_spl)
                yield (i_prj, i_spl), spl

    def iter_molecules(
        self,
        spline_props: str | Sequence[str] = (),
        suffix: str = "",
    ) -> Iterable[tuple[tuple[int, str], Molecules]]:
        """
        Iterate over all the molecules in all the projects.

        Parameters
        ----------
        spline_props : str or sequence of str
            Spline global properties to be concatenated to the molecules features.
        suffix : str, default is ""
            Suffix that will be added to the spline global property names.
        """
        if isinstance(spline_props, str):
            spline_props = [spline_props]
        for i_prj, prj in enumerate(self._projects):
            for i_mole in range(prj.nmolecules):
                name = prj.molecules[i_mole].stem
                mole = prj.load_molecules(i_mole)
                if (
                    spline_props
                    and prj.molecules_info is not None
                    and (src := prj.molecules_info[i_mole].source) is not None
                ):
                    spl = prj.load_spline(src)
                    features = list[pl.Expr]()
                    for propname in spline_props:
                        prop = spl.props.get_glob(propname, None)
                        if prop is None:
                            continue
                        propname_glob = propname + suffix
                        features.append(
                            pl.repeat(prop, pl.count()).alias(propname_glob)
                        )
                    mole = mole.with_features(features)
                yield (i_prj, name), mole

    def collect_spline_coords(self, ders: int | Iterable[int] = 0) -> pl.DataFrame:
        """
        Collect spline coordinates or its derivative(s) as a dataframe.

        Coordinates will be labeled as "z", "y", "x". The 1st derivative will be
        labeled as "dz", "dy", "dx", and so on.

        Parameters
        ----------
        ders : int or iterable of int, default is 0
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
                    pl.repeat(j, nanc, eager=True, dtype=pl.UInt16).alias(
                        IDName.spline
                    ),
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
        if id == "int":
            pass
        elif id == "path":
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
            out = out.with_columns(
                pl.col(Mole.image).map_dict(_map, return_dtype=pl.Categorical)
            )
        else:
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
