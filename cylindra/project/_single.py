from typing import Union, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel
import polars as pl
import numpy as np

from cylindra.const import IDName, PropertyNames as H, get_versions, cast_dataframe
from cylindra.project._base import BaseProject, PathLike, resolve_path
from cylindra.project._utils import extract, as_main_function

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget
    from cylindra.components import CylSpline, CylTomogram
    from acryo import Molecules


class MoleculesInfo(BaseModel):
    """Info of molecules layer."""

    source: Union[int, None] = None
    visible: bool = True


class CylindraProject(BaseProject):
    """A project of cylindra."""

    class Config:
        # this allows extra fields in the json file, for backward compatibility
        extra = "allow"

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    image: PathLike
    scale: float
    multiscales: list[int]
    splines: list[PathLike]
    localprops: Union[PathLike, None]
    globalprops: Union[PathLike, None]
    molecules: list[PathLike]
    molecules_info: Union[list[MoleculesInfo], None] = None
    global_variables: PathLike
    template_image: Union[PathLike, None]
    mask_parameters: Union[None, tuple[float, float], PathLike]
    tilt_range: Union[tuple[float, float], None]
    macro: PathLike
    project_path: Union[Path, None] = None
    project_description: str = ""

    def _post_init(self):
        if (attr := getattr(self, "molecule_sources", None)) is not None:
            self.molecules_info = [MoleculesInfo(source=s, visible=True) for s in attr]
            delattr(self, "molecule_sources")

    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.image = resolve_path(self.image, file_dir)
        self.localprops = resolve_path(self.localprops, file_dir)
        self.globalprops = resolve_path(self.globalprops, file_dir)
        self.template_image = resolve_path(self.template_image, file_dir, default=None)
        if isinstance(self.mask_parameters, (Path, str)):
            self.mask_parameters = resolve_path(self.mask_parameters, file_dir)
        self.global_variables = resolve_path(self.global_variables, file_dir)
        self.splines = [resolve_path(p, file_dir) for p in self.splines]
        self.molecules = [resolve_path(p, file_dir) for p in self.molecules]
        self.macro = resolve_path(self.macro, file_dir)
        return self

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraMainWidget",
        json_path: Path,
        results_dir: Union[Path, None] = None,
    ) -> "CylindraProject":
        """Construct a project from a widget state."""
        from cylindra.types import MoleculesLayer
        from datetime import datetime

        if json_path.suffix == "":
            json_path = json_path.with_suffix(".json")

        _versions = get_versions()
        tomo = gui.tomogram
        localprops = tomo.collect_localprops()
        globalprops = tomo.collect_globalprops()

        if results_dir is None:
            results_dir = json_path.parent / (json_path.stem + "_results")
        else:
            results_dir = Path(results_dir)
        localprops_path = None if localprops is None else results_dir / "localprops.csv"
        globalprops_path = (
            None if globalprops is None else results_dir / "globalprops.csv"
        )

        # Save path of splines
        spline_paths = list[Path]()
        for i in range(len(gui.tomogram.splines)):
            spline_paths.append(results_dir / f"spline-{i}.json")

        # Save path of molecules
        molecules_paths = list[Path]()
        molecules_info = list[MoleculesInfo]()
        for layer in gui.parent_viewer.layers:
            if not isinstance(layer, MoleculesLayer):
                continue
            molecules_paths.append((results_dir / layer.name).with_suffix(".csv"))
            try:
                _src = gui.tomogram.splines.index(layer.source_component)
            except ValueError:
                _src = None
            molecules_info.append(MoleculesInfo(source=_src, visible=layer.visible))

        # Save path of  global variables
        gvar_path = results_dir / "global_variables.json"

        # Save path of macro
        macro_path = results_dir / "script.py"

        file_dir = json_path.parent

        def as_relative(p: "Path | None"):
            assert isinstance(p, Path) or p is None
            try:
                out = p.relative_to(file_dir)
            except Exception:
                out = p
            return out

        return cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=_versions.pop("cylindra", "unknown"),
            dependency_versions=_versions,
            image=as_relative(tomo.source),
            scale=tomo.scale,
            multiscales=[x[0] for x in tomo.multiscaled],
            splines=[as_relative(p) for p in spline_paths],
            localprops=as_relative(localprops_path),
            globalprops=as_relative(globalprops_path),
            molecules=[as_relative(p) for p in molecules_paths],
            molecules_info=molecules_info,
            global_variables=as_relative(gvar_path),
            template_image=as_relative(gui.sta.params.template_path.value),
            mask_parameters=gui.sta.params._get_mask_params(),
            tilt_range=tomo.tilt_range,
            macro=as_relative(macro_path),
            project_path=json_path,
        )

    @classmethod
    def save_gui(
        cls: "type[CylindraProject]",
        gui: "CylindraMainWidget",
        json_path: Path,
        results_dir: Union[Path, None] = None,
    ) -> None:
        """
        Serialize the GUI state to a json file.

        Parameters
        ----------
        gui : CylindraMainWidget
            The main widget from which project model will be constructed.
        json_path : Path
            The path to the project json file.
        results_dir : Path, optional
            The directory to save the results.
        """
        from cylindra.types import MoleculesLayer

        self = cls.from_gui(gui, json_path, results_dir)

        tomo = gui.tomogram
        localprops = tomo.collect_localprops(allow_none=True)
        globalprops = tomo.collect_globalprops(allow_none=True)

        if results_dir is None:
            results_dir = json_path.parent / (json_path.stem + "_results")
        else:
            results_dir = Path(results_dir)
        localprops_path = None if localprops is None else results_dir / "localprops.csv"
        globalprops_path = (
            None if globalprops is None else results_dir / "globalprops.csv"
        )

        molecule_dataframes: "list[pl.DataFrame]" = []
        for layer in gui.parent_viewer.layers:
            if not isinstance(layer, MoleculesLayer):
                continue
            mole = layer.molecules
            molecule_dataframes.append(mole.to_dataframe())

        if not results_dir.exists():
            results_dir.mkdir()
        if localprops_path:
            localprops.write_csv(localprops_path)
        if globalprops_path:
            globalprops.write_csv(globalprops_path)
        if self.splines:
            for spl, path in zip(gui.tomogram.splines, self.splines):
                spl.to_json(results_dir / path)
        if self.molecules:
            for df, path in zip(molecule_dataframes, self.molecules):
                df.write_csv(results_dir / path)

        gui.global_variables.save_variables(results_dir / self.global_variables)

        # save macro
        fp = results_dir / str(self.macro)
        fp.write_text(
            as_main_function(gui._format_macro(gui.macro[gui._macro_offset :]))
        )

        self.project_description = gui.GeneralInfo.project_desc.value
        self.to_json(json_path)
        return None

    def to_gui(
        self,
        gui: "CylindraMainWidget | None" = None,
        filter: bool = True,
        paint: bool = True,
    ):
        """Update CylindraMainWidget state based on the project model."""
        gui = _get_instance(gui)
        tomogram = self.load_tomogram()
        gui._macro_offset = len(gui.macro)

        # load splines
        molecules_list = [self.load_molecules(i) for i in range(self.nmolecules)]

        def _load_project_on_return():
            gui._send_tomogram_to_viewer(tomogram, filt=filter)

            if len(tomogram.splines) > 0:
                gui._update_splines_in_images()
                with gui.macro.blocked():
                    gui.sample_subtomograms()

            # load global variables
            if self.global_variables:
                with gui.macro.blocked():
                    gui.Others.GlobalVariables.load_variables(self.global_variables)

            # append macro
            gui.macro.extend(extract(Path(self.macro).read_text()).args)

            # load subtomogram analyzer state
            gui.sta.params.template_path.value = self.template_image or ""
            gui.sta._set_mask_params(self.mask_parameters)
            gui.reset_choices()
            gui._need_save = False

            # paint if needed
            if paint and self.localprops:
                gui.paint_cylinders()

            # load molecules
            for idx, mole in enumerate(molecules_list):
                _src = None
                if self.molecules_info is not None:
                    _info = self.molecules_info[idx]
                    if _info.source is not None:
                        _src = tomogram.splines[_info.source]
                    visible = _info.visible
                else:
                    visible = True
                _fpath = self.molecules[idx]

                layer = gui.add_molecules(mole, name=Path(_fpath).stem, source=_src)
                if not visible:
                    layer.visible = False

            # update project description widget
            gui.GeneralInfo.project_desc.value = self.project_description

        return _load_project_on_return

    def load_spline(self, idx: int) -> "CylSpline":
        """Load the spline with the given index."""
        from cylindra.components import CylSpline

        spl = CylSpline.from_json(self.splines[idx])
        if self.localprops is not None:
            _loc = pl.read_csv(self.localprops).filter(pl.col(IDName.spline) == idx)
            _loc = _drop_null_columns(_loc)
        else:
            _loc = pl.DataFrame([])
        if self.globalprops is not None:
            _glob = pl.read_csv(self.globalprops)[idx]
            _glob = _drop_null_columns(_glob)
        else:
            _glob = pl.DataFrame([])

        if all((c in _loc.columns) for c in [H.splDist, H.splPos]):
            spl._anchors = np.asarray(_loc[H.splPos])
        for c in [IDName.spline, IDName.pos]:
            if c in _loc.columns:
                _loc = _loc.drop(c)
        spl._localprops = cast_dataframe(_loc)
        spl._globalprops = cast_dataframe(_glob)

        return spl

    def load_molecules(self, idx: "int | str") -> "Molecules":
        """Load the molecules with the given index or name."""
        from acryo import Molecules

        if isinstance(idx, str):
            for path in self.molecules:
                if Path(path).stem == idx:
                    return Molecules.from_csv(path)
            _names = [repr(Path(path).stem) for path in self.molecules]
            raise ValueError(
                f"Cannot find molecule with name {idx}, available names are: {_names}."
            )
        return Molecules.from_csv(self.molecules[idx])

    def load_tomogram(self) -> "CylTomogram":
        """Load the tomogram object of the project."""
        from cylindra.components import CylTomogram

        tomo = CylTomogram.imread(
            path=self.image,
            scale=self.scale,
            tilt_range=self.tilt_range,
            binsize=self.multiscales,
        )

        splines_list = [self.load_spline(i) for i in range(self.nsplines)]
        tomo.splines.extend(splines_list)

        return tomo

    def make_project_viewer(self):
        """Build a project viewer widget from this project."""
        from ._widgets import ProjectViewer

        pviewer = ProjectViewer()
        pviewer._from_project(self)
        return pviewer

    def make_component_viewer(self):
        from ._widgets import ComponentsViewer

        """Build a molecules viewer widget from this project."""
        mviewer = ComponentsViewer()
        mviewer._from_project(self)
        return mviewer

    @property
    def result_dir(self) -> Path:
        return Path(self.macro).parent

    @property
    def nsplines(self) -> int:
        """Number of splines in the project."""
        return len(self.splines)

    @property
    def nmolecules(self) -> int:
        """Number of molecules in the project."""
        return len(self.molecules)


def _get_instance(gui: "CylindraMainWidget | None" = None):
    if gui is not None:
        return gui
    from cylindra import instance

    ui = instance()
    if ui is None:
        raise RuntimeError("No CylindraMainWidget GUI found.")
    return ui


def _drop_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    nrows = df.shape[0]
    to_drop = list[str]()
    for count in df.null_count().row(0):
        if count == nrows:
            to_drop.append(df.columns[count])
    if to_drop:
        return df.drop(to_drop)
    return df
