import os
from typing import Union, TYPE_CHECKING
from pathlib import Path
import macrokit as mk
from pydantic import BaseModel
import polars as pl

from cylindra.const import PropertyNames as H, get_versions
from ._base import BaseProject, PathLike, resolve_path

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget


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
    current_ft_size: float
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
        spline_paths: list[Path] = []
        for i in range(len(gui.tomogram.splines)):
            spline_paths.append(results_dir / f"spline-{i}.json")

        # Save path of molecules
        molecules_paths: list[Path] = []
        molecules_info: list[MoleculesInfo] = []
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

        def as_relative(p: Path):
            assert isinstance(p, Path)
            try:
                out = p.relative_to(file_dir)
            except Exception:
                out = p
            return out

        self = cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=_versions.pop("cylindra", "unknown"),
            dependency_versions=_versions,
            image=as_relative(tomo.source),
            scale=tomo.scale,
            multiscales=[x[0] for x in tomo.multiscaled],
            current_ft_size=gui._current_ft_size,
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
        return self

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
        macro_str = str(gui._format_macro(gui.macro[gui._macro_offset :]))

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

        molecule_dataframes: "list[pl.DataFrame]" = []
        for layer in gui.parent_viewer.layers:
            if not isinstance(layer, MoleculesLayer):
                continue
            mole = layer.molecules
            molecule_dataframes.append(mole.to_dataframe())

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)  # create a directory if not exists.
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

        if macro_str:
            fp = results_dir / str(self.macro)
            fp.write_text(macro_str)
        self.to_json(json_path)
        return None

    def to_gui(self, gui: "CylindraMainWidget", filter: bool = True):
        from cylindra.components import CylSpline, CylTomogram
        import numpy as np
        from acryo import Molecules
        import polars as pl

        gui.tomogram = CylTomogram.imread(
            path=self.image,
            scale=self.scale,
            tilt_range=self.tilt_range,
            binsize=self.multiscales,
        )

        gui._current_ft_size = self.current_ft_size
        gui._macro_offset = len(gui.macro)

        # load splines
        splines = [CylSpline.from_json(path) for path in self.splines]
        localprops_path = self.localprops
        if localprops_path is not None:
            all_localprops = dict(
                iter(pl.read_csv(localprops_path).groupby("SplineID"))
            )
        else:
            all_localprops = {}
        globalprops_path = self.globalprops
        if globalprops_path is not None:
            df = pl.read_csv(globalprops_path)
            all_globalprops = {i: df[i] for i in range(len(df))}
        else:
            all_globalprops = {}

        for i, spl in enumerate(splines):
            spl.localprops = all_localprops.get(i, None)
            if spl.has_localprops([H.splDist, H.splPos]):
                spl._anchors = np.asarray(spl.localprops[H.splPos])
                spl.localprops.drop(["SplineID", "PosID"])
            spl.globalprops = all_globalprops.get(i, None)

            spl.localprops = spl.localprops.with_columns(_get_casting(spl.localprops))
            spl.globalprops = spl.globalprops.with_columns(
                _get_casting(spl.globalprops)
            )

        def _load_project_on_return():
            gui._send_tomogram_to_viewer(filt=filter)

            if splines:
                gui.tomogram.splines.clear()
                gui.tomogram.splines.extend(splines)
                gui._update_splines_in_images()
                with gui.macro.blocked():
                    gui.sample_subtomograms()

            # load global variables
            if self.global_variables:
                with gui.macro.blocked():
                    gui.Others.GlobalVariables.load_variables(self.global_variables)

            # append macro
            with open(self.macro) as f:
                txt = f.read()

            macro = mk.parse(txt)
            gui.macro.extend(macro.args)

            # load subtomogram analyzer state
            gui.sta.params.template_path.value = self.template_image or ""
            gui.sta._set_mask_params(self.mask_parameters)
            gui.reset_choices()
            gui._need_save = False

            # paint if needed
            if self.localprops:
                gui.paint_cylinders()

            # load molecules
            for idx, path in enumerate(self.molecules):
                mole = Molecules.from_csv(path)
                _src = None
                if self.molecules_info is not None:
                    _info = self.molecules_info[idx]
                    if _info.source is not None:
                        _src = splines[_info.source]
                    visible = _info.visible
                else:
                    visible = True
                layer = gui.add_molecules(mole, name=Path(path).stem, source=_src)
                if not visible:
                    layer.visible = False

        return _load_project_on_return

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


def _get_casting(df: pl.DataFrame):
    out = []
    for cname in df.columns:
        if cname == H.nPF:
            out.append(pl.col(cname).cast(pl.UInt8))
        elif cname == H.orientation:
            out.append(pl.col(cname).cast(pl.Utf8))
        else:
            out.append(pl.col(cname).cast(pl.Float32))
    return out
