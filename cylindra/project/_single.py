from typing import Iterable, Union, TYPE_CHECKING
from pathlib import Path
import json
import warnings
from pydantic import BaseModel
import polars as pl
import numpy as np

from cylindra.const import IDName, PropertyNames as H, get_versions, cast_dataframe
from cylindra.project._base import BaseProject, PathLike, resolve_path, MissingWedge
from cylindra.project._utils import extract, as_main_function

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget
    from cylindra.components import CylSpline, CylTomogram
    from acryo import Molecules


class MoleculesInfo(BaseModel):
    """Info of molecules layer."""

    name: str  # including extension
    source: Union[int, None] = None
    visible: bool = True

    @property
    def stem(self) -> str:
        return Path(self.name).stem


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
    molecules_info: list[MoleculesInfo]
    template_image: Union[PathLike, None]
    mask_parameters: Union[None, tuple[float, float], PathLike]
    missing_wedge: MissingWedge = MissingWedge(params={}, kind="none")
    project_path: Union[Path, None] = None
    project_description: str = ""

    def _post_init(self):
        if hasattr(self, "tilt_range"):
            self.missing_wedge = MissingWedge.parse(self.tilt_range)
            del self.tilt_range

    @property
    def localprops_path(self) -> Path:
        return self.project_dir / "localprops.csv"

    @property
    def globalprops_path(self) -> Path:
        return self.project_dir / "globalprops.csv"

    @property
    def default_spline_config_path(self) -> Path:
        return self.project_dir / "default_spline_config.json"

    @property
    def macro_path(self) -> Path:
        return self.project_dir / "script.py"

    def molecules_path(self, name: str) -> Path:
        """Get the path of the molecule file of give name (needs extention)."""
        return self.project_dir / name

    def spline_path(self, idx: int) -> Path:
        """Get the path of the idx-th spline json file."""
        return self.project_dir / f"spline-{idx}.json"

    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.image = resolve_path(self.image, file_dir)
        self.template_image = resolve_path(self.template_image, file_dir, default=None)
        if isinstance(self.mask_parameters, (Path, str)):
            self.mask_parameters = resolve_path(self.mask_parameters, file_dir)
        return self

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraMainWidget",
        json_path: Path,
        mole_ext: str = ".csv",
    ) -> "CylindraProject":
        """Construct a project from a widget state."""
        from cylindra.types import MoleculesLayer
        from datetime import datetime

        if json_path.suffix == "":
            json_path = json_path.with_suffix(".json")

        _versions = get_versions()
        tomo = gui.tomogram

        results_dir = json_path.parent

        # Save path of molecules
        mole_infos = list[MoleculesInfo]()
        for layer in gui.parent_viewer.layers:
            if not isinstance(layer, MoleculesLayer):
                continue
            try:
                _src = gui.tomogram.splines.index(layer.source_component)
            except ValueError:
                _src = None
            mole_infos.append(
                MoleculesInfo(
                    name=f"{layer.name}{mole_ext}", source=_src, visible=layer.visible
                )
            )

        def as_relative(p: "Path | None"):
            assert isinstance(p, Path) or p is None
            try:
                out = p.relative_to(results_dir)
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
            molecules_info=mole_infos,
            template_image=as_relative(gui.sta.params.template_path.value),
            mask_parameters=gui.sta.params._get_mask_params(),
            missing_wedge=MissingWedge.parse(tomo.tilt_range),
            project_path=json_path,
        )

    @classmethod
    def save_gui(
        cls: "type[CylindraProject]",
        gui: "CylindraMainWidget",
        json_path: Path,
        mole_ext: str = ".csv",
    ) -> None:
        """
        Serialize the GUI state to a json file.

        Parameters
        ----------
        gui : CylindraMainWidget
            The main widget from which project model will be constructed.
        json_path : Path
            The path to the project json file.
        """
        from cylindra.types import MoleculesLayer

        self = cls.from_gui(gui, json_path, mole_ext)

        tomo = gui.tomogram
        localprops = tomo.splines.collect_localprops(allow_none=True)
        globalprops = tomo.splines.collect_globalprops(allow_none=True)

        results_dir = json_path.parent

        if not results_dir.exists():
            results_dir.mkdir()
        if localprops is not None:
            localprops.write_csv(results_dir / "localprops.csv")
        if globalprops is not None:
            globalprops.write_csv(results_dir / "globalprops.csv")
        for i, spl in enumerate(gui.tomogram.splines):
            spl.to_json(results_dir / f"spline-{i}.json")
        for layer in gui.parent_viewer.layers:
            if not isinstance(layer, MoleculesLayer):
                continue
            layer.molecules.to_file(results_dir / f"{layer.name}{mole_ext}")
        with open(results_dir / "default_spline_config.json", mode="w") as f:
            js = gui.default_config.asdict()
            json.dump(js, f, indent=4, separators=(", ", ": "))

        # save macro
        expr = as_main_function(gui._format_macro(gui.macro[gui._macro_offset :]))
        (results_dir / "script.py").write_text(expr)

        self.project_description = gui.GeneralInfo.project_desc.value
        self.to_json(json_path)
        return None

    def _to_gui(
        self,
        gui: "CylindraMainWidget | None" = None,
        filter: bool = True,
        paint: bool = True,
        read_image: bool = True,
        update_config: bool = True,
    ):
        """Update CylindraMainWidget state based on the project model."""
        from cylindra.components import SplineConfig
        from magicclass.utils import thread_worker

        gui = _get_instance(gui)
        tomogram = self.load_tomogram(compute=read_image)
        gui._macro_offset = len(gui.macro)

        cb = gui._send_tomogram_to_viewer.with_args(tomogram, filt=filter)
        yield cb
        cb.await_call()
        gui._reserved_layers.image.bounding_box.visible = not read_image

        @thread_worker.callback
        def _update_widget():
            if len(tomogram.splines) > 0:
                gui._update_splines_in_images()
                with gui.macro.blocked():
                    gui.sample_subtomograms()
            if self.default_spline_config_path.exists() and update_config:
                gui.default_config = SplineConfig.from_file(
                    self.default_spline_config_path
                )

            # append macro
            gui.macro.extend(extract(self.macro_path.read_text()).args)

            # load subtomogram analyzer state
            gui.sta.params.template_path.value = self.template_image or ""
            gui.sta._set_mask_params(self.mask_parameters)
            gui.reset_choices()
            gui._need_save = False

        yield _update_widget
        _update_widget.await_call()

        if paint and self.localprops_path.exists():
            yield from gui.paint_cylinders.arun()

        # load molecules
        _add_mole = thread_worker.callback(gui.add_molecules)
        for info in self.molecules_info:
            path = self.molecules_path(info.name)
            if not path.exists():
                warnings.warn(
                    f"Cannot find molecule file {path}. Probably it was moved?"
                )
                continue
            mole = self.load_molecules(info.name)
            if info.source is not None:
                src = tomogram.splines[info.source]
            else:
                src = None
            cb = _add_mole.with_args(mole, info.stem, src, visible=info.visible)
            yield cb
            cb.await_call(timeout=10)

        @thread_worker.callback
        def out():
            # update project description widget
            gui.GeneralInfo.project_desc.value = self.project_description

        return out

    def load_spline(self, idx: int) -> "CylSpline":
        """Load the spline with the given index."""
        from cylindra.components import CylSpline

        spl = CylSpline.from_json(self.spline_path(idx))
        if self.localprops_path.exists():
            _loc = pl.read_csv(self.localprops_path).filter(
                pl.col(IDName.spline) == idx
            )
            _loc = _drop_null_columns(_loc)
        else:
            _loc = pl.DataFrame([])
        if self.globalprops_path.exists():
            _glob = pl.read_csv(self.globalprops_path)[idx]
            _glob = _drop_null_columns(_glob)
        else:
            _glob = pl.DataFrame([])

        if H.spl_dist in _loc.columns:
            _loc = _loc.drop(H.spl_dist)
        if H.spl_pos in _loc.columns:
            spl._anchors = np.asarray(_loc[H.spl_pos])
            _loc = _loc.drop(H.spl_pos)
        for c in [IDName.spline, IDName.pos]:
            if c in _loc.columns:
                _loc = _loc.drop(c)
        spl.props.loc = cast_dataframe(_loc)
        spl.props.glob = cast_dataframe(_glob)

        return spl

    def iter_spline_paths(self) -> "Iterable[Path]":
        """Iterate over the paths of splines."""
        yield from self.project_dir.glob("spline-*.json")

    def iter_load_splines(self) -> "Iterable[CylSpline]":
        """Load all splines iteratively."""
        from cylindra.components import CylSpline

        if self.localprops_path.exists():
            _localprops = pl.read_csv(self.localprops_path)
        else:
            _localprops = None
        if self.globalprops_path.exists():
            _globalprops = pl.read_csv(self.globalprops_path)
        else:
            _globalprops = None
        for spl_path in self.iter_spline_paths():
            spl = CylSpline.from_json(spl_path)
            idx = int(spl_path.stem.split("-")[1])
            if _localprops is not None:
                _loc = _localprops.filter(pl.col(IDName.spline) == idx)
                _loc = _drop_null_columns(_loc)
            else:
                _loc = pl.DataFrame([])
            if _globalprops is not None:
                _glob = _globalprops[idx]
                _glob = _drop_null_columns(_glob)
            else:
                _glob = pl.DataFrame([])

            if H.spl_dist in _loc.columns:
                _loc = _loc.drop(H.spl_dist)
            if H.spl_pos in _loc.columns:
                spl._anchors = np.asarray(_loc[H.spl_pos])
                _loc = _loc.drop(H.spl_pos)
            for c in [IDName.spline, IDName.pos]:
                if c in _loc.columns:
                    _loc = _loc.drop(c)
            spl.props.loc = cast_dataframe(_loc)
            spl.props.glob = cast_dataframe(_glob)
            yield spl

    def iter_molecule_paths(self) -> "Iterable[Path]":
        """Iterate over the paths of molecules."""
        for info in self.molecules_info:
            yield self.molecules_path(info.name)

    def load_molecules(self, name: str) -> "Molecules":
        """Load the molecules with the given name."""
        from acryo import Molecules

        mole_path = self.molecules_path(name)
        return Molecules.from_file(mole_path)

    def iter_load_molecules(self) -> "Iterable[tuple[str, Molecules]]":
        """Load all molecules iteratively."""
        for info in self.molecules_info:
            path = self.molecules_path(info.name)
            if not path.exists():
                warnings.warn(
                    f"Cannot find molecule file {path}. Probably it was moved?"
                )
                continue
            yield info.stem, self.load_molecules(info.name)

    def load_tomogram(self, compute: bool = True) -> "CylTomogram":
        """Load the tomogram object of the project."""
        from cylindra.components import CylTomogram

        tomo = CylTomogram.imread(
            path=self.image,
            scale=self.scale,
            tilt=self.missing_wedge.as_param(),
            binsize=self.multiscales,
            compute=compute,
        )
        tomo.splines.extend(self.iter_load_splines())
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
    def project_dir(self) -> Path:
        """The directory of the project."""
        if self.project_path is None:
            raise ValueError("Project path is not set.")
        return self.project_path.parent


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
