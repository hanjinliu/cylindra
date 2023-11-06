from contextlib import contextmanager
import tempfile
from datetime import datetime
from typing import Generator, Iterable, TYPE_CHECKING
from pathlib import Path
import json
import warnings
from pydantic import Field
import polars as pl

from cylindra.const import (
    PropertyNames as H,
    get_versions,
    cast_dataframe,
    ImageFilter,
)
from cylindra.project._base import BaseProject, PathLike, resolve_path, MissingWedge
from cylindra.project._utils import extract, as_main_function
from cylindra.project._layer_info import MoleculesInfo, LandscapeInfo

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget
    from cylindra.components import CylSpline, CylTomogram
    from acryo import Molecules


class CylindraProject(BaseProject):
    """A project of cylindra."""

    class Config:
        # this allows extra fields in the json file, for backward compatibility
        extra = "allow"

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    image: PathLike | None
    scale: float
    multiscales: list[int]
    molecules_info: list[MoleculesInfo] = Field(default_factory=list)
    landscape_info: list[LandscapeInfo] = Field(default_factory=list)
    missing_wedge: MissingWedge = MissingWedge(params={}, kind="none")
    project_path: Path | None = None
    project_description: str = ""

    def resolve_path(self, file_dir: PathLike):
        """Resolve the path of the project."""
        file_dir = Path(file_dir)
        self.image = resolve_path(self.image, file_dir)
        return self

    @classmethod
    def new(
        cls,
        image: PathLike,
        scale: float,
        multiscales: list[int],
        missing_wedge: tuple[float, float] | None = None,
        project_path: Path | None = None,
    ):
        """Create a new project."""
        _versions = get_versions()
        if image is None:
            raise ValueError("image must not be None.")
        return CylindraProject(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=_versions.pop("cylindra", "unknown"),
            dependency_versions=_versions,
            image=image,
            scale=scale,
            multiscales=multiscales,
            missing_wedge=MissingWedge.parse(missing_wedge),
            project_path=project_path,
        )

    def save(self, project_dir: Path):
        from macrokit import parse

        path = Path(self.image).as_posix()
        scale = self.scale
        bin_size = self.multiscales
        tilt_range = self.missing_wedge.as_param()
        with _prep_save_dir(project_dir) as results_dir:
            expr_open = parse(
                f"ui.open_image({path=}, {scale=:.4f}, {bin_size=}, {tilt_range=})",
                squeeze=False,
            )
            expr = as_main_function(expr_open)
            self.script_py_path(results_dir).write_text(expr)
            self.to_json(self.project_json_path(results_dir))

    @classmethod
    def from_gui(
        cls,
        gui: "CylindraMainWidget",
        project_dir: Path,
        mole_ext: str = ".csv",
        save_landscape: bool = False,
    ) -> "CylindraProject":
        """Construct a project from a widget state."""

        _versions = get_versions()
        tomo = gui.tomogram

        # Save path of molecules
        mole_infos = list[MoleculesInfo]()
        for layer in gui.mole_layers:
            mole_infos.append(MoleculesInfo.from_layer(gui, layer, mole_ext))

        # Save paths of landscape
        landscape_infos = list[LandscapeInfo]()
        if save_landscape:
            from cylindra._napari import LandscapeSurface

            for layer in gui.parent_viewer.layers:
                if not isinstance(layer, LandscapeSurface):
                    continue
                landscape_infos.append(LandscapeInfo.from_layer(gui, layer))

        def as_relative(p: "Path | None"):
            assert isinstance(p, Path) or p is None
            try:
                out = p.relative_to(project_dir)
            except Exception:
                out = p
            return out

        return cls(
            datetime=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version=_versions.pop("cylindra", "unknown"),
            dependency_versions=_versions,
            image=as_relative(tomo.metadata.get("source", None)),
            scale=tomo.scale,
            multiscales=[x[0] for x in tomo.multiscaled],
            molecules_info=mole_infos,
            landscape_info=landscape_infos,
            missing_wedge=MissingWedge.parse(tomo.tilt_range),
            project_path=project_dir,
        )

    @classmethod
    def save_gui(
        cls: "type[CylindraProject]",
        gui: "CylindraMainWidget",
        project_dir: Path,
        mole_ext: str = ".csv",
        save_landscape: bool = False,
    ) -> None:
        """
        Serialize the GUI state to a json file.

        Parameters
        ----------
        gui : CylindraMainWidget
            The main widget from which project model will be constructed.
        project_dir : Path
            The path to the project json file.
        """
        self = cls.from_gui(gui, project_dir, mole_ext, save_landscape)

        tomo = gui.tomogram
        localprops = tomo.splines.collect_localprops(allow_none=True)
        globalprops = tomo.splines.collect_globalprops(allow_none=True)

        with _prep_save_dir(project_dir) as results_dir:
            if localprops is not None:
                localprops.write_csv(self.localprops_path(results_dir))
            if globalprops is not None:
                globalprops.write_csv(self.globalprops_path(results_dir))
            for i, spl in enumerate(gui.tomogram.splines):
                spl.to_json(results_dir / f"spline-{i}.json")
            for info in self.molecules_info + self.landscape_info:
                info.save_layer(gui, results_dir)

            js = gui.default_config.asdict()
            with open(self.default_spline_config_path(results_dir), mode="w") as f:
                json.dump(js, f, indent=4, separators=(", ", ": "))

            # save macro
            expr = as_main_function(gui._format_macro(gui.macro[gui._macro_offset :]))
            self.script_py_path(results_dir).write_text(expr)

            self.project_description = gui.GeneralInfo.project_desc.value
            self.to_json(self.project_json_path(results_dir))
        return None

    def _to_gui(
        self,
        gui: "CylindraMainWidget | None" = None,
        filter: "ImageFilter | None" = True,
        read_image: bool = True,
        update_config: bool = True,
    ):
        """Update CylindraMainWidget state based on the project model."""
        from cylindra.components import SplineConfig
        from magicclass.utils import thread_worker

        gui = _get_instance(gui)
        with self.open_project() as project_dir:
            tomogram = self.load_tomogram(project_dir, compute=read_image)
            macro_expr = extract(self.script_py_path(project_dir).read_text()).args
            cfg_path = project_dir / "default_spline_config.json"
            if cfg_path.exists() and update_config:
                default_config = SplineConfig.from_file(cfg_path)
            else:
                default_config = None

            cb = gui._send_tomogram_to_viewer.with_args(tomogram, filt=filter)
            yield cb
            cb.await_call()
            gui._macro_offset = len(gui.macro)

            @thread_worker.callback
            def _update_widget():
                gui._reserved_layers.image.bounding_box.visible = not read_image
                if len(tomogram.splines) > 0:
                    gui._update_splines_in_images()
                    with gui.macro.blocked():
                        gui.sample_subtomograms()
                if default_config is not None:
                    gui.default_config = default_config

                gui.macro.extend(macro_expr)

                # load subtomogram analyzer state
                gui.reset_choices()
                gui._need_save = False

            yield _update_widget
            _update_widget.await_call()

            # load molecules
            _add_layer = thread_worker.callback(gui.parent_viewer.add_layer)
            for info in self.molecules_info + self.landscape_info:
                layer = info.to_layer(gui, project_dir)
                cb = _add_layer.with_args(layer)
                yield cb
                cb.await_call(timeout=10)

        @thread_worker.callback
        def out():
            # update project description widget
            gui.GeneralInfo.project_desc.value = self.project_description

        return out

    def load_spline(self, dir: Path, idx: int) -> "CylSpline":
        """Load the spline with the given index."""
        from cylindra.components import CylSpline

        spl = CylSpline.from_json(dir / f"spline-{idx}.json")
        localprops_path = self.localprops_path(dir)
        globalprops_path = self.globalprops_path(dir)
        if localprops_path.exists():
            _loc = pl.read_csv(localprops_path).filter(pl.col(H.spline_id) == idx)
            _loc = _drop_null_columns(_loc)
        else:
            _loc = pl.DataFrame([])
        if globalprops_path.exists():
            _glob = pl.read_csv(globalprops_path)[idx]
            _glob = _drop_null_columns(_glob)
        else:
            _glob = pl.DataFrame([])

        if H.spl_dist in _loc.columns:
            _loc = _loc.drop(H.spl_dist)
        if H.spl_pos in _loc.columns:
            spl._anchors = _loc[H.spl_pos].to_numpy()
            _loc = _loc.drop(H.spl_pos)
        for c in [H.spline_id, H.pos_id]:
            if c in _loc.columns:
                _loc = _loc.drop(c)
        spl.props.loc = cast_dataframe(_loc)
        spl.props.glob = cast_dataframe(_glob)

        return spl

    def iter_spline_paths(
        self, dir: Path | None = None
    ) -> "Iterable[tuple[int, Path]]":
        """Iterate over the paths of splines and their indices."""
        if dir is None:
            with self.open_project() as dir:
                paths = list(dir.glob("spline-*.json"))
        else:
            paths = list(dir.glob("spline-*.json"))
        # sort by index
        idx_paths = [(int(p.stem.split("-")[1]), p) for p in paths]
        idx_paths.sort(key=lambda x: x[0])
        yield from idx_paths

    def iter_load_splines(
        self,
        dir: Path,
        drop_columns: bool = True,
    ) -> "Iterable[CylSpline]":
        """Load all splines including its properties iteratively."""
        from cylindra.components import CylSpline

        localprops_path = self.localprops_path(dir)
        globalprops_path = self.globalprops_path(dir)
        if localprops_path.exists():
            _localprops = pl.read_csv(localprops_path)
        else:
            _localprops = None
        if globalprops_path.exists():
            _globalprops = pl.read_csv(globalprops_path)
        else:
            _globalprops = None
        for idx, spl_path in self.iter_spline_paths(dir):
            spl = CylSpline.from_json(spl_path)
            if _localprops is not None:
                _loc = _localprops.filter(pl.col(H.spline_id) == idx)
                _loc = _drop_null_columns(_loc)
                if len(_loc) == 0:
                    _loc = pl.DataFrame([])
            else:
                _loc = pl.DataFrame([])
            if _globalprops is not None:
                _glob = _globalprops.filter(pl.col(H.spline_id) == idx)
                _glob = _drop_null_columns(_glob)
                if len(_glob) == 0:
                    _glob = pl.DataFrame([])
            else:
                _glob = pl.DataFrame([])

            if H.spl_dist in _loc.columns and drop_columns:
                _loc = _loc.drop(H.spl_dist)
            if H.spl_pos in _loc.columns:
                spl._anchors = _loc[H.spl_pos].to_numpy()
                if drop_columns:
                    _loc = _loc.drop(H.spl_pos)
            for c in [H.spline_id, H.pos_id]:
                if c in _loc.columns and drop_columns:
                    _loc = _loc.drop(c)
            spl.props.loc = cast_dataframe(_loc)
            spl.props.glob = cast_dataframe(_glob)
            yield spl

    def iter_load_molecules(
        self, dir: Path
    ) -> "Iterable[tuple[MoleculesInfo, Molecules]]":
        """Load all molecules iteratively."""
        from acryo import Molecules

        for info in self.molecules_info:
            path = dir / info.name
            if not path.exists():
                warnings.warn(
                    f"Cannot find molecule file {path}. Probably it was moved?"
                )
                continue
            mole = Molecules.from_file(path)
            yield info, mole

    def load_tomogram(self, dir: Path, compute: bool = True) -> "CylTomogram":
        """Load the tomogram object of the project."""
        from cylindra.components import CylTomogram

        if self.image is not None:
            tomo = CylTomogram.imread(
                path=self.image,
                scale=self.scale,
                tilt=self.missing_wedge.as_param(),
                binsize=self.multiscales,
                compute=compute,
            )
        else:
            tomo = CylTomogram.dummy(
                scale=self.scale,
                tilt=self.missing_wedge.as_param(),
                binsize=self.multiscales,
            )
        tomo.splines.extend(self.iter_load_splines(dir))
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

    @contextmanager
    def open_project(self) -> Generator[Path, None, None]:
        """Open the project within this context."""
        if self.project_path is None:
            raise ValueError("Project path is not set.")
        ext = self.project_path.suffix
        if ext == "":
            yield self.project_path

        elif ext in (".tar",):
            import tarfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(self.project_path) as tar:
                    tar.extractall(tmpdir)
                yield Path(tmpdir)

        elif ext in (".zip",):
            import zipfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(self.project_path) as zip:
                    zip.extractall(tmpdir)
                yield Path(tmpdir)

        else:
            raise ValueError(f"Unsupported extension {ext}.")

        return None

    def localprops_path(self, dir: Path) -> Path:
        """Path to the spline local properties file."""
        return dir / "localprops.csv"

    def globalprops_path(self, dir: Path) -> Path:
        """Path to the spline global properties file."""
        return dir / "globalprops.csv"

    def default_spline_config_path(self, dir: Path) -> Path:
        """Path to the default spline config file."""
        return dir / "default_spline_config.json"

    def script_py_path(self, dir: Path) -> Path:
        """Path to the script.py file."""
        return dir / "script.py"

    def project_json_path(self, dir: Path) -> Path:
        """Path to the project.json file."""
        return dir / "project.json"


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


@contextmanager
def _prep_save_dir(project_path: Path) -> Generator[Path, None, None]:
    ext = project_path.suffix
    if ext == "":
        if not project_path.exists():
            project_path.mkdir()
        yield project_path

    elif ext in (".tar",):
        import tarfile

        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            with tarfile.open(project_path, mode="w") as tar:
                for file in Path(tmpdir).glob("*"):
                    tar.add(file, arcname=file.name)

    elif ext in (".zip",):
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            with zipfile.ZipFile(project_path, mode="w") as zip:
                for file in Path(tmpdir).glob("*"):
                    zip.write(file, arcname=file.name)

    else:
        raise ValueError(f"Unsupported extension {ext}.")

    return None
