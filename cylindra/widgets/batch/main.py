from typing import Annotated, NamedTuple
import impy as ip
import polars as pl

from acryo import BatchLoader, Molecules
from macrokit import Symbol, Expr
from magicclass import (
    confirm,
    magicclass,
    do_not_record,
    set_design,
    MagicTemplate,
    field,
)
from magicclass.types import Bound, Path

from cylindra.const import (
    GlobalVariables as GVar,
    MoleculesHeader as Mole,
    PropertyNames as H,
)
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import FileFilter, POLARS_NAMESPACE
from cylindra.project import CylindraProject, CylindraBatchProject, get_project_json

from .sta import BatchSubtomogramAveraging
from ._sequence import ProjectSequenceEdit
from ._loaderlist import LoaderList, LoaderInfo

_SPLINE_FEATURES = [H.spacing, H.skew, H.npf, H.start, H.rise, H.radius, H.orientation]


@magicclass(
    widget_type="split",
    layout="horizontal",
    name="Batch Analysis",
    properties={"min_height": 400},
    symbol=Expr("getattr", [Symbol("ui"), "batch"]),
)
class CylindraBatchWidget(MagicTemplate):
    constructor = ProjectSequenceEdit
    sta = field(BatchSubtomogramAveraging)

    def __init__(self):
        self._loaders = LoaderList()
        self._loaders.events.inserted.connect(self.reset_choices)
        self._loaders.events.removed.connect(self.reset_choices)
        self._loaders.events.moved.connect(self.reset_choices)

    @constructor.wraps
    def construct_loader(
        self,
        paths: Bound[constructor._get_loader_paths],
        predicate: Bound[constructor._get_expression],
        name: Annotated[str, {"bind": constructor.seq_name}],
    ):
        """
        Construct a batch loader object from the given paths and predicate.

        Parameters
        ----------
        paths : list of (Path, list[Path]) or list of (Path, list[Path], Path)
            List of tuples of image path, list of molecule paths, and project path. The
            project path is optional.
        """
        if name == "":
            raise ValueError("Name is empty!")
        loader = BatchLoader()
        image_paths: dict[int, Path] = {}
        to_drop = set[str]()
        for img_id, _path_info in enumerate(paths):
            path_info = PathInfo(*_path_info)
            img = ip.lazy.imread(path_info.image, chunks=GVar.dask_chunk)
            image_paths[img_id] = Path(path_info.image)
            if path_info.project is not None:
                _converter = _molecule_to_spline_converter(path_info.project)
            else:
                _converter = lambda _: None
            for mole_path in path_info.molecules:
                mole_path = Path(mole_path)
                mole = Molecules.from_csv(mole_path)
                spl = _converter(mole_path)
                features = [pl.repeat(mole_path.stem, pl.count()).alias(Mole.id)]
                if spl is not None:
                    for propname in _SPLINE_FEATURES:
                        prop = spl.props.get_glob(propname, None)
                        if prop is None:
                            continue
                        propname_glob = propname + "_glob"
                        features.append(
                            pl.repeat(prop, pl.count()).alias(propname_glob)
                        )
                        to_drop.add(propname_glob)
                mole = mole.with_features(features)
                loader.add_tomogram(img.value, mole, img_id)

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = eval(predicate, POLARS_NAMESPACE, {})
            loader = loader.filter(predicate)
        new = loader.replace(
            molecules=loader.molecules.drop_features(to_drop),
            scale=self.constructor.scale.value,
        )
        self._add_loader(new, name, image_paths)
        return new

    def _add_loader(self, loader: BatchLoader, name: str, image_paths: dict[int, Path]):
        self._loaders.append(LoaderInfo(loader, name=name, image_paths=image_paths))

    @constructor.MacroMenu.wraps
    @do_not_record
    def show_macro(self):
        from cylindra import instance

        ui = instance()
        macro_str = self.macro.widget.textedit.value
        win = ui.macro.widget.new_window("Batch")
        win.textedit.value = macro_str
        win.show()
        CylindraMainWidget._active_widgets.add(win)
        return None

    @constructor.MacroMenu.wraps
    @do_not_record
    def show_native_macro(self):
        self.macro.widget.show()
        CylindraMainWidget._active_widgets.add(self.macro.widget)
        return None

    @constructor.File.wraps
    @set_design(text="Load batch analysis project")
    @confirm(
        text="Are you sure to clear all loaders?", condition="len(self._loaders) > 0"
    )
    def load_batch_project(self, path: Path.Read[FileFilter.PROJECT]):
        """
        Load a batch project from a JSON file.

        Parameters
        ----------
        path : path-like
            Path to the JSON file.
        """
        self._loaders.clear()
        return CylindraBatchProject.from_json(get_project_json(path)).to_gui(self)

    @constructor.File.wraps
    @set_design(text="Save as batch analysis project")
    def save_batch_project(self, save_path: Path.Save):
        """
        Save the GUI state to a JSON file.

        Parameters
        ----------
        save_path : path-like
            Path to the JSON file.
        """
        save_path = Path(save_path)
        json_path = save_path / "project.json"
        return CylindraBatchProject.save_gui(self, json_path, save_path)


class PathInfo(NamedTuple):
    image: Path
    molecules: list[Path]
    project: Path | None = None


def _molecule_to_spline_converter(path: Path):
    prj = CylindraProject.from_json(path)
    molecule_paths = [Path(_p) for _p in prj.molecules]

    def _converter(fp: Path):
        try:
            idx = molecule_paths.index(fp)
        except ValueError:
            return None
        if prj.molecules_info is None:
            return None
        source = prj.molecules_info[idx].source
        if source is None:
            return None
        return prj.load_spline(source)

    return _converter
