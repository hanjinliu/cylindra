from typing import Annotated, Literal, NamedTuple, Any, TYPE_CHECKING
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
from magicclass.types import Path

from cylindra.const import (
    MoleculesHeader as Mole,
    PropertyNames as H,
    FileFilter,
)
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import POLARS_NAMESPACE
from cylindra.project import CylindraProject, CylindraBatchProject
from cylindra._config import get_config
from .sta import BatchSubtomogramAveraging
from ._sequence import ProjectSequenceEdit
from ._loaderlist import LoaderList, LoaderInfo

if TYPE_CHECKING:
    from cylindra.components import CylSpline

_SPLINE_FEATURES = [
    H.spacing,
    H.dimer_twist,
    H.npf,
    H.start,
    H.rise,
    H.radius,
    H.orientation,
]


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

    @set_design(text="Construct loader", location=constructor)
    def construct_loader(
        self,
        paths: Annotated[Any, {"bind": constructor._get_loader_paths}],
        predicate: Annotated[str | pl.Expr | None, {"bind": constructor._get_expression}],
        name: str = "Loader",
    ):  # fmt: skip
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
            img = ip.lazy.imread(path_info.image, chunks=get_config().dask_chunk)
            image_paths[img_id] = Path(path_info.image)
            prj = CylindraProject.from_file(path_info.project)
            with prj.open_project() as dir:
                for mole_path in path_info.molecules:
                    mole_abs_path = dir / mole_path
                    mole = Molecules.from_file(mole_abs_path)
                    spl = _find_source(prj, dir, mole_path)
                    features = [
                        pl.repeat(mole_abs_path.stem, pl.count()).alias(Mole.id)
                    ]
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

    @set_design(text="Show macro", location=constructor.MacroMenu)
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

    @set_design(text="Show native macro", location=constructor.MacroMenu)
    @do_not_record
    def show_native_macro(self):
        self.macro.widget.show()
        CylindraMainWidget._active_widgets.add(self.macro.widget)
        return None

    @set_design(text="Load batch analysis project", location=constructor.File)
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
        return CylindraBatchProject.from_file(path)._to_gui(self)

    @set_design(text="Save as batch analysis project", location=constructor.File)
    def save_batch_project(
        self,
        save_path: Path.Save,
        molecules_ext: Literal[".csv", ".parquet"] = ".csv",
    ):
        """
        Save the GUI state to a JSON file.

        Parameters
        ----------
        save_path : path-like
            Path to the JSON file.
        molecules_ext : str, default is ".csv"
            Extension of the molecule files.
        """
        return CylindraBatchProject.save_gui(self, Path(save_path), molecules_ext)


class PathInfo(NamedTuple):
    image: Path
    molecules: list[str]
    project: Path


def _find_source(prj: CylindraProject, dir: Path, mole_path: str) -> "CylSpline | None":
    for info in prj.molecules_info:
        if info.name == mole_path:
            source = info.source
            if source is None:
                return None
            return prj.load_spline(dir, source)
    return None
