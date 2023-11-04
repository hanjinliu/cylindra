from typing import Annotated, Literal, NamedTuple, Any, TYPE_CHECKING
import impy as ip
import polars as pl

from acryo import BatchLoader, Molecules
from macrokit import Symbol, Expr
from magicclass import (
    confirm,
    get_button,
    magicclass,
    do_not_record,
    set_design,
    MagicTemplate,
    field,
)
from magicclass.types import Path
from magicclass.utils import thread_worker
from cylindra.const import (
    MoleculesHeader as Mole,
    PropertyNames as H,
    FileFilter,
)
from cylindra.core import ACTIVE_WIDGETS
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
    H.twist,
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
    constructor = field(ProjectSequenceEdit)
    sta = field(BatchSubtomogramAveraging)

    def __init__(self):
        self._loaders = LoaderList()
        self._loaders.events.inserted.connect(self.reset_choices)
        self._loaders.events.removed.connect(self.reset_choices)
        self._loaders.events.moved.connect(self.reset_choices)

    def _get_loader_paths(self, *_):
        return [prj._get_loader_paths() for prj in self.constructor.projects]

    def _get_expression(self, *_):
        return self.constructor._get_expression()

    @set_design(text="Construct loader", location=constructor)
    @thread_worker
    def construct_loader(
        self,
        paths: Annotated[Any, {"bind": _get_loader_paths}],
        predicate: Annotated[str | pl.Expr | None, {"bind": _get_expression}],
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

        yield 0.0, 0.0
        loader = BatchLoader()
        image_paths: dict[int, Path] = {}
        to_drop = set[str]()
        _total = len(paths)
        for img_id, _path_info in enumerate(paths):
            path_info = PathInfo(*_path_info)
            img = ip.lazy.imread(path_info.image, chunks=get_config().dask_chunk)
            image_paths[img_id] = Path(path_info.image)
            prj = CylindraProject.from_file(path_info.project)
            with prj.open_project() as dir:
                _total_sub = len(path_info.molecules)
                for molecule_id, mole_path in enumerate(path_info.molecules):
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
                    yield img_id / _total, molecule_id / _total_sub
                yield (img_id + 1) / _total, 0.0
        yield 1.0, 1.0

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = eval(predicate, POLARS_NAMESPACE, {})
            loader = loader.filter(predicate)
        new = loader.replace(
            molecules=loader.molecules.drop_features(to_drop),
            scale=self.constructor.scale.value,
        )

        @thread_worker.callback
        def _on_return():
            self._add_loader(new, name, image_paths)

        return _on_return

    @construct_loader.yielded.connect
    def _on_construct_loader_yielded(self, prog: tuple[float, float]):
        btn = get_button(self.construct_loader, cache=True)
        btn.text = f"Constructing... ({prog[0]:.1%}, {prog[1]:.1%})"

    @construct_loader.finished.connect
    def _on_construct_loader_finished(self):
        btn = get_button(self.construct_loader, cache=True)
        btn.text = "Construct loader"

    def _add_loader(self, loader: BatchLoader, name: str, image_paths: dict[int, Path]):
        self._loaders.append(LoaderInfo(loader, name=name, image_paths=image_paths))

    @set_design(text="Show macro", location=ProjectSequenceEdit.MacroMenu)
    @do_not_record
    def show_macro(self):
        from cylindra import instance

        ui = instance()
        macro_str = self.macro.widget.textedit.value
        win = ui.macro.widget.new_window("Batch")
        win.textedit.value = macro_str
        win.show()
        ACTIVE_WIDGETS.add(win)
        return None

    @set_design(text="Show native macro", location=ProjectSequenceEdit.MacroMenu)
    @do_not_record
    def show_native_macro(self):
        self.macro.widget.show()
        ACTIVE_WIDGETS.add(self.macro.widget)
        return None

    @set_design(text="Load batch analysis project", location=ProjectSequenceEdit.File)
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

    @set_design(
        text="Save as batch analysis project", location=ProjectSequenceEdit.File
    )
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
