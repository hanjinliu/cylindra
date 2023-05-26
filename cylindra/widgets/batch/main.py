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

from cylindra.const import GlobalVariables as GVar, MoleculesHeader as Mole
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import FileFilter, POLARS_NAMESPACE
from cylindra.project import CylindraBatchProject, get_project_json

from .sta import BatchSubtomogramAveraging
from ._sequence import ProjectSequenceEdit
from ._loaderlist import LoaderList, LoaderInfo


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
        name: Bound[constructor.seq_name],
    ):
        """Construct a batch loader object from the given paths and predicate"""
        if name == "":
            raise ValueError("Name is empty!")
        loader = BatchLoader()
        image_paths: dict[int, Path] = {}
        for img_id, (img_path, mole_paths) in enumerate(paths):
            img = ip.lazy_imread(img_path, chunks=GVar.dask_chunk)
            image_paths[img_id] = Path(img_path)
            for mole_path in mole_paths:
                mole_path = Path(mole_path)
                mole = Molecules.from_csv(mole_path).with_features(
                    [pl.repeat(mole_path.stem, pl.count()).alias(Mole.id)]
                )
                loader.add_tomogram(img.value, mole, img_id)

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = eval(predicate, POLARS_NAMESPACE, {})
            loader = loader.filter(predicate)
        new = loader.replace(scale=self.constructor.scale.value)
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
