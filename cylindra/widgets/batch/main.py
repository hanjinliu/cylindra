from typing import Annotated, TYPE_CHECKING, Any

import numpy as np
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
from magicclass.types import Bound, Path, Optional

from cylindra.const import GlobalVariables as GVar, MoleculesHeader as Mole
from cylindra.widgets.widget_utils import FileFilter
from cylindra.project import CylindraBatchProject

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
            img = ip.lazy_imread(img_path, chunks=GVar.daskChunk)
            image_paths[img_id] = Path(img_path)
            for mole_path in mole_paths:
                mole_path = Path(mole_path)
                mole = Molecules.from_csv(mole_path).with_features(
                    [pl.repeat(mole_path.stem, pl.count()).alias(Mole.id)]
                )
                loader.add_tomogram(img.value, mole, img_id)

        if predicate is not None:
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
        return None

    @constructor.MacroMenu.wraps
    @do_not_record
    def show_native_macro(self):
        self.macro.widget.show()
        return None

    @constructor.File.wraps
    @set_design(text="Load batch analysis project")
    @confirm(
        text="Are you sure to clear all loaders?", condition="len(self._loaders) > 0"
    )
    def load_batch_project(self, path: Path.Read[FileFilter.JSON]):
        """
        Load a batch project from a JSON file.

        Parameters
        ----------
        path : path-like
            Path to the JSON file.
        """
        self._loaders.clear()
        return CylindraBatchProject.from_json(path).to_gui(self)

    @constructor.File.wraps
    @set_design(text="Save as batch analysis project")
    def save_batch_project(
        self,
        json_path: Path.Save[FileFilter.JSON],
        results_dir: Annotated[
            Optional[Path.Dir], {"text": "Save at the same directory"}
        ] = None,
    ):
        """
        Save the GUI state to a JSON file.

        Parameters
        ----------
        json_path : path-like
            Path to the JSON file.
        results_dir : path-like, optional
            If given, results will be saved to this directory.
        """
        return CylindraBatchProject.save_gui(self, Path(json_path), results_dir)
