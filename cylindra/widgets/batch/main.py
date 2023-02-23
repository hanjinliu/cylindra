from typing import Annotated, TYPE_CHECKING, Any

import numpy as np
import impy as ip
import polars as pl

from acryo import BatchLoader, Molecules
from macrokit import Symbol, Expr
from magicclass import magicclass, do_not_record, field, nogui, MagicTemplate
from magicclass.types import Bound

from cylindra.project import ProjectSequence
from cylindra.const import GlobalVariables as GVar, MoleculesHeader as Mole

from .menus import Macro
from .sta import BatchSubtomogramAveraging
from ._sequence import ProjectSequenceEdit
from ._loaderlist import LoaderList, LoaderInfo


@magicclass(
    widget_type="split",
    layout="horizontal",
    name="Batch Analysis",
    properties={"min_height": 240},
    symbol=Expr("getattr", [Symbol("ui"), "batch"]),
)
class CylindraBatchWidget(MagicTemplate):
    
    # Menus
    MacroMenu = field(Macro, name="Macro")

    constructor = ProjectSequenceEdit
    sta = BatchSubtomogramAveraging
    
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
        if name == "":
            raise ValueError("Name is empty!")
        loader = BatchLoader()
        scales: list[float] = []
        for img_id, (img_path, mole_paths) in enumerate(paths):
            img = ip.lazy_imread(img_path, chunks=GVar.daskChunk)
            scales.append(img.scale.x)
            for mole_path in mole_paths:
                mole = Molecules.from_csv(mole_path).with_features(
                    [pl.repeat(mole_path.stem, pl.count()).alias(Mole.id)]
                )
                loader.add_tomogram(img.value, mole, img_id)
            
        if abs(max(scales) / min(scales) - 1) > 0.01:
            raise ValueError("Scale error must be less than 1%.")
        if predicate is not None:
            loader = loader.filter(predicate)
        new = loader.replace(scale=np.mean(scales))
        self._loaders.append(
            LoaderInfo(new, name=name, paths=paths, predicate=predicate, parent=None)
        )
        return new

    @nogui
    @do_not_record
    def set_sequence(self, col: ProjectSequence):
        if not isinstance(col, ProjectSequence):
            raise TypeError(f"Expected a ProjectCollection, got {type(col)}")
        for prj in col:
            self.constructor.projects._add(prj.project_path)
        self.reset_choices()

    @MacroMenu.wraps
    @do_not_record
    def show_macro(self):
        from cylindra import instance
        ui = instance()
        macro_str = self.macro.widget.textedit.value
        win = ui.macro.widget.new_window("Batch")
        win.textedit.value = macro_str
        win.show()
        return None
    
    @MacroMenu.wraps
    @do_not_record
    def show_native_macro(self):
        self.macro.widget.show()
        return None
