from typing import Iterator, Union, TYPE_CHECKING, Annotated
from timeit import default_timer
from magicclass import (
    magicclass, field, vfield, MagicTemplate, set_design, abstractapi, FieldGroup
)
from magicgui.widgets import Table, PushButton
from magicclass.types import Path
from acryo import TomogramCollection, Molecules

import numpy as np
import impy as ip
import polars as pl

from cylindra.project import CylindraProject
from cylindra.const import GlobalVariables as GVar, nm, MoleculesHeader as Mole


@magicclass(widget_type="frame", record=False, properties={"margins": (0, 0, 0, 0)})
class Project(MagicTemplate):
    # a widget representing a single project
    check = vfield(True, name="").with_options(text="")
    path = field("").with_options(enabled=False)
    
    @set_design(text="✕", max_width=30)
    def remove_project(self):
        """Remove this project from the list."""
        parent =self.find_ancestor(ProjectPaths)
        parent.remove(self)

    @classmethod
    def _from_path(cls, path: Path):
        self = cls(layout="horizontal", labels=False)
        self.path.value = str(path)
        self.path.tooltip = self.path
        self.margins = (0, 0, 0, 0)
        return self

@magicclass(widget_type="scrollable", labels=False, record=False, properties={"min_height": 20})
class ProjectPaths(MagicTemplate):
    def _add(self, path: Path):
        prj = Project._from_path(path)
        self.append(prj)
        self.min_height = min(len(self) * 50, 165)
    
    def __iter__(self) -> Iterator[Project]:
        return super().__iter__()

    @property
    def checked_indices(self) -> list[int]:
        indices: list[int] = []
        for i, wdt in enumerate(self):
            if wdt.check:
                indices.append(i)
        return indices
            
    @property
    def paths(self) -> list[Path]:
        return [Path(wdt.path.value) for wdt in self]



@magicclass(widget_type="collapsible", name="Projects", record=False)
class ProjectSequenceEdit(MagicTemplate):
    """
    Attributes
    ----------
    filter_expression : str
        A `polars` expression to filter molecules. e.g. `pl.col("score") > 0.5`
    """

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class Buttons(MagicTemplate):
        check_all = abstractapi()
        preview_all = abstractapi()

    projects = field(ProjectPaths)
    
    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class FilterExpr(MagicTemplate):
        filter_expression = vfield(str, label="Filter:")
        preview_filtered_molecules = abstractapi()
    
    @Buttons.wraps
    @set_design(text="✓", max_width=26)
    def check_all(self):
        """Check all projects."""
        for wdt in self.projects:
            wdt.check = True
    
    @FilterExpr.wraps
    @set_design(text="Preview", max_width=52)
    def preview_filtered_molecules(self):
        """Preview filtered molecules."""
        col = get_collection(self.projects.paths, predicate=self._get_expression())
        df = col.molecules.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = Table(value=df.to_pandas())
        table.read_only = True
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features", area="left")
        dock.setFloating(True)
        
    def _get_expression(self, w=None) -> pl.Expr:
        expr = self.FilterExpr.filter_expression
        if expr == "":
            return None
        return eval(expr, {"pl": pl}, {})

def get_collection(
    project_paths: list[Path], 
    order: int = 3, 
    output_shape: tuple[nm, nm, nm] = None,
    predicate: "str | pl.Expr | None" = None,
):
    # check scales
    projects: list[CylindraProject] = []
    scales: list[nm] = []
    for path in project_paths:
        project = CylindraProject.from_json(path)
        projects.append(project)
        scales.append(project.scale)
    scale = scales[0]
    if len(set(scales)) > 1:
        raise ValueError("All projects must have the same scale.")

    if output_shape is None:
        col = TomogramCollection(order=order)
    else:
        output_shape = tuple(np.round(np.array(output_shape) / scale).astype(int))
        col = TomogramCollection(order=order, output_shape=output_shape, scale=scale)
        
    for idx, project in enumerate(projects):
        tomogram = ip.lazy_imread(project.image, chunks=GVar.daskChunk)

        for fp in project.molecules:
            fp = Path(fp)
            mole = Molecules.from_csv(fp)
            mole.features = mole.features.with_columns(pl.Series(Mole.id, np.full(len(mole), fp.stem)))
            col.add_tomogram(tomogram.value, molecules=mole, image_id=idx)
    if predicate is not None:
        col = col.filter(predicate)
    return col
