from typing import Iterator
from magicclass import (
    magicclass, field, vfield, MagicTemplate, set_design, abstractapi
)
from magicclass.widgets import EvalLineEdit, ComboBox, Container
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
from acryo import BatchLoader, Molecules

import numpy as np
import impy as ip
import polars as pl

from cylindra.project import CylindraProject, ProjectSequence
from cylindra.const import GlobalVariables as GVar, nm, MoleculesHeader as Mole


@magicclass(widget_type="frame", record=False, properties={"margins": (0, 0, 0, 0)})
class Project(MagicTemplate):
    # a widget representing a single project
    check = vfield(True).with_options(text="")
    path = field("").with_options(enabled=False)
    
    @set_design(text="✕", max_width=30)
    def remove_project(self):
        """Remove this project from the list."""
        parent = self.find_ancestor(ProjectPaths)
        idx = parent.index(self)
        del parent[idx]
        del parent._project_sequence[idx]
    
    @set_design(text="Open")
    def send_to_viewer(self):
        """Send this project to the viewer."""
        from cylindra.core import instance
        
        if ui := instance():
            ui.load_project(self.path.value)
        else:
            raise ValueError("No Cylindra widget found!")

    def __post_init__(self):
        self["check"].text = ""  # NOTE: should be updated here!

    @classmethod
    def _from_path(cls, path: Path):
        self = cls(layout="horizontal", labels=False)
        self.path.value = str(path)
        self.path.tooltip = self.path
        self.margins = (0, 0, 0, 0)
        return self

@magicclass(widget_type="scrollable", labels=False, record=False, properties={"min_height": 20})
class ProjectPaths(MagicTemplate):
    def __init__(self):
        self._project_sequence = ProjectSequence()

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
        preview_components = abstractapi()
        preview_features = abstractapi()

    projects = field(ProjectPaths)
    
    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)}, record=False)
    class FilterExpr(MagicTemplate):
        filter_expression = field(str, label="Filter:", widget_type=EvalLineEdit).with_options(namespace={"pl": pl})
        preview_filtered_molecules = abstractapi()
    
    @Buttons.wraps
    @set_design(text="✓", max_width=26)
    def check_all(self):
        """Check all projects."""
        for wdt in self.projects:
            wdt.check = True
    
    
    def _get_project_paths(self, w=None) -> list[Path]:
        return self.projects.paths

    @Buttons.wraps
    @set_design(text="Preview components")
    def preview_components(self):
        """Preview all the splines and molecules that exist in this project."""
        from cylindra.project import ComponentsViewer
        cbox = ComboBox(choices=self._get_project_paths)
        comp_viewer = ComponentsViewer()
        
        self.changed.connect(lambda: cbox.reset_choices())
        cbox.changed.connect(
            lambda path: comp_viewer._from_project(CylindraProject.from_json(path))
        )
        cont = Container(widgets=[cbox, comp_viewer], labels=False)
        cont.native.setParent(self.native, cont.native.windowFlags())
        cont.show()
        cbox.changed.emit(cbox.value)
        return None
    
    @Buttons.wraps
    @set_design(text="Preview table")
    def preview_features(self):
        col = get_batch_loader(self.projects.paths, predicate=None)
        df = col.molecules.to_dataframe()
        table = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features", area="left")
        dock.setFloating(True)

    @FilterExpr.wraps
    @set_design(text="Preview", max_width=52)
    def preview_filtered_molecules(self):
        """Preview filtered molecules."""
        col = get_batch_loader(self.projects.paths, predicate=self._get_expression())
        df = col.molecules.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features (filtered)", area="left")
        dock.setFloating(True)
        
    def _get_expression(self, w=None) -> pl.Expr:
        wdt = self.FilterExpr.filter_expression
        if wdt.value == "":
            return None
        return wdt.eval()
    
    def _get_dummy_loader(self):
        paths = self.projects.paths
        if len(paths) == 0:
            raise ValueError("No projects found.")
        loader = get_batch_loader(paths, order=1, load=False)
        return loader

def get_batch_loader(
    project_paths: list[Path], 
    order: int = 3, 
    output_shape: tuple[nm, nm, nm] = None,
    predicate: "str | pl.Expr | None" = None,
    load: bool = True,
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
        col = BatchLoader(order=order)
    else:
        output_shape = tuple(np.round(np.array(output_shape) / scale).astype(int))
        col = BatchLoader(order=order, output_shape=output_shape, scale=scale)
    
    if load:
        for idx, project in enumerate(projects):
            tomogram = ip.lazy_imread(project.image, chunks=GVar.daskChunk)

            for fp in project.molecules:
                fp = Path(fp)
                mole = Molecules.from_csv(fp)
                mole.features = mole.features.with_columns(
                    [pl.repeat(fp.stem, pl.count()).alias(Mole.id)]
                )
                col.add_tomogram(tomogram.value, molecules=mole, image_id=idx)
    if predicate is not None:
        col = col.filter(predicate)
    return col
