from typing import Iterator

from magicgui.widgets import ComboBox, Container
from magicclass import (
    magicclass, field, nogui, vfield, MagicTemplate, set_design, abstractapi
)
from magicclass.widgets import EvalLineEdit
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
from acryo import BatchLoader, Molecules, SubtomogramLoader

import numpy as np
import impy as ip
import polars as pl

from cylindra.project import CylindraProject, ProjectSequence
from cylindra.const import GlobalVariables as GVar, nm, MoleculesHeader as Mole

@magicclass(labels=False, properties={"margins": (0, 0, 0, 0)}, record=False, layout="horizontal")
class MoleculeWidget(MagicTemplate):
    check = vfield(True).with_options(text="")
    line = field("").with_options(enabled=False)
    
    def _get_molecules(self) -> Molecules:
        return Molecules.from_csv(self.line.value)

@magicclass(widget_type="collapsible", record=False, name="Molecules")
class MoleculeList(MagicTemplate):
    def __iter__(self) -> Iterator[MoleculeWidget]:
        return super().__iter__()

    def _add_path(self, path: Path):
        wdt = MoleculeWidget()
        wdt.line.value = str(path)
        wdt["check"].text = ""
        self.append(wdt)

@magicclass(labels=False, record=False, properties={"margins": (0, 0, 0, 0)}, layout="horizontal")
class SplineWidget(MagicTemplate):
    check = vfield(True).with_options(text="")
    line = field("").with_options(enabled=False)

@magicclass(widget_type="collapsible", record=False, name="Splines")
class SplineList(MagicTemplate):
    def _add_path(self, path: Path):
        wdt = SplineWidget()
        wdt.line.value = str(path)
        wdt["check"].text = ""
        self.append(wdt)


@magicclass(widget_type="frame", labels=False, record=False, properties={"margins": (0, 0, 0, 0)})
class Project(MagicTemplate):
    # a widget representing a single project
    @magicclass(layout="horizontal", labels=False, properties={"margins": (0, 0, 0, 0)}, record=False)
    class Header(MagicTemplate):
        """
        Project info.

        Attributes
        ----------
        check : bool
            Whether to include this project in the batch processing.
        path : str
            Project path.
        """
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
    
    @magicclass(widget_type="groupbox", name="Components")
    class Components(MagicTemplate):
        pass
    
    splines = Components.field(SplineList)
    molecules = Components.field(MoleculeList)

    @classmethod
    def _from_path(cls, path: Path):
        self = cls()
        path = str(path)
        self.Header.path.value = path
        self.Header.path.tooltip = path
        
        project = CylindraProject.from_json(path)
        
        # load splines
        for spline_path in project.splines:
            self.splines._add_path(spline_path)
        
        # load molecules
        for mole_path in project.molecules:
            self.molecules._add_path(mole_path)

        self.margins = (0, 0, 0, 0)
        return self
    
    @nogui
    def get_loader(self, order: int = 3) -> SubtomogramLoader:
        path = self.Header.path.value
        project = CylindraProject.from_json(path)
        molecules = [mole._get_molecules() for mole in self.molecules if mole.check]
        return SubtomogramLoader(
            ip.lazy_imread(project.image).value,
            molecules=Molecules.concat(molecules),
            order=order,
            scale=project.scale,
        )

@magicclass(widget_type="scrollable", labels=False, record=False, properties={"min_height": 20})
class ProjectPaths(MagicTemplate):
    def __init__(self):
        self._project_sequence = ProjectSequence()

    def _add(self, path: Path):
        prj = Project._from_path(path)
        self.append(prj)
        self.min_height = min(len(self) * 100, 280)
    
    def __iter__(self) -> Iterator[Project]:
        return super().__iter__()

    @property
    def checked_indices(self) -> list[int]:
        indices: list[int] = []
        for i, wdt in enumerate(iter(self)):
            if wdt.Header.check:
                indices.append(i)
        return indices
            
    @property
    def paths(self) -> list[Path]:
        return [Path(wdt.Header.path.value) for wdt in self]



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
            wdt.Header.check = True
    
    def _get_project_paths(self, w=None) -> list[Path]:
        return self.projects.paths
    
    def _get_batch_loader(self, order: int, output_shape=None, predicate=None) -> BatchLoader:
        loaders: list[SubtomogramLoader] = []
        for prj in self.projects:
            loader = prj.get_loader(order=order)
            loaders.append(loader)
        if len(set(ldr.scale for ldr in loaders)) > 1:
            raise ValueError("All projects must have the same scale!")
        batch_loader = BatchLoader.from_loaders(
            loaders, order=order, scale=loaders[0].scale
        )
        if predicate is not None:
            batch_loader = batch_loader.filter(predicate)
        if output_shape is not None:
            batch_loader = batch_loader.replace(output_shape=output_shape)
        return batch_loader

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
        loader = get_batch_loader(self.projects.paths, predicate=None)
        df = loader.molecules.to_dataframe()
        table = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features", area="left")
        dock.setFloating(True)

    @FilterExpr.wraps
    @set_design(text="Preview", max_width=52)
    def preview_filtered_molecules(self):
        """Preview filtered molecules."""
        loader = get_batch_loader(self.projects.paths, predicate=self._get_expression())
        df = loader.molecules.to_dataframe()
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
