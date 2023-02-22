from typing import Iterator
from fnmatch import fnmatch
from magicgui.widgets import ComboBox, Container
from magicclass import (
    magicclass, field, magicmenu, nogui, vfield, MagicTemplate, set_design, abstractapi
)
from magicclass.widgets import EvalLineEdit
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
from acryo import BatchLoader, Molecules, SubtomogramLoader

import numpy as np
import impy as ip
import polars as pl

from cylindra.project import CylindraProject, ProjectSequence
from cylindra.const import GlobalVariables as GVar, nm, MoleculesHeader as Mole, IDName
from cylindra.widgets.widget_utils import FileFilter
from ._localprops import LocalPropsViewer

@magicclass(labels=False, properties={"margins": (0, 0, 0, 0)}, record=False, layout="horizontal")
class MoleculeWidget(MagicTemplate):
    check = vfield(True).with_options(text="")
    line = field("").with_options(enabled=False)
    
    def _get_molecules(self) -> Molecules:
        fp = Path(self.line.value)
        mole = Molecules.from_csv(self.line.value)
        return mole.with_features([pl.repeat(fp.stem, pl.count()).alias(Mole.id)])

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
    def __iter__(self) -> Iterator[SplineWidget]:
        return super().__iter__()

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
    
    @Header.check.connect
    def _on_checked(self, value: bool):
        self.splines.enabled = value
        self.molecules.enabled = value
    
    @magicclass(widget_type="groupbox", name="Components")
    class Components(MagicTemplate):
        pass
    
    splines = Components.field(SplineList)
    molecules = Components.field(MoleculeList)

    @classmethod
    def _from_path(cls, path: Path):
        """Create a Project widget from a project path."""
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
            ip.lazy_imread(project.image, chunks=GVar.daskChunk).value,
            molecules=Molecules.concat(molecules),
            order=order,
            scale=project.scale,
        )
    
    def _get_loader_paths(self) -> tuple[Path, list[Path]]:
        path = self.Header.path.value
        project = CylindraProject.from_json(path)
        img_path = Path(project.image)
        mole_paths = [Path(mole.line.value) for mole in self.molecules if mole.check]
        return img_path, mole_paths
    
    def _get_localprops(self) -> pl.DataFrame:
        path = self.Header.path.value
        project = CylindraProject.from_json(path)
        if project.localprops is None:
            raise ValueError("No localprops file found.")
        
        df = pl.read_csv(project.localprops)
        checked = [spl.check for spl in self.splines]
        return df.filter(pl.col(IDName.spline).is_in(checked))


@magicclass(widget_type="scrollable", labels=False, record=False, properties={"min_height": 20})
class ProjectPaths(MagicTemplate):
    def _add(self, path: Path):
        prj = Project._from_path(path)
        self.append(prj)
        self.min_height = min(len(self) * 100, 280)
    
    def __iter__(self) -> Iterator[Project]:
        return super().__iter__()

    @property
    def paths(self) -> list[Path]:
        return [Path(wdt.Header.path.value) for wdt in self]

    
@magicclass(name="_Projects", record=False)
class ProjectSequenceEdit(MagicTemplate):
    """
    Attributes
    ----------
    filter_expression : str
        A `polars` expression to filter molecules. e.g. `pl.col("score") > 0.5`
    """

    @magicmenu
    class File(MagicTemplate):
        add_children = abstractapi()
        add_children_glob = abstractapi()
    
    @magicmenu
    class Select(MagicTemplate):
        select_all_projects = abstractapi()
        select_projects_by_pattern = abstractapi()
        select_molecules_by_pattern = abstractapi()
    
    @magicmenu
    class View(MagicTemplate):
        view_components = abstractapi()
        view_selected_components = abstractapi()
        view_molecules = abstractapi()
        view_filtered_molecules = abstractapi()
        view_localprops = abstractapi()

    projects = field(ProjectPaths)
    
    filter_expression = field(str, label="Filter:", widget_type=EvalLineEdit).with_options(namespace={"pl": pl})
    
    @Select.wraps
    def select_all_projects(self):
        """Select all projects."""
        for wdt in self.projects:
            wdt.Header.check = True
    
    @Select.wraps
    def select_projects_by_pattern(self, pattern: str):
        """Select projects by pattern matching."""
        for prj in self.projects:
            prj.Header.check = fnmatch(prj.Header.path.value, pattern)
    
    @Select.wraps
    def select_molecules_by_pattern(self, pattern: str):
        """Select molecules by pattern matching."""
        for prj in self.projects:
            for mole in prj.molecules:
                mole.check = fnmatch(mole.line.value, pattern)
    
    def _get_project_paths(self, _=None) -> list[Path]:
        return [wdt.Header.path.value for wdt in self.projects]
    
    def _get_selected_project_paths(self, _=None) -> list[Path]:
        return [prj.Header.path.value for prj in self._iter_selected_projects()]
    
    def _iter_selected_projects(self) -> Iterator[Project]:
        for prj in self.projects:
            if prj.Header.check:
                yield prj
    
    def _get_batch_loader(self, order: int = 3, output_shape=None, predicate=None) -> BatchLoader:
        batch_loader = BatchLoader()
        for i, prj in enumerate(iter(self.projects)):
            if not prj.Header.check:
                continue
            loader = prj.get_loader(order=order)
            batch_loader.add_tomogram(loader.image, loader.molecules, image_id=i)

        if predicate is not None:
            batch_loader = batch_loader.filter(predicate)
        if output_shape is not None:
            batch_loader = batch_loader.replace(output_shape=output_shape)
        return batch_loader
    
    def _get_loader_paths(self, _=None) -> list[tuple[Path, list[Path]]]:
        return [prj._get_loader_paths() for prj in self.projects]
    
    def _get_localprops(self) -> pl.DataFrame:
        dataframes: list[pl.DataFrame] = []
        for idx, prj in enumerate(iter(self.projects)):
            df = prj._get_localprops()
            dataframes.append(
                df.with_columns(pl.repeat(idx, pl.count()).cast(pl.UInt16).alias(Mole.image))
            )
        return pl.concat(dataframes, how="diagonal")

    @View.wraps
    @set_design(text="View components in 3D")
    def view_components(self):
        """View all the splines and molecules that exist in this project."""
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

    
    @View.wraps
    @set_design(text="View selected components in 3D")
    def view_selected_components(self):
        from cylindra.project import ComponentsViewer
        cbox = ComboBox(choices=self._get_selected_project_paths)
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
    
    
    @View.wraps
    @set_design(text="View selected molecules in table")
    def view_molecules(self):
        mole = self._get_batch_loader().molecules
        df = mole.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features (filtered)", area="left")
        dock.setFloating(True)

    @View.wraps
    @set_design(text="View filtered molecules in table")
    def view_filtered_molecules(self):
        """Preview filtered molecules."""
        mole = self._get_batch_loader(predicate=self._get_expression()).molecules
        df = mole.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(table, name="Features (filtered)", area="left")
        dock.setFloating(True)
    
    @View.wraps
    @set_design(text="View local properties")
    def view_localprops(self):
        """View local properties of splines."""
        wdt = LocalPropsViewer()
        wdt.native.setParent(self.native, wdt.native.windowFlags())
        wdt.show()
        wdt._set_localprops(self._get_localprops())
        return
        
    def _get_expression(self, _=None) -> pl.Expr:
        wdt = self.filter_expression
        if wdt.value == "":
            return None
        return wdt.eval()

    @File.wraps
    @set_design(text="Add projects")
    def add_children(self, paths: Path.Multiple[FileFilter.JSON]):
        """Add project json files as the child projects."""
        for path in paths:
            self.projects._add(path)
        self.reset_choices()
        return 
    
    @File.wraps
    @set_design(text="Add projects with wildcard path")
    def add_children_glob(self, pattern: str):
        """Add project json files using wildcard path."""
        import glob

        pattern = str(pattern)
        for path in glob.glob(pattern):
            self.projects._add(path)
        self.reset_choices()
        return 

    construct_loader = abstractapi()

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
