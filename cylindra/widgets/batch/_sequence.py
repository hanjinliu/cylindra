from typing import Iterator
from fnmatch import fnmatch
import glob

from magicgui.widgets import ComboBox, Container, Widget
from magicclass import (
    do_not_record,
    impl_preview,
    magicclass,
    field,
    magicmenu,
    nogui,
    vfield,
    MagicTemplate,
    set_design,
    abstractapi,
)
from magicclass.types import Path, ExprStr
from magicclass.widgets import Separator, ConsoleTextEdit
from magicclass.ext.polars import DataFrameView
from acryo import BatchLoader, Molecules, SubtomogramLoader

import impy as ip
import polars as pl

from cylindra.project import CylindraProject, get_project_file
from cylindra.const import MoleculesHeader as Mole, FileFilter
from cylindra.widgets import CylindraMainWidget
from cylindra._config import get_config
from ._localprops import LocalPropsViewer


@magicclass(
    labels=False,
    properties={"margins": (0, 0, 0, 0)},
    record=False,
    layout="horizontal",
)
class MoleculeWidget(MagicTemplate):
    check = vfield(True).with_options(text="")
    line = field("").with_options(enabled=False)


@magicclass(widget_type="collapsible", record=False, name="Molecules")
class MoleculeList(MagicTemplate):
    def __iter__(self) -> Iterator[MoleculeWidget]:
        return super().__iter__()

    def _add_path(self, path: str):
        wdt = MoleculeWidget()
        wdt.line.value = path
        wdt["check"].text = ""
        self.append(wdt)


@magicclass(
    labels=False,
    record=False,
    properties={"margins": (0, 0, 0, 0)},
    layout="horizontal",
)
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


@magicclass(
    widget_type="frame",
    labels=False,
    record=False,
    properties={"margins": (0, 0, 0, 0)},
)
class Project(MagicTemplate):
    # a widget representing a single project
    @magicclass(
        layout="horizontal",
        labels=False,
        properties={"margins": (0, 0, 0, 0)},
        record=False,
    )
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

        remove_project = abstractapi()

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

    def __init__(self, project: "CylindraProject | None" = None):
        self._project = project

    @property
    def project(self) -> "CylindraProject | None":
        """The project model."""
        return self._project

    @property
    def path(self) -> str:
        """The project path."""
        return self.Header.path.value

    @property
    def check(self) -> bool:
        """True if the project is checked."""
        return self.Header.check

    @check.setter
    def check(self, value: bool):
        self.Header.check = value

    @Header.wraps
    @set_design(text="✕", max_width=30)
    def remove_project(self):
        """Remove this project from the list."""
        parent = self.find_ancestor(ProjectPaths)
        idx = parent.index(self)
        del parent[idx]

    @Header.check.connect
    def _on_checked(self, value: bool):
        self.splines.enabled = value
        self.molecules.enabled = value

    @magicclass(
        widget_type="collapsible",
        name="Components",
        record=False,
        properties={"margins": (12, 0, 0, 0)},
    )
    class Components(MagicTemplate):
        """List of components (molecules and/or splines)."""

        splines = abstractapi()
        molecules = abstractapi()

    splines = Components.field(SplineList)
    molecules = Components.field(MoleculeList)

    @classmethod
    def _from_path(cls, path: Path):
        """Create a Project widget from a project path."""
        path = str(path)
        project = CylindraProject.from_file(path)
        self = cls(project)
        self.Header.path.value = path
        self.Header.path.tooltip = path

        # load splines
        for _, spline_path in project.iter_spline_paths():
            self.splines._add_path(spline_path.name)

        # load molecules
        for info in project.molecules_info:
            self.molecules._add_path(info.name)

        return self

    @nogui
    @do_not_record
    def get_loader(self, order: int = 3) -> SubtomogramLoader:
        project = CylindraProject.from_file(self.path)
        with project.open_project() as dir:
            molecules = [
                Molecules.from_file(dir / mole.line.value)
                for mole in self.molecules
                if mole.check
            ]
        return SubtomogramLoader(
            ip.lazy.imread(project.image, chunks=get_config().dask_chunk).value,
            molecules=Molecules.concat(molecules),
            order=order,
            scale=project.scale,
        )

    def _get_loader_paths(self) -> tuple[Path, list[str], Path]:
        project = CylindraProject.from_file(self.path)
        img_path = Path(project.image)
        prj_path = Path(self.path)
        mole_paths = [mole.line.value for mole in self.molecules if mole.check]
        return img_path, mole_paths, prj_path

    def _get_localprops(self) -> pl.DataFrame:
        project = CylindraProject.from_file(self.path)
        with project.open_project() as dir:
            localprops_path = project.localprops_path(dir)
            if not localprops_path.exists():
                raise ValueError("No localprops file found.")

            df = pl.read_csv(localprops_path)
        return df


@magicclass(
    widget_type="scrollable",
    labels=False,
    record=False,
    properties={"min_height": 20, "min_width": 250},
)
class ProjectPaths(MagicTemplate):
    def _add(self, path: Path) -> Project:
        prj = Project._from_path(path)
        self.append(prj)
        return prj

    def __iter__(self) -> Iterator[Project]:
        return super().__iter__()

    @property
    def paths(self) -> list[Path]:
        return [Path(wdt.path) for wdt in self]


@magicclass(name="Projects", record=False)
class ProjectSequenceEdit(MagicTemplate):
    """
    Attributes
    ----------
    filter_expression : str
        A `polars` expression to filter molecules. e.g. `pl.col("score") > 0.5`.
        Spline global properties are also available during filtering, with suffix
        "_glob". e.g. `pl.col("nPF_glob") == 13`
    """

    @magicmenu
    class File(MagicTemplate):
        add_children = abstractapi()
        add_children_glob = abstractapi()
        clear_children = abstractapi()
        sep0 = field(Separator)
        load_batch_project = abstractapi()
        save_batch_project = abstractapi()

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

    @magicmenu(name="Macro")
    class MacroMenu(MagicTemplate):
        show_macro = abstractapi()
        show_native_macro = abstractapi()

    seq_name = vfield("Loader").with_options(label="Name:")
    projects = field(ProjectPaths)
    scale = field(1.0, label="Scale (nm):").with_options(
        min=0.001, step=0.0001, max=10.0
    )
    filter_expression = field(ExprStr.In[{"pl": pl}], label="Filter:")

    @Select.wraps
    @do_not_record
    def select_all_projects(self):
        """Select all projects."""
        for wdt in self.projects:
            wdt.check = True
        return None

    @Select.wraps
    @do_not_record
    def select_projects_by_pattern(self, pattern: str):
        """Select projects by pattern matching."""
        for prj in self.projects:
            prj.check = fnmatch(prj.path, pattern)
        return None

    @Select.wraps
    @do_not_record
    def select_molecules_by_pattern(self, pattern: str):
        """Select molecules by pattern matching."""
        for prj in self.projects:
            for mole in prj.molecules:
                mole.check = fnmatch(mole.line.value, pattern)
        return None

    def _get_project_paths(self, _=None) -> list[Path]:
        return [wdt.path for wdt in self.projects]

    def _get_selected_project_paths(self, _=None) -> list[Path]:
        return [prj.path for prj in self._iter_selected_projects()]

    def _iter_selected_projects(self) -> Iterator[Project]:
        for prj in self.projects:
            if prj.check:
                yield prj

    def _get_batch_loader(
        self, order: int = 3, output_shape=None, predicate=None
    ) -> BatchLoader:
        batch_loader = BatchLoader()
        for i, prj in enumerate(iter(self.projects)):
            if not prj.check:
                continue
            loader = prj.get_loader(order=order)
            batch_loader.add_tomogram(loader.image, loader.molecules, image_id=i)

        if predicate is not None:
            batch_loader = batch_loader.filter(predicate)
        if output_shape is not None:
            batch_loader = batch_loader.replace(output_shape=output_shape)
        return batch_loader

    def _get_loader_paths(self, _=None) -> list[tuple[Path, list[Path], Path]]:
        return [prj._get_loader_paths() for prj in self.projects]

    def _get_localprops(self) -> pl.DataFrame:
        dataframes = list[pl.DataFrame]()
        for idx, prj in enumerate(iter(self.projects)):
            df = prj._get_localprops()
            dataframes.append(
                df.with_columns(
                    pl.repeat(idx, pl.count()).cast(pl.UInt16).alias(Mole.image)
                )
            )
        return pl.concat(dataframes, how="diagonal")

    @View.wraps
    @do_not_record
    @set_design(text="View components in 3D")
    def view_components(self):
        """View all the splines and molecules that exist in this project."""
        from cylindra.project import ComponentsViewer

        cbox = ComboBox(choices=self._get_project_paths)
        comp_viewer = ComponentsViewer()

        self.changed.connect(lambda: cbox.reset_choices())

        @cbox.changed.connect
        def _view_project(path: str):
            prj = CylindraProject.from_file(path)
            with prj.open_project() as dir:
                comp_viewer._from_project(prj, dir)

        cont = Container(widgets=[cbox, comp_viewer], labels=False)
        _set_parent(cont, self)
        CylindraMainWidget._active_widgets.add(cont)
        cont.show()
        cbox.changed.emit(cbox.value)
        return None

    @View.wraps
    @set_design(text="View selected components in 3D")
    def view_selected_components(self):
        """View selected components in a 3D viewer."""
        from cylindra.project import ComponentsViewer

        cbox = ComboBox(choices=self._get_selected_project_paths)
        comp_viewer = ComponentsViewer()

        self.changed.connect(lambda: cbox.reset_choices())
        cbox.changed.connect(
            lambda path: comp_viewer._from_project(CylindraProject.from_file(path))
        )
        cont = Container(widgets=[cbox, comp_viewer], labels=False)
        _set_parent(cont, self)
        CylindraMainWidget._active_widgets.add(cont)
        cont.show()
        cbox.changed.emit(cbox.value)
        return None

    @View.wraps
    @do_not_record
    @set_design(text="View selected molecules in table")
    def view_molecules(self):
        """View selected molecules in a table"""
        mole = self._get_batch_loader().molecules
        df = mole.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        CylindraMainWidget._active_widgets.add(table)
        _set_parent(table, self)
        table.show()
        return None

    @View.wraps
    @do_not_record
    @set_design(text="View filtered molecules in table")
    def view_filtered_molecules(self):
        """Preview filtered molecules in a table."""
        mole = self._get_batch_loader(predicate=self._get_expression()).molecules
        df = mole.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        CylindraMainWidget._active_widgets.add(table)
        _set_parent(table, self)
        table.show()
        return None

    @View.wraps
    @do_not_record
    @set_design(text="View local properties")
    def view_localprops(self):
        """View local properties of splines."""
        wdt = LocalPropsViewer()
        _set_parent(wdt, self)
        CylindraMainWidget._active_widgets.add(wdt)
        wdt.show()

        dataframes = list[pl.DataFrame]()
        path_map = dict[int, str]()
        for idx, prj in enumerate(iter(self.projects)):
            df = prj._get_localprops()

            dataframes.append(
                df.with_columns(
                    pl.repeat(idx, pl.count()).cast(pl.UInt16).alias(Mole.image)
                )
            )
            path_map[idx] = prj.path

        df_all = pl.concat(dataframes, how="diagonal")
        wdt._set_localprops(df_all, path_map)
        return None

    def _get_expression(self, _=None) -> pl.Expr:
        wdt = self.filter_expression
        if wdt.value == "":
            return None
        return wdt.eval()

    @File.wraps
    @do_not_record
    @set_design(text="Add projects")
    def add_children(self, paths: Path.Multiple[FileFilter.JSON]):
        """Add project json files as the child projects."""
        for path in paths:
            wdt = self.projects._add(get_project_file(path))
            self.scale.value = wdt.project.scale
        self.reset_choices()
        return None

    @File.wraps
    @do_not_record
    @set_design(text="Add projects with wildcard path")
    def add_children_glob(self, pattern: str):
        """Add project json files using wildcard path."""
        pattern = str(pattern)
        for path in glob.glob(pattern):
            wdt = self.projects._add(path)
            self.scale.value = wdt.project.scale
        self.reset_choices()
        return

    @File.wraps
    @do_not_record
    def clear_children(self):
        """Clear all the projects in the list."""
        self.projects.clear()
        return None

    construct_loader = abstractapi()


def _set_parent(wdt: Widget, parent: Widget):
    wdt.native.setParent(parent.native, wdt.native.windowFlags())


@impl_preview(ProjectSequenceEdit.add_children_glob)
def _(self: ProjectSequenceEdit, pattern: str):
    paths = list[str]()
    for path in glob.glob(str(pattern)):
        paths.append(Path(path).as_posix())
    wdt = ConsoleTextEdit(value="\n".join(paths))
    _set_parent(wdt, self)
    CylindraMainWidget._active_widgets.add(wdt)
    wdt.show()
