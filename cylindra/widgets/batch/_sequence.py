import glob
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Iterator

import impy as ip
import polars as pl
from acryo import BatchLoader, Molecules, SubtomogramLoader
from magicclass import (
    MagicTemplate,
    abstractapi,
    do_not_record,
    field,
    impl_preview,
    magicclass,
    magicmenu,
    nogui,
    set_design,
    vfield,
)
from magicclass.ext.polars import DataFrameView
from magicclass.types import ExprStr, Path
from magicclass.widgets import ConsoleTextEdit, EvalLineEdit
from magicgui.types import Separator
from magicgui.widgets import ComboBox, Container, Widget

from cylindra._config import get_config
from cylindra.const import FileFilter
from cylindra.const import MoleculesHeader as Mole
from cylindra.core import ACTIVE_WIDGETS
from cylindra.project import CylindraProject, get_project_file
from cylindra.widget_utils import POLARS_NAMESPACE, capitalize
from cylindra.widgets.batch._utils import PathInfo, TempFeatures


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

    def _add_path(self, path: str):
        wdt = MoleculeWidget()
        wdt.line.value = path
        wdt["check"].text = ""
        self.append(wdt)

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[MoleculeWidget]: ...
        def __getitem__(self, idx: int) -> MoleculeWidget: ...


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

    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[SplineWidget]: ...
        def __getitem__(self, idx: int) -> SplineWidget: ...


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
        send_to_viewer = abstractapi()

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

    @set_design(text="✕", max_width=30, location=Header)
    def remove_project(self):
        """Remove this project from the list."""
        parent = self.find_ancestor(ProjectPaths)
        idx = parent.index(self)
        del parent[idx]

    @set_design(text="Open", location=Header)
    def send_to_viewer(self):
        """Send this project to the viewer."""
        from cylindra.core import instance

        if ui := instance():
            ui.load_project(self.path, filter=None)
        else:
            raise ValueError("No Cylindra widget found!")

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

    splines = field(SplineList, location=Components)
    molecules = field(MoleculeList, location=Components)

    @classmethod
    def _from_path(cls, path: Path):
        """Create a Project widget from a project path."""
        path = str(path)
        project = CylindraProject.from_file(path)
        self = cls(project)
        self._update_from_project(clear=False)
        return self

    def _update_from_project(self, clear: bool = True):
        assert self._project is not None
        if clear:
            self.splines.clear()
            self.molecules.clear()

        project = self._project
        self.Header.path.value = project.project_path.as_posix()
        self.Header.path.tooltip = project.project_path.as_posix()

        # load splines
        for _, spline_path in project.iter_spline_paths():
            self.splines._add_path(spline_path.name)

        # load molecules
        for info in project.molecules_info:
            self.molecules._add_path(info.name)

        # collapse empty lists
        if len(self.splines) == 0:
            self.splines.collapsed = True
        if len(self.molecules) == 0:
            self.molecules.collapsed = True
        if len(self.splines) == 0 and len(self.molecules) == 0:
            self.Components.collapsed = True

    @nogui
    @do_not_record
    def get_loader(self, order: int = 3) -> SubtomogramLoader:
        """Get the subtomogram loader of this project"""
        project = CylindraProject.from_file(self.path)
        with project.open_project() as dir:
            molecules = [
                Molecules.from_file(dir / mole.line.value)
                for mole in self.molecules
                if mole.check
            ]
        img = ip.lazy.imread(project.image, chunks=get_config().dask_chunk).value
        if project.invert:
            img = -img
        return SubtomogramLoader(
            img,
            molecules=Molecules.concat(molecules),
            order=order,
            scale=project.scale,
        )

    def _get_loader_paths(self) -> tuple[Path, list[str], Path]:
        """Return (image, molecules, project) paths."""
        prj_path = Path(self.path)
        project = CylindraProject.from_file(self.path)
        if project.image is None or not Path(project.image).exists():
            if rpath := project._try_resolve_image_relative():
                img_path = rpath
            else:
                raise ValueError("No image path found in the project.")
        else:
            img_path = Path(project.image)
        mole_paths = [mole.line.value for mole in self.molecules if mole.check]
        return img_path, mole_paths, prj_path

    def _get_localprops(self) -> pl.DataFrame:
        project = CylindraProject.from_file(self.path)
        with project.open_project() as dir:
            localprops_path = project._localprops_path(dir)
            if not localprops_path.exists():
                raise ValueError("No localprops file found.")

            df = pl.read_csv(localprops_path)
        return df


@magicclass(
    widget_type="scrollable",
    labels=False,
    record=False,
    properties={"min_height": 200, "min_width": 250},
)
class ProjectPaths(MagicTemplate):
    def _add(self, path: Path) -> Project:
        prj = Project._from_path(path)
        self.append(prj)
        return prj

    def __getitem__(self, idx: int) -> Project:
        return super().__getitem__(idx)

    def __iter__(self) -> Iterator[Project]:
        return super().__iter__()

    @property
    def paths(self) -> list[Path]:
        return [Path(wdt.path) for wdt in self]

    def _set_checked(self, checked: bool):
        for wdt in self:
            wdt.check = checked
        return None


@magicclass(name="Projects", record=False, use_native_menubar=False)
class ProjectSequenceEdit(MagicTemplate):
    """The left-side widget that contains list of projects.

    Attributes
    ----------
    scale : nm
        The common scale of images.
    filter_expression : str
        A `polars` expression to filter molecules. e.g. `pl.col("score") > 0.5`.
        Spline global properties are also available during filtering, with suffix
        "_glob". e.g. `pl.col("npf_glob") == 13`
    """

    @magicmenu
    class File(MagicTemplate):
        new_projects = abstractapi()
        add_projects = abstractapi()
        clear_projects = abstractapi()
        sep0 = Separator
        load_batch_project = abstractapi()
        save_batch_project = abstractapi()
        sep1 = Separator
        construct_loader_by_list = abstractapi()

    @magicmenu
    class Select(MagicTemplate):
        select_all_projects = abstractapi()
        deselect_all_projects = abstractapi()
        select_projects_by_pattern = abstractapi()
        select_molecules_by_pattern = abstractapi()

    @magicmenu
    class View(MagicTemplate):
        view_components = abstractapi()
        view_selected_components = abstractapi()
        view_molecules = abstractapi()
        view_filtered_molecules = abstractapi()

    @magicmenu(name="Macro")
    class MacroMenu(MagicTemplate):
        show_macro = abstractapi()
        show_native_macro = abstractapi()

    projects = field(ProjectPaths)
    scale = field(1.0, label="Scale (nm):").with_options(
        min=0.001, step=0.0001, max=10.0
    )
    filter_expression = field(ExprStr.In[POLARS_NAMESPACE], label="Filter:")

    @set_design(text="Select all projects", location=Select)
    @do_not_record
    def select_all_projects(self):
        """Select all projects."""
        return self.projects._set_checked(True)

    @set_design(text="Select projects by pattern", location=Select)
    @do_not_record
    def select_projects_by_pattern(self, pattern: str):
        """Select projects by pattern matching."""
        for prj in self.projects:
            prj.check = fnmatch(prj.path, pattern)
        return None

    @set_design(text="Select molecules by pattern", location=Select)
    @do_not_record
    def select_molecules_by_pattern(self, pattern: str):
        """Select molecules by pattern matching."""
        for prj in self.projects:
            for mole in prj.molecules:
                mole.check = fnmatch(mole.line.value, pattern)
        return None

    @set_design(text="Deselect all projects", location=Select)
    @do_not_record
    def deselect_all_projects(self):
        """Deselect all projects."""
        return self.projects._set_checked(False)

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
        batch_loader = BatchLoader(order=order)
        image_paths: dict[int, Path] = {}
        _temp_features = TempFeatures(enabled=predicate is not None)
        for img_id, prj_wdt in enumerate(iter(self.projects)):
            if not prj_wdt.check:
                continue
            path_info = PathInfo(*prj_wdt._get_loader_paths())
            img = path_info.lazy_imread()
            image_paths[img_id] = Path(path_info.image)
            prj = CylindraProject.from_file(path_info.project)
            with prj.open_project() as dir:
                for mole_wdt in prj_wdt.molecules:
                    if not mole_wdt.check:
                        continue
                    mole = _temp_features.read_molecules(dir / mole_wdt.line.value, prj)
                    batch_loader.add_tomogram(img.value, mole, img_id)

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = eval(predicate, POLARS_NAMESPACE, {})
            batch_loader = batch_loader.filter(predicate)
        if output_shape is not None:
            batch_loader = batch_loader.replace(output_shape=output_shape)
        return batch_loader

    def _get_localprops(self) -> pl.DataFrame:
        dataframes = list[pl.DataFrame]()
        for idx, prj in enumerate(iter(self.projects)):
            df = prj._get_localprops()
            dataframes.append(
                df.with_columns(
                    pl.repeat(idx, pl.len()).cast(pl.UInt16).alias(Mole.image)
                )
            )
        return pl.concat(dataframes, how="diagonal")

    @set_design(text="View components in 3D", location=View)
    @do_not_record
    def view_components(self):
        """View all the splines and molecules that exist in this project."""
        from cylindra.project._widgets import ComponentsViewer

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
        ACTIVE_WIDGETS.add(cont)
        cont.show()
        cbox.changed.emit(cbox.value)
        return None

    @set_design(text="View selected components in 3D", location=View)
    def view_selected_components(self):
        """View selected components in a 3D viewer."""
        from cylindra.project._widgets import ComponentsViewer

        cbox = ComboBox(choices=self._get_selected_project_paths)
        comp_viewer = ComponentsViewer()

        self.changed.connect(lambda: cbox.reset_choices())
        cbox.changed.connect(
            lambda path: comp_viewer._from_project(
                CylindraProject.from_file(path), _get_project_dir(path)
            )
        )
        cont = Container(widgets=[cbox, comp_viewer], labels=False)
        _set_parent(cont, self)
        ACTIVE_WIDGETS.add(cont)
        cont.show()
        cbox.changed.emit(cbox.value)
        return cont

    @set_design(text="View selected molecules in table", location=View)
    @do_not_record
    def view_molecules(self):
        """View selected molecules in a table"""
        mole = self._get_batch_loader().molecules
        df = mole.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        ACTIVE_WIDGETS.add(table)
        _set_parent(table, self)
        table.show()
        return None

    @set_design(text="View filtered molecules in table", location=View)
    @do_not_record
    def view_filtered_molecules(self):
        """Preview filtered molecules in a table."""
        mole = self._get_batch_loader(predicate=self._get_expression()).molecules
        df = mole.to_dataframe()
        if df.shape[0] == 0:
            raise ValueError("All molecules were filtered out.")
        table = DataFrameView(value=df)
        ACTIVE_WIDGETS.add(table)
        _set_parent(table, self)
        table.show()
        return None

    def _get_expression(self, _=None) -> str:
        wdt: EvalLineEdit = self.filter_expression
        if wdt.value == "":
            return None
        return wdt.value

    @set_design(text=capitalize, location=File)
    @do_not_record
    def add_projects(
        self,
        paths: Path.Multiple[FileFilter.PROJECT],
        clear: bool = False,
    ):
        """Add project json files as the child projects.

        Parameters
        ----------
        paths : list of str or Path
            A list of paths or wildcard patterns, such as "path/to/*.json".
        clear : bool, default True
            Whether to clear the existing projects added to the list.
        """

        if isinstance(paths, (str, Path)):
            input_paths = [str(paths)]
        else:
            input_paths = [str(p) for p in paths]
        if clear:
            self.projects.clear()
        for path_or_pattern in input_paths:
            if "*" in path_or_pattern or "?" in path_or_pattern:
                for path in glob.glob(path_or_pattern):
                    wdt = self.projects._add(path)
                    self.scale.value = wdt.project.scale
            else:
                wdt = self.projects._add(get_project_file(path_or_pattern))
                self.scale.value = wdt.project.scale
        self.reset_choices()

    @set_design(text="Clear projects", location=File)
    @do_not_record
    def clear_projects(self):
        """Clear all the projects in the list."""
        self.projects.clear()

    construct_loader = abstractapi()


def _set_parent(wdt: Widget, parent: Widget):
    wdt.native.setParent(parent.native, wdt.native.windowFlags())


def _get_project_dir(path: str):
    _path = Path(path)
    if _path.suffix == ".json":
        return _path.parent
    return _path


@impl_preview(ProjectSequenceEdit.add_projects)
def _(self: ProjectSequenceEdit, paths: list[str]):
    input_paths = list[str]()
    patterns = [str(p) for p in paths]
    for each_pattern in patterns:
        if "*" in each_pattern or "?" in each_pattern:
            for path in glob.glob(each_pattern):
                input_paths.append(Path(path).as_posix())
        else:
            input_paths.append(Path(each_pattern).as_posix())
    wdt = ConsoleTextEdit(value="\n".join(input_paths))
    _set_parent(wdt, self)
    ACTIVE_WIDGETS.add(wdt)
    wdt.show()
