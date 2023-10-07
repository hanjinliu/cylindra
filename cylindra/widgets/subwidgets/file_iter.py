from glob import glob
from pathlib import Path
from typing import Annotated, TYPE_CHECKING
from magicclass import (
    abstractapi,
    magicclass,
    field,
    vfield,
    set_design,
    get_button,
    get_function_gui,
)
from ._child_widget import ChildWidget
from cylindra.widgets._previews import view_image
from cylindra.project import ProjectSequence, CylindraProject
from cylindra.widgets.subwidgets import LocalPropertiesWidget

if TYPE_CHECKING:
    from cylindra.components import CylSpline


@magicclass(name="_File Iterator", record=False)
class FileIterator(ChildWidget):
    @magicclass(layout="horizontal")
    class Header(ChildWidget):
        pattern = abstractapi()
        set_pattern = abstractapi()

    pattern = Header.vfield(str)
    path = vfield(str).with_options(enabled=False)

    @magicclass(layout="horizontal")
    class Arrows(ChildWidget):
        first_file = abstractapi()
        prev_file = abstractapi()
        next_file = abstractapi()
        last_file = abstractapi()

    def __init__(self):
        self._current_index = 0
        self._files: list[str] = []

    @Header.wraps
    @set_design(text="Set")
    def set_pattern(self, val: Annotated[str, {"bind": pattern}]):
        """Set pattern for file search."""
        if val == "":
            raise ValueError("Pattern is not set.")
        self.pattern = Path(val).as_posix()
        self._files = list(
            Path(p).as_posix() for p in glob(self.pattern, recursive=True)
        )
        self._files.sort()
        if len(self._files) == 0:
            self.path = "No files found"
        self._update_index(0)

    @Arrows.wraps
    @set_design(text="|<<")
    def first_file(self):
        """Select the first file."""
        self._update_index(0)

    @Arrows.wraps
    @set_design(text=">>|")
    def last_file(self):
        """Select the last file."""
        self._update_index(len(self._files) - 1)

    @Arrows.wraps
    @set_design(text="<")
    def prev_file(self):
        """Select the previous file."""
        self._update_index(self._current_index - 1)

    @Arrows.wraps
    @set_design(text=">")
    def next_file(self):
        """Select next file."""
        self._update_index(self._current_index + 1)

    def _update_index(self, inext: int):
        if inext < 0 or inext >= len(self._files):
            left_most = right_most = False
        else:
            self._current_index = inext
            self.path = self._files[self._current_index]
            left_most = inext > 0
            right_most = inext < len(self._files) - 1
        get_button(self.first_file).enabled = left_most
        get_button(self.prev_file).enabled = left_most
        get_button(self.last_file).enabled = right_most
        get_button(self.next_file).enabled = right_most

    @set_design(text="Open image ...")
    def open_image(self, path: Annotated[str, {"bind": path}]):
        loader = self._get_main().File.open_image_loader()
        loader.path = path
        loader.show()

    @set_design(text="Load project")
    def load_project(self, path: Annotated[str, {"bind": path}]):
        if Path(path).suffix not in (".json", "", ".tar", ".zip"):
            raise ValueError("Not a project file")
        fgui = get_function_gui(self._get_main().load_project)
        fgui.path.value = path
        fgui.call_button.changed()

    @set_design(text="Re-analyze project")
    def load_project_for_reanalysis(self, path: Annotated[str, {"bind": path}]):
        if Path(path).suffix not in (".json", "", ".tar", ".zip"):
            raise ValueError("Not a project file")
        fgui = get_function_gui(self._get_main().load_project_for_reanalysis)
        fgui.path.value = path
        fgui.call_button.changed()

    @set_design(text="Preview all")
    def preview_all(self):
        """Preview all the images."""
        file_paths = self._files
        for path in file_paths:
            path = Path(path)
            if path.is_dir():
                raise ValueError(f"{path} is a directory.")
            if not path.exists():
                raise ValueError(f"{path} does not exist.")
            if path.suffix not in (".tif", ".tiff", ".mrc", ".rec", ".map", ".st"):
                raise ValueError(f"Cannot open {path} as an image.")

        return view_image(file_paths, parent=self._get_main())

    @set_design(text="View localprops")
    def view_local_props(self):
        seq = ProjectSequence.from_paths(self._files, skip_exc=False)
        wdt = LocalPropsViewer(seq)
        self.parent_viewer.window.add_dock_widget(wdt)


@magicclass
class LocalPropsViewer(ChildWidget):
    def __init__(self, seq: ProjectSequence):
        self._seq = seq

    def _get_projects(self, *_) -> list[tuple[str, CylindraProject]]:
        return [(_simple_path(p.project_path), p) for p in self._seq]

    def _get_splines(self, *_) -> "list[tuple[str, CylSpline]]":
        prj = self.project
        if prj is None:
            return []
        with prj.open_project() as dir:
            out = list(self.project.iter_load_splines(dir))
        return out

    project = vfield(CylindraProject).with_choices(_get_projects)
    spline = vfield().with_choices(_get_splines)
    pos = vfield(int, widget_type="Slider")
    localprops = field(LocalPropertiesWidget)

    @project.connect
    def _project_changed(self, prj: CylindraProject):
        self["spline"].reset_choices()

    @spline.connect
    def _spline_changed(self, spline: "CylSpline"):
        if not spline.has_anchors:
            return
        self["pos"].max = spline.anchors.size - 1
        self.localprops._plot_properties(spline)

    @pos.connect
    def _pos_changed(self, pos: int):
        spl = self.spline
        if spl is None or not spl.has_anchors:
            return
        x = spl.anchors * spl.length()
        self.localprops._plot_spline_position(x[pos])
        self.localprops._set_text(spl, pos)

    def _reset(self):
        self._spline_changed(self.spline)
        self._pos_changed(self.pos)

    def __post_init__(self):
        self.localprops._props_changed.connect(self._reset)
        self._reset()


def _simple_path(path: Path) -> str:
    parts = path.parts
    if len(parts) < 3:
        return path.as_posix()
    return "/".join(["...", parts[-2], parts[-1]])
