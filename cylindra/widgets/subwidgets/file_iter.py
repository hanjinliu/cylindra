from glob import glob
from pathlib import Path
from typing import Annotated
from magicclass import (
    abstractapi,
    magicclass,
    vfield,
    set_design,
    get_button,
    get_function_gui,
)
from ._child_widget import ChildWidget
from cylindra.widgets._previews import view_image


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
    def set_pattern(self):
        """Set pattern for file search."""
        if self.pattern == "":
            raise ValueError("Pattern is not set.")
        self.pattern = Path(self.pattern).as_posix()
        self._files = list(
            Path(p).as_posix() for p in glob(self.pattern, recursive=True)
        )
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
        if Path(path).name not in ("project.json", ""):
            raise ValueError("Not a project file")
        fgui = get_function_gui(self._get_main().load_project)
        fgui.path.value = path
        fgui.call_button.changed()

    @set_design(text="Re-analyze project")
    def load_project_for_reanalysis(self, path: Annotated[str, {"bind": path}]):
        if Path(path).name not in ("project.json", ""):
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
