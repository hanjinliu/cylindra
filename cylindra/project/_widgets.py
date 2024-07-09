from typing import TYPE_CHECKING, Annotated

import impy as ip
import numpy as np
import polars as pl
from magicclass import (
    MagicTemplate,
    abstractapi,
    field,
    magicclass,
    magicmenu,
    set_design,
)
from magicclass.ext.polars import DataFrameView
from magicclass.ext.vispy import Vispy3DCanvas
from magicclass.types import Path
from magicclass.utils import thread_worker
from magicclass.widgets import ConsoleTextEdit, FrameContainer, Label, ToggleSwitch

from cylindra.const import ImageFilter
from cylindra.widget_utils import capitalize

if TYPE_CHECKING:
    from magicclass.ext.vispy._base import LayerItem

    from cylindra.project._single import CylindraProject
    from cylindra.widgets import CylindraMainWidget


@magicclass(labels=False, widget_type="split", layout="horizontal", record=False)
class TextInfo(MagicTemplate):
    project_text = field(ConsoleTextEdit)

    @magicclass(labels=False, record=False, widget_type="split")
    class Right(MagicTemplate):
        config = field(ConsoleTextEdit)
        macro_script = field(ConsoleTextEdit)

    def _from_project(self, project: "CylindraProject", dir: Path):
        from cylindra.widget_utils import get_code_theme

        theme = get_code_theme(self)
        path = dir / "project.json"
        if path.exists():
            self.project_text.value = path.read_text()
        else:
            # NOTE: paths are alreadly resolved in project.json() so this might be
            # different from the original json file.
            self.project_text.value = str(
                project.json(indent=4, separators=(", ", ": "))
            )

        self.project_text.read_only = True
        self.project_text.syntax_highlight("json", theme=theme)
        cfg_path = project._default_spline_config_path(dir)
        if cfg_path.exists():
            self.Right.config.value = cfg_path.read_text()
        self.Right.config.read_only = True
        self.Right.config.syntax_highlight("json", theme=theme)

        macro_path = dir / "script.py"
        if macro_path.exists():
            self.Right.macro_script.value = macro_path.read_text()
        self.Right.macro_script.read_only = True
        self.Right.macro_script.syntax_highlight("python", theme=theme)


@magicclass(labels=False, layout="horizontal", record=False)
class ComponentsViewer(MagicTemplate):
    canvas = field(Vispy3DCanvas)

    @magicclass(labels=False, widget_type="scrollable", properties={"min_width": 220})
    class components(MagicTemplate):
        def _add_layer(self, layer: "LayerItem", visible: bool = True):
            visible_btn = ToggleSwitch(value=visible, text="")
            label = Label(value=layer.name)
            cont = FrameContainer(
                widgets=[visible_btn, label], layout="horizontal", labels=False
            )
            cont.margins = (0, 0, 0, 0)
            cont.min_width = 200

            @visible_btn.changed.connect
            def _on_visible_change(value: bool):
                layer.visible = value

            self.append(cont)

    def _from_project(self, project: "CylindraProject", dir: Path):
        self.canvas.layers.clear()
        self.components.clear()

        for i, spl in enumerate(project.iter_load_splines(dir)):
            coords = spl.partition(100)
            layer = self.canvas.add_curve(
                coords, color="crimson", width=5.0, name=f"spline-{i}"
            )
            self.components._add_layer(layer)

        for info, mole in project.iter_load_molecules(dir):
            layer = self.canvas.add_points(mole.pos, face_color="lime", name=info.stem)
            self.components._add_layer(layer, info.visible)

        # draw edge
        if project.image is None or not project.image.exists():
            return
        img = ip.lazy.imread(project.image)
        nz, ny, nx = img.shape
        for z in [0, nz]:
            arr = (
                np.array([[z, 0, 0], [z, 0, nx], [z, ny, nx], [z, ny, 0], [z, 0, 0]])
                * img.scale.x
            )
            self.canvas.add_curve(arr, color="gray")
        for y, x in [(0, 0), (0, nx), (ny, nx), (ny, 0)]:
            arr = np.array([[0, y, x], [nz, y, x]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")


@magicclass(labels=False, widget_type="split", record=False)
class Properties(MagicTemplate):
    table_local = field(widget_type=DataFrameView)
    table_global = field(widget_type=DataFrameView)

    def _from_project(self, project: "CylindraProject", dir: Path):
        localprops_path = project._localprops_path(dir)
        globalprops_path = project._globalprops_path(dir)
        if localprops_path.exists():
            df = pl.read_csv(localprops_path)
            self.table_local.value = df
        else:
            self.table_local.value = pl.DataFrame([["Not found"]])

        if globalprops_path.exists():
            df = pl.read_csv(globalprops_path)
            self.table_global.value = df
        else:
            self.table_global.value = pl.DataFrame([["Not found"]])


@magicclass(widget_type="tabbed", name="Project Viewer", record=False)
class ProjectViewer(MagicTemplate):
    @magicmenu
    class Menu(MagicTemplate):
        load_this_project = abstractapi()
        preview_image = abstractapi()
        close_window = abstractapi()

    def __init__(self):
        self._project: CylindraProject | None = None

    info_viewer = field(TextInfo, name="Text files")
    component_viewer = field(ComponentsViewer, name="Components")
    properties = field(Properties, name="Properties")

    def _from_project(self, project: "CylindraProject"):
        self._project = project
        with project.open_project() as dir:
            self.info_viewer._from_project(project, dir)
            self.component_viewer._from_project(project, dir)
            self.properties._from_project(project, dir)

    def _get_project_path(self, *_):
        if self._project is None:
            raise ValueError("Project path is not known.")
        return self._project.project_path

    @thread_worker
    @set_design(text=capitalize, location=Menu)
    def load_this_project(
        self,
        path: Annotated[str, {"bind": _get_project_path}],
        filter: ImageFilter | None = ImageFilter.Lowpass,
        read_image: Annotated[bool, {"label": "read image data"}] = True,
        update_config: bool = False,
    ):
        """Load current project in main window."""
        from cylindra import instance

        ui: "CylindraMainWidget | None" = None

        @thread_worker.callback
        def _launch_ui():
            nonlocal ui
            ui = instance(create=True)
            self.native.setParent(ui.native, self.native.windowFlags())

        yield _launch_ui
        _launch_ui.await_call()

        if ui is None:
            raise RuntimeError("Main window is not running.")
        yield from ui.load_project.arun(path, filter, read_image, update_config)
        return thread_worker.callback(self.close)

    @set_design(text=capitalize, location=Menu)
    def preview_image(self):
        """Preview the tomogram image."""
        from cylindra._previews import view_image

        return view_image(self._project.image, parent=self)

    @set_design(text="Close", location=Menu)
    def close_window(self):
        """Close this preview."""
        return self.close()
