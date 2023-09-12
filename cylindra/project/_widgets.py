from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import polars as pl
import impy as ip
from magicclass import magicclass, field, MagicTemplate
from magicclass.widgets import ConsoleTextEdit, FrameContainer, ToggleSwitch, Label
from magicclass.ext.vispy import Vispy3DCanvas
from magicclass.ext.polars import DataFrameView

if TYPE_CHECKING:
    from ._single import CylindraProject
    from magicclass.ext.vispy._base import LayerItem


@magicclass(labels=False, widget_type="split", layout="horizontal", record=False)
class TextInfo(MagicTemplate):
    project_text = field(ConsoleTextEdit)

    @magicclass(labels=False, record=False, widget_type="split")
    class Right(MagicTemplate):
        config = field(ConsoleTextEdit)
        macro_script = field(ConsoleTextEdit)

    def _from_project(self, project: "CylindraProject", dir: Path):
        from cylindra.widgets.widget_utils import get_code_theme

        theme = get_code_theme(self)

        if path := project.project_path:
            with open(path / "project.json") as f:
                self.project_text.value = f.read()
        else:
            # NOTE: paths are alreadly resolved in project.json() so this might be
            # different from the original json file.
            self.project_text.value = str(
                project.json(indent=4, separators=(", ", ": "))
            )

        self.project_text.read_only = True
        self.project_text.syntax_highlight("json", theme=theme)
        cfg_path = dir / "default_spline_config.json"
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
        def _add_layer(self, layer: "LayerItem"):
            visible_btn = ToggleSwitch(value=True, text="")
            label = Label(value=layer.name)
            cont = FrameContainer(
                widgets=[visible_btn, label], layout="horizontal", labels=False
            )
            cont.margins = (0, 0, 0, 0)
            cont.min_width = 200

            @visible_btn.changed.connect
            def _on_visible_change(value):
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
            self.components._add_layer(layer)

        # draw edge
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
        localprops_path = dir / "local_properties.csv"
        globalprops_path = dir / "global_properties.csv"
        if localprops_path.exists():
            df = pl.read_csv(localprops_path)
            self.table_local.value = df

        if globalprops_path.exists():
            df = pl.read_csv(globalprops_path)
            self.table_global.value = df


@magicclass(widget_type="tabbed", name="Project Viewer", record=False)
class ProjectViewer(MagicTemplate):
    info_viewer = field(TextInfo, name="Text files")
    component_viewer = field(ComponentsViewer, name="Components")
    properties = field(Properties, name="Properties")

    def _from_project(self, project: "CylindraProject"):
        with project.open_project() as dir:
            self.info_viewer._from_project(project, dir)
            self.component_viewer._from_project(project, dir)
            self.properties._from_project(project, dir)
