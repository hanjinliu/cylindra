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
        global_variables = field(ConsoleTextEdit)
        macro_script = field(ConsoleTextEdit)

    def _from_project(self, project: "CylindraProject"):
        from cylindra.widgets.widget_utils import get_code_theme

        theme = get_code_theme(self)

        if path := project.project_path:
            with open(path) as f:
                self.project_text.value = f.read()
        else:
            # NOTE: paths are alreadly resolved in project.json() so this might be
            # different from the original json file.
            self.project_text.value = str(
                project.json(indent=4, separators=(", ", ": "))
            )

        self.project_text.read_only = True
        self.project_text.syntax_highlight("json", theme=theme)

        if path := project.tomogram_config:
            with open(path) as f:
                self.Right.global_variables.value = f.read()
        self.Right.global_variables.read_only = True
        self.Right.global_variables.syntax_highlight("json", theme=theme)

        if path := project.macro:
            with open(path) as f:
                self.Right.macro_script.value = f.read()
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

    def _from_project(self, project: "CylindraProject"):
        from cylindra.components import CylSpline
        from acryo import Molecules

        self.canvas.layers.clear()
        self.components.clear()

        for path in project.splines:
            spl = CylSpline.from_json(path)
            coords = spl.partition(100)
            layer = self.canvas.add_curve(
                coords, color="crimson", width=5.0, name=path.stem
            )
            self.components._add_layer(layer)

        for path in project.molecules:
            mole = Molecules.from_csv(path)
            layer = self.canvas.add_points(mole.pos, face_color="lime", name=path.stem)
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

    def _from_project(self, project: "CylindraProject"):
        if path := project.localprops:
            df = pl.read_csv(path)
            self.table_local.value = df

        if path := project.globalprops:
            df = pl.read_csv(path)
            self.table_global.value = df


@magicclass(widget_type="tabbed", name="Project Viewer", record=False)
class ProjectViewer(MagicTemplate):
    info_viewer = field(TextInfo, name="Text files")
    component_viewer = field(ComponentsViewer, name="Components")
    properties = field(Properties, name="Properties")

    def _from_project(self, project: "CylindraProject"):
        self.info_viewer._from_project(project)
        self.component_viewer._from_project(project)
        self.properties._from_project(project)
