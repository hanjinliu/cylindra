from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import pandas as pd
import impy as ip
from magicclass import magicclass, field, vfield, MagicTemplate
from magicclass.widgets import ConsoleTextEdit, FrameContainer, ToggleSwitch, Label, Table
from magicclass.ext.vispy import Vispy3DCanvas
from acryo import Molecules

from cylindra.const import PropertyNames as H

if TYPE_CHECKING:
    from .single import CylindraProject
    from magicclass.ext.vispy._base import LayerItem

@magicclass(labels=False, name="General info", layout="horizontal", record=False)
class Info(MagicTemplate):
    text = field(ConsoleTextEdit)
    global_variables = field(ConsoleTextEdit)
    
    def _from_project(self, project: "CylindraProject"):
        info = {
            "Date": project.datetime,
            "Version": project.version,
            "Dependencies": "<br>".join(
                map(lambda x: "{}={}".format(*x), project.dependency_versions.items())
            ),
            "Image": str(project.image),
            "Image scale": f"{project.scale} nm/pixel",
            "Multiscales": ", ".join(map(str, project.multiscales,)),
            "FT size": f"{project.current_ft_size:.1f} nm",
        }
        info_str = "<br>".join(map(lambda x: "<h2>{}</h2>{}".format(*x), info.items()))
        self.text.value = info_str
        self.text.read_only = True
        
        if path := project.global_variables:
            with open(path, mode="r") as f:
                self.global_variables.value = f.read()
        self.global_variables.read_only = True
        self.global_variables.syntax_highlight("json")

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
        
        self.canvas.layers.clear()
        self.components.clear()
        
        for path in project.splines:
            spl = CylSpline.from_json(path)
            coords = spl.partition(100)
            layer = self.canvas.add_curve(coords, color="crimson", width=5.0, name=path.stem)
            self.components._add_layer(layer)
            
        for path in project.molecules:
            mole = Molecules.from_csv(path)
            layer = self.canvas.add_points(mole.pos, face_color="lime", name=path.stem)
            self.components._add_layer(layer)
        
        # draw edge
        img = ip.lazy_imread(project.image)
        nz, ny, nx = img.shape
        for z in [0, nz]:
            arr = np.array([[z, 0, 0], [z, 0, nx], [z, ny, nx], [z, ny, 0], [z, 0, 0]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")
        for y, x in [(0, 0), (0, nx), (ny, nx), (ny, 0)]:
            arr = np.array([[0, y, x], [nz, y, x]]) * img.scale.x
            self.canvas.add_curve(arr, color="gray")

@magicclass(labels=False, widget_type="split", record=False)
class Properties(MagicTemplate):
    table_local = field([], widget_type=Table)
    table_global = field([], widget_type=Table)
    
    def _from_project(self, project: "CylindraProject"):
        if path := project.localprops:
            df = pd.read_csv(path)
            self.table_local.value = df
        self.table_local.read_only = True
        
        if path := project.globalprops:
            df = pd.read_csv(path)
            self.table_global.value = df
        self.table_global.read_only = True


@magicclass(name="Subtomogram averaging", record=False)
class SubtomogramAveraging(MagicTemplate):
    template_image = field(Vispy3DCanvas)
    mask_parameters = vfield(str)
    tilt_range = vfield(str)
    
    def _from_project(self, project: "CylindraProject"):
        from skimage.filters.thresholding import threshold_yen
        
        if project.template_image is None or Path(project.template_image).is_dir():
            # no template image available
            pass
        else:
            img = ip.imread(project.template_image)
            thr = threshold_yen(img.value)
            self.template_image.add_image(img, rendering="iso", iso_threshold=thr)
        self.mask_parameters = str(project.mask_parameters)
        if project.tilt_range is not None:
            s0, s1 = project.tilt_range
            self.tilt_range = f"({s0:.1f}, {s1:.1f})"

@magicclass(labels=False, record=False)
class Macro(MagicTemplate):
    text = field(str, widget_type=ConsoleTextEdit)
    
    def _from_project(self, project: "CylindraProject"):
        if path := project.macro:
            with open(path, mode="r") as f:
                self.text.value = f.read()
        self.text.read_only = True
        self.text.syntax_highlight("python")

@magicclass(widget_type="tabbed", name="Project Viewer", record=False)
class ProjectViewer(MagicTemplate):
    info_viewer = field(Info)
    component_viewer = field(ComponentsViewer)
    properties = field(Properties)
    subtomogram_averaging = field(SubtomogramAveraging)
    macro_viewer = field(Macro)
    
    def _from_project(self, project: "CylindraProject"):
        self.info_viewer._from_project(project)
        self.component_viewer._from_project(project)
        self.properties._from_project(project)
        self.subtomogram_averaging._from_project(project)
        self.macro_viewer._from_project(project)
