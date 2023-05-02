from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager
from typing import TYPE_CHECKING
from magicclass.widgets import ConsoleTextEdit
from magicclass import setup_function_gui, impl_preview

import numpy as np
import impy as ip
import polars as pl

from cylindra.project import CylindraProject, get_project_json
from cylindra.widgets import widget_utils
from cylindra.widgets.main import CylindraMainWidget
from cylindra.widgets._previews import view_tables
from napari.utils.colormaps import label_colormap

if TYPE_CHECKING:
    from cylindra._molecules_layer import MoleculesLayer
    from napari.layers import Layer


# Implement preview functions.


@impl_preview(CylindraMainWidget.load_molecules)
def _(self: CylindraMainWidget, paths: list[str]):
    return view_tables(paths, parent=self)


@impl_preview(CylindraMainWidget.load_project)
def _(self: CylindraMainWidget, path: str):
    pviewer = CylindraProject.from_json(get_project_json(path)).make_project_viewer()
    pviewer.native.setParent(self.native, pviewer.native.windowFlags())
    return pviewer.show()


@impl_preview(CylindraMainWidget.clip_spline, auto_call=True)
def _(self: CylindraMainWidget, spline: int, clip_lengths: tuple[float, float]):
    tomo = self.tomogram
    name = "Spline preview"
    spl = self.tomogram.splines[spline]
    length = spl.length()
    start, stop = np.array(clip_lengths) / length
    verts = tomo.splines[spline].clip(start, 1 - stop).partition(100)
    verts_2d = verts[:, 1:]
    viewer = self.parent_viewer
    if name in viewer.layers:
        layer: Layer = viewer.layers[name]
        layer.data = verts_2d
    else:
        layer = viewer.add_shapes(
            verts_2d,
            shape_type="path",
            edge_color="crimson",
            edge_width=3,
            name=name,
        )
    try:
        is_active = yield
    finally:
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)


@impl_preview(
    CylindraMainWidget.load_project_for_reanalysis, text="Preview extracted code"
)
def _(self: CylindraMainWidget, path: Path):
    macro = self._get_reanalysis_macro(path)
    w = ConsoleTextEdit(value=str(macro))
    w.syntax_highlight("python")
    w.read_only = True
    w.native.setParent(self.native, w.native.windowFlags())
    w.show()
    return None


@impl_preview(CylindraMainWidget.extend_molecules, auto_call=True)
def _(
    self: CylindraMainWidget,
    layer: MoleculesLayer,
    counts: list[tuple[int, tuple[int, int]]],
):
    out = widget_utils.extend_protofilament(layer.molecules, dict(counts))
    viewer = self.parent_viewer
    name = "<Preview>"
    if name in viewer.layers:
        layer: Layer = viewer.layers[name]
        layer.data = out.pos
    else:
        layer = self.add_molecules(out, name=name)
    layer.face_color = layer.edge_color = "crimson"
    try:
        is_active = yield
    finally:
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)
    return out


@impl_preview(CylindraMainWidget.split_molecules, auto_call=True)
def _(self: CylindraMainWidget, layer: MoleculesLayer, by: str):
    with _temp_layer_colors(layer):
        series = layer.molecules.features[by]
        unique_values = series.unique()
        # NOTE: the first color is translucent
        cmap = label_colormap(unique_values.len() + 1, seed=0.9414)
        layer.face_color_cycle = layer.edge_color_cycle = cmap.colors[1:]
        layer.face_color = layer.edge_color = by
        yield


@impl_preview(CylindraMainWidget.translate_molecules, auto_call=True)
def _(self: CylindraMainWidget, layer: MoleculesLayer, translation, internal: bool):
    mole = layer.molecules
    if internal:
        out = mole.translate_internal(translation)
    else:
        out = mole.translate(translation)
    viewer = self.parent_viewer
    name = "<Preview>"
    if name in viewer.layers:
        layer: Layer = viewer.layers[name]
        layer.data = out.pos
    else:
        layer = self.add_molecules(out, name=name)
        layer.face_color = layer.edge_color = "crimson"
    try:
        is_active = yield
    finally:
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)
    return out


@impl_preview(CylindraMainWidget.filter_molecules, auto_call=True)
def _(self: CylindraMainWidget, layer: MoleculesLayer, predicate: str):
    mole = layer.molecules
    viewer = self.parent_viewer
    try:
        expr = eval(predicate, widget_utils.POLARS_NAMESPACE, {})
    except Exception:
        yield
        return
    out = mole.filter(expr)
    name = "<Preview>"
    if name in viewer.layers:
        layer: Layer = viewer.layers[name]
        layer.data = out.pos
    else:
        layer = self.add_molecules(out, name=name)
    # filtering changes the number of molecules. We need to update the colors.
    layer.face_color = layer.edge_color = "crimson"
    try:
        is_active = yield
    finally:
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)
    return out


@impl_preview(CylindraMainWidget.paint_molecules, auto_call=True)
def _(self: CylindraMainWidget, layer: MoleculesLayer, feature_name: str, low, high):
    with _temp_layer_colors(layer):
        self.paint_molecules(layer, feature_name, low, high)
        yield


# setup FunctionGUIs


@setup_function_gui(CylindraMainWidget.paint_molecules)
def _(self, gui):
    gui.layer.changed.connect(gui.feature_name.reset_choices)

    @gui.layer.changed.connect
    @gui.feature_name.changed.connect
    def _on_feature_change():
        feature_name: str = gui.feature_name.value
        if feature_name is None:
            return
        layer: MoleculesLayer = gui.layer.value
        series = layer.molecules.features[feature_name]
        if series.dtype.__name__[0] in "IUF":
            gui.low[0].min = gui.high[0].min = series.min()
            gui.low[0].max = gui.high[0].max = series.max()
            gui.low[0].value = gui.low[0].min
            gui.high[0].value = gui.high[0].max
            if series.dtype.__name__[0] in "IU":
                gui.low[0].step = gui.high[0].step = 1
            else:
                gui.low[0].step = gui.high[0].step = None

    return None


@setup_function_gui(CylindraMainWidget.split_molecules)
@setup_function_gui(CylindraMainWidget.seam_search_by_feature)
def _(self, gui):
    gui[0].changed.connect(gui[1].reset_choices)


@contextmanager
def _temp_layer_colors(layer: MoleculesLayer):
    """Temporarily change the colors of a layer and restore them afterwards."""
    fc = layer.face_color
    ec = layer.edge_color
    fcmap = layer.face_colormap
    ecmap = layer.edge_colormap
    fclim = layer.face_contrast_limits
    eclim = layer.edge_contrast_limits
    try:
        yield
    finally:
        layer.face_color = fc
        layer.edge_color = ec
        layer.face_colormap = fcmap
        layer.edge_colormap = ecmap
        layer.face_contrast_limits = fclim
        layer.edge_contrast_limits = eclim