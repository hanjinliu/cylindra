from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import TYPE_CHECKING

import polars as pl
from magicgui.widgets import FunctionGui
from magicclass.widgets import ConsoleTextEdit
from magicclass import setup_function_gui, impl_preview

import numpy as np

from cylindra.const import PropertyNames as H, PREVIEW_LAYER_NAME
from cylindra.project import CylindraProject, get_project_json
from cylindra.widgets import widget_utils
from cylindra.widgets.main import CylindraMainWidget
from cylindra.widgets._previews import view_tables
from cylindra.widgets._main_utils import normalize_offsets
from napari.utils.colormaps import label_colormap

if TYPE_CHECKING:
    from cylindra._custom_layers import MoleculesLayer
    from napari.layers import Layer


# Implement preview functions.


@impl_preview(CylindraMainWidget.load_molecules)
def _(self: CylindraMainWidget, paths: list[str]):
    w = view_tables(paths, parent=self)
    self._active_widgets.add(w)


@impl_preview(CylindraMainWidget.load_project)
def _(self: CylindraMainWidget, path: str):
    pviewer = CylindraProject.from_json(get_project_json(path)).make_project_viewer()
    pviewer.native.setParent(self.native, pviewer.native.windowFlags())
    self._active_widgets.add(pviewer)
    return pviewer.show()


@impl_preview(CylindraMainWidget.clip_spline, auto_call=True)
def _(self: CylindraMainWidget, spline: int, lengths: tuple[float, float]):
    tomo = self.tomogram
    name = "Spline preview"
    spl = tomo.splines[spline]
    length = spl.length()
    start, stop = np.array(lengths) / length
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
    is_active = False
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
    self._active_widgets.add(w)
    return None


@impl_preview(CylindraMainWidget.map_along_pf, auto_call=True)
def _(
    self: CylindraMainWidget,
    spline: int,
    molecule_interval: float | None,
    offsets: tuple[float, float] | None = None,
    orientation: str | None = None,
):  # fmt: skip
    tomo = self.tomogram
    viewer = self.parent_viewer
    out = tomo.map_pf_line(
        i=spline,
        interval=molecule_interval,
        offsets=normalize_offsets(offsets, tomo.splines[spline]),
        orientation=orientation,
    )
    if PREVIEW_LAYER_NAME in viewer.layers:
        layer: MoleculesLayer = viewer.layers[PREVIEW_LAYER_NAME]
        layer.molecules = out
    else:
        layer = self.add_molecules(out, name=PREVIEW_LAYER_NAME)
    layer.face_color = layer.edge_color = "crimson"
    is_active = False
    try:
        is_active = yield
    finally:
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)
    return out


@impl_preview(CylindraMainWidget.map_monomers_with_extensions, auto_call=True)
def _(
    self: CylindraMainWidget,
    spline: int,
    n_extend: dict[int, tuple[int, int]],
    orientation=None,
    offsets: tuple[float, float] | None = None,
):
    tomo = self.tomogram
    spl = tomo.splines[spline]
    coords = widget_utils.coordinates_with_extensions(spl, n_extend)
    out = tomo.map_on_grid(
        i=spline,
        coords=coords,
        orientation=orientation,
        offsets=normalize_offsets(offsets, tomo.splines[spline]),
    )
    viewer = self.parent_viewer

    if PREVIEW_LAYER_NAME in viewer.layers:
        layer: MoleculesLayer = viewer.layers[PREVIEW_LAYER_NAME]
        layer.molecules = out
    else:
        layer = self.add_molecules(out, name=PREVIEW_LAYER_NAME)
    layer.face_color = layer.edge_color = "crimson"
    is_active = False
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
    if PREVIEW_LAYER_NAME in viewer.layers:
        layer: Layer = viewer.layers[PREVIEW_LAYER_NAME]
        layer.data = out.pos
    else:
        layer = self.add_molecules(out, name=PREVIEW_LAYER_NAME)
        layer.face_color = layer.edge_color = "crimson"
    is_active = False
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
    if PREVIEW_LAYER_NAME in viewer.layers:
        layer: MoleculesLayer = viewer.layers[PREVIEW_LAYER_NAME]
        layer.molecules = out
    else:
        layer = self.add_molecules(out, name=PREVIEW_LAYER_NAME)
    # filtering changes the number of molecules. We need to update the colors.
    layer.face_color = layer.edge_color = "crimson"
    is_active = False
    try:
        is_active = yield
    finally:
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)
    return out


@impl_preview(CylindraMainWidget.paint_molecules, auto_call=True)
def _(
    self: CylindraMainWidget,
    layer: MoleculesLayer,
    cmap,
    color_by: str,
    limits: tuple[float, float],
):
    with _temp_layer_colors(layer):
        self.paint_molecules(layer, cmap, color_by, limits)
        yield


# setup FunctionGUIs


@setup_function_gui(CylindraMainWidget.paint_molecules)
def _(self: CylindraMainWidget, gui: FunctionGui):
    gui.layer.changed.connect(gui.color_by.reset_choices)

    @gui.layer.changed.connect
    @gui.color_by.changed.connect
    def _on_feature_change():
        color_by: str = gui.color_by.value
        if color_by is None:
            return
        layer: MoleculesLayer = gui.layer.value
        series = layer.molecules.features[color_by]
        if series.dtype in pl.NUMERIC_DTYPES:
            min_, max_ = series.min(), series.max()
            offset_ = (max_ - min_) / 2
            gui.limits[0].min = gui.limits[1].min = min_ - offset_
            gui.limits[0].max = gui.limits[1].max = max_ + offset_
            gui.limits[0].value = min_
            gui.limits[1].value = max_
            if series.dtype in pl.INTEGER_DTYPES:
                gui.limits[0].step = gui.limits[1].step = 1
            else:
                gui.limits[0].step = gui.limits[1].step = None

    @gui.limits[0].changed.connect
    def _assert_limits_0(val: float):
        if val > gui.limits[1].value:
            gui.limits[1].value = val

    @gui.limits[1].changed.connect
    def _assert_limits_1(val: float):
        if val < gui.limits[0].value:
            gui.limits[0].value = val

    return None


@setup_function_gui(CylindraMainWidget.split_molecules)
@setup_function_gui(CylindraMainWidget.seam_search_by_feature)
def _(self: CylindraMainWidget, gui: FunctionGui):
    gui[0].changed.connect(gui[1].reset_choices)


@setup_function_gui(CylindraMainWidget.map_monomers_with_extensions)
def _(self: CylindraMainWidget, gui: FunctionGui):
    @gui.spline.changed.connect
    def _on_spline_change(spline: int | None):
        if spline is None:
            return None
        npf = self.get_spline(spline).get_globalprops(H.nPF, None)
        if npf is None:
            value = {}
        else:
            value = {i: (0, 0) for i in range(npf)}
        if gui.n_extend.value.keys() != value.keys():
            gui.n_extend.value = value

    gui.spline.changed.emit(gui.spline.value)  # initialize


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
