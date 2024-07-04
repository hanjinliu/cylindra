"""Implement preview functions for the main widget."""
from __future__ import annotations

from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from acryo import Molecules
from magicclass import impl_preview, setup_function_gui
from magicclass.widgets import ConsoleTextEdit
from magicgui.widgets import FunctionGui, Label, PushButton
from napari.utils.colormaps import label_colormap

from cylindra import _config, utils, widget_utils
from cylindra._napari import MoleculesLayer
from cylindra._previews import view_tables
from cylindra.const import (
    POLARS_INTEGER_DTYPES,
    POLARS_NUMERIC_DTYPES,
    PREVIEW_LAYER_NAME,
)
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.core import ACTIVE_WIDGETS
from cylindra.project import CylindraProject
from cylindra.widgets._main_utils import (
    degrees_to_rotator,
    normalize_offsets,
    normalize_radius,
)
from cylindra.widgets.main import CylindraMainWidget
from cylindra.widgets.sta import SubtomogramAveraging

if TYPE_CHECKING:
    from magicgui.widgets import FloatSpinBox
    from napari.layers import Layer


@impl_preview(CylindraMainWidget.load_molecules)
def _preview_load_molecules(self: CylindraMainWidget, paths: list[str]):
    w = view_tables(paths, parent=self)
    ACTIVE_WIDGETS.add(w)


@impl_preview(CylindraMainWidget.load_project)
def _preview_load_project(self: CylindraMainWidget, path: str):
    pviewer = CylindraProject.from_file(path).make_project_viewer()
    pviewer.native.setParent(self.native, pviewer.native.windowFlags())
    ACTIVE_WIDGETS.add(pviewer)
    return pviewer.show()


@impl_preview(CylindraMainWidget.clip_spline, auto_call=True)
def _preview_clip_spline(
    self: CylindraMainWidget, spline: int, lengths: tuple[float, float]
):
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
    is_active = yield
    if not is_active and layer in viewer.layers:
        viewer.layers.remove(layer)


@impl_preview(
    CylindraMainWidget.load_project_for_reanalysis, text="Preview extracted code"
)
def _preview_load_project_for_reanalysis(self: CylindraMainWidget, path: Path):
    macro = self._get_reanalysis_macro(path)
    w = ConsoleTextEdit(value=str(macro))
    w.syntax_highlight("python", theme=widget_utils.get_code_theme(self))
    w.read_only = True
    w.native.setParent(self.native, w.native.windowFlags())
    w.show()
    ACTIVE_WIDGETS.add(w)


@impl_preview(CylindraMainWidget.map_along_pf, auto_call=True)
def _preview_map_along_pf(
    self: CylindraMainWidget,
    spline: int,
    molecule_interval: str,
    offsets: tuple[float, float] | None = None,
    orientation: str | None = None,
):  # fmt: skip
    tomo = self.tomogram
    viewer = self.parent_viewer
    spl = tomo.splines[spline]
    interv_expr = widget_utils.norm_scalar_expr(molecule_interval)
    interval = spl.props.glob.select(interv_expr).to_series()[0]
    out = tomo.map_pf_line(
        i=spline,
        interval=interval,
        offsets=normalize_offsets(offsets, spl),
        orientation=orientation,
    )
    layer = _update_existing_layer(self, out)
    layer.face_color = "crimson"
    is_active = yield
    if not is_active and layer in viewer.layers:
        viewer.layers.remove(layer)


@impl_preview(CylindraMainWidget.map_monomers_with_extensions, auto_call=True)
def _preview_map_monomers_with_extensions(
    self: CylindraMainWidget,
    spline: int,
    n_extend: dict[int, tuple[int, int]],
    orientation=None,
    offsets: tuple[float, float] | None = None,
    radius: float | None = None,
):
    tomo = self.tomogram
    spl = tomo.splines[spline]
    coords = widget_utils.coordinates_with_extensions(spl, n_extend)
    out = tomo.map_on_grid(
        i=spline,
        coords=coords,
        orientation=orientation,
        offsets=normalize_offsets(offsets, spl),
        radius=normalize_radius(radius, spl),
    )
    viewer = self.parent_viewer

    layer = _update_existing_layer(self, out)
    layer.view_ndim = 2
    layer.text = {"string": "{pf-id}"}
    layer.face_color = "crimson"
    is_active = yield
    if not is_active and layer in viewer.layers:
        viewer.layers.remove(layer)


@impl_preview(CylindraMainWidget.split_molecules, auto_call=True)
def _preview_split_molecules(self: CylindraMainWidget, layer: MoleculesLayer, by: str):
    with _temp_layer_colors(layer):
        series = layer.molecules.features[by]
        unique_values = series.unique()
        # NOTE: the first color is translucent
        cmap = label_colormap(unique_values.len() + 1, seed=0.9414)
        colors = cmap.colors[1:]
        layer.face_color_cycle = colors
        if layer._view_ndim == 3:
            layer.edge_color_cycle = colors
        layer.face_color = by
        yield


@impl_preview(SubtomogramAveraging.seam_search_manually, auto_call=True)
def _preview_seam_search_manual(
    self: SubtomogramAveraging, layer: MoleculesLayer, location: int
):
    from cylindra.components.seam_search import ManualSeamSearcher

    feat = layer.features
    npf = utils.roundint(layer.molecules.features[Mole.pf].max() + 1)
    seam_searcher = ManualSeamSearcher(npf)
    result = seam_searcher.search(location)
    series = result.as_series(feat.shape[0]).to_numpy()
    with _temp_layer_colors(layer):
        layer.face_color = np.where(series, "#FF005E", "#A5A5A5")
        layer.edge_color = "#00105B"
        yield


@impl_preview(CylindraMainWidget.translate_molecules, auto_call=True)
def _preview_translate_molecules(
    self: CylindraMainWidget,
    layers: list[MoleculesLayer],
    translation: list[float],
    internal: bool,
):
    if len(layers) == 0:
        yield
        return
    all_mole = list[Molecules]()
    for layer in layers:
        mole = layer.molecules
        if internal:
            out = mole.translate_internal(translation)
        else:
            out = mole.translate(translation)
        all_mole.append(out)
    out = Molecules.concat(all_mole, concat_features=False)
    viewer = self.parent_viewer
    layer = _update_existing_layer(self, out)
    layer.face_color = "crimson"
    is_active = False
    is_active = yield
    if not is_active and layer in viewer.layers:
        viewer.layers.remove(layer)


@impl_preview(CylindraMainWidget.rotate_molecules, auto_call=True)
def _preview_rotate_molecules(
    self: CylindraMainWidget,
    layers: list[MoleculesLayer],
    degrees: list[tuple[Literal["z", "y", "x"], float]],
):
    if len(layers) == 0:
        yield
        return
    all_data = list[np.ndarray]()
    rotvec = degrees_to_rotator(degrees).as_rotvec()
    for layer in layers:
        mole = layer.molecules.rotate_by_rotvec_internal(rotvec)

        nmol = len(mole)
        zvec = np.stack([mole.pos, mole.z], axis=1)
        yvec = np.stack([mole.pos, mole.y], axis=1)
        xvec = np.stack([mole.pos, mole.x], axis=1)

        vector_data = np.concatenate([zvec, yvec, xvec], axis=0)
        all_data.append(vector_data)

    vector_data = np.concatenate(all_data, axis=0)
    viewer = self.parent_viewer
    if PREVIEW_LAYER_NAME in viewer.layers:
        layer: Layer = viewer.layers[PREVIEW_LAYER_NAME]
        layer.data = vector_data
    else:
        layer = viewer.add_vectors(
            vector_data,
            edge_width=0.3,
            edge_color="direction",
            edge_color_cycle=["crimson", "cyan", "orange"],
            features={"direction": ["z"] * nmol + ["y"] * nmol + ["x"] * nmol},
            length=_config.get_config().point_size * 0.8,
            name=PREVIEW_LAYER_NAME,
            vector_style="arrow",
        )
    is_active = yield
    if not is_active and layer in viewer.layers:
        viewer.layers.remove(layer)


@impl_preview(CylindraMainWidget.filter_molecules, auto_call=True)
def _preview_filter_molecules(
    self: CylindraMainWidget, layer: MoleculesLayer, predicate: str
):
    try:
        expr: pl.Expr = eval(predicate, widget_utils.POLARS_NAMESPACE, {})
    except Exception:
        yield
        return
    out = layer.molecules.to_dataframe().select(expr)
    out = out[out.columns[0]].to_numpy()
    with _temp_layer_colors(layer):
        face_color = layer.face_color
        face_color_trans = face_color.copy()
        face_color_trans[:, 3] *= 0.2
        colors = np.where(out[:, np.newaxis], face_color, face_color_trans)
        layer.face_color = colors
        yield


@impl_preview(CylindraMainWidget.paint_molecules, auto_call=True)
def _preview_paint_molecules(
    self: CylindraMainWidget,
    layer: MoleculesLayer,
    cmap,
    color_by: str,
    limits: tuple[float, float],
):
    with _temp_layer_colors(layer):
        self.paint_molecules(layer, cmap, color_by, limits)
        yield


@impl_preview(CylindraMainWidget.convolve_feature, auto_call=True)
def _convolve_feature_preview(
    self: CylindraMainWidget,
    layer: MoleculesLayer,
    target: str,
    method: Literal["mean", "max", "min", "median"],
    footprint: Any,
):
    from cylindra import cylfilters

    if isinstance(layer.colormap_info, str):
        yield
        return
    nrise = layer.source_spline.nrise()
    df = layer.molecules.features
    out = cylfilters.run_filter(df, footprint, target, nrise, method)
    colors = layer.colormap_info.map(out.to_numpy())
    with _temp_layer_colors(layer):
        layer.face_color = colors
        yield


@impl_preview(CylindraMainWidget.binarize_feature, auto_call=True)
def _binarize_feature_preview(
    self: CylindraMainWidget,
    layer: MoleculesLayer,
    target: str,
    threshold: float,
    larger_true: bool,
):
    if larger_true:
        out = layer.molecules.features[target] >= threshold
    else:
        out = layer.molecules.features[target] < threshold
    with _temp_layer_colors(layer):
        layer.face_color = np.where(out, "#FF0000", "#A5A5A5")
        layer.edge_color = "#00105B"
        yield


@impl_preview(CylindraMainWidget.label_feature_clusters, auto_call=True)
def _label_feature_clusters_preview(
    self: CylindraMainWidget,
    layer: MoleculesLayer,
    target: str,
):
    from cylindra import cylfilters

    nrise = layer.source_spline.nrise()
    out = cylfilters.label(layer.molecules.features, target, nrise)
    max_value = int(out.max())
    with _temp_layer_colors(layer):
        cmap = label_colormap(max_value, seed=0.9414)
        layer.face_color = cmap.map(out.to_numpy() / max_value)
        yield


# setup FunctionGUIs


@setup_function_gui(CylindraMainWidget.paint_molecules)
def _setup_paint_molecules(self: CylindraMainWidget, gui: FunctionGui):
    gui.layer.changed.connect(gui.color_by.reset_choices)
    lim_l: FloatSpinBox = gui.limits[0]
    lim_h: FloatSpinBox = gui.limits[1]

    @gui.layer.changed.connect
    @gui.color_by.changed.connect
    def _on_feature_change():
        color_by: str = gui.color_by.value
        if color_by is None:
            return
        layer: MoleculesLayer = gui.layer.value
        series = layer.molecules.features[color_by]
        if series.dtype in POLARS_NUMERIC_DTYPES:
            series = series.filter(~series.is_infinite())
            min_, max_ = series.min(), series.max()
            offset_ = (max_ - min_) / 2
            lim_l.min = lim_h.min = min_ - offset_
            lim_l.max = lim_h.max = max_ + offset_
            lim_l.value = min_
            lim_h.value = max_
            if series.dtype in POLARS_INTEGER_DTYPES:
                lim_l.step = lim_h.step = 1
            else:
                lim_l.step = lim_h.step = None

    @lim_l.changed.connect
    def _assert_limits_0(val: float):
        if val > lim_h.value:
            with suppress(ValueError):
                lim_h.value = val

    @lim_h.changed.connect
    def _assert_limits_1(val: float):
        if val < lim_l.value:
            with suppress(ValueError):
                lim_l.value = val

    return None


@setup_function_gui(CylindraMainWidget.copy_spline_new_config)
def _setup_copy_spline_new_config(self: CylindraMainWidget, gui: FunctionGui):
    btn = PushButton(
        text="Scan spline config",
        tooltip="Scan current spline config and update the parameters.",
    )

    @btn.clicked.connect
    def _scan_config(_=None):
        idx: int = gui.i.value
        if idx is None:
            return
        spl = self.splines[idx]
        gui.update(spl.config.asdict())

    gui.insert(-1, btn)


@setup_function_gui(CylindraMainWidget.split_molecules)
@setup_function_gui(SubtomogramAveraging.seam_search_by_feature)
@setup_function_gui(CylindraMainWidget.convolve_feature)
@setup_function_gui(CylindraMainWidget.label_feature_clusters)
def _setup_fn_with_column_selection(self: CylindraMainWidget, gui: FunctionGui):
    gui[0].changed.connect(gui[1].reset_choices)


@setup_function_gui(CylindraMainWidget.binarize_feature)
def _setup_binarize_feature(self: CylindraMainWidget, gui: FunctionGui):
    gui.layer.changed.connect(gui.target.reset_choices)

    @gui.target.changed.connect
    def _on_feature_change(target: str):
        layer = gui.layer.value
        assert isinstance(layer, MoleculesLayer)
        ser = layer.molecules.features[target]
        ser = ser.filter(~ser.is_infinite())
        if len(ser) == 0:
            return
        low, high = ser.min(), ser.max()
        gui.threshold.min, gui.threshold.max = low, high
        gui.threshold.step = (high - low) / 200
        gui.threshold.value = ser.median()

    if (val := gui.target.value) is not None:
        _on_feature_change(val)


@setup_function_gui(CylindraMainWidget.map_monomers_with_extensions)
def _setup_map_monomers_with_extensions(self: CylindraMainWidget, gui: FunctionGui):
    @gui.spline.changed.connect
    def _on_spline_change(spline: int | None):
        if spline is None:
            return None
        npf = self.splines[spline].props.get_glob(H.npf, None)
        if npf is None:
            value = {}
        else:
            value = {i: (0, 0) for i in range(npf)}
        if gui.n_extend.value.keys() != value.keys():
            gui.n_extend.value = value

    gui.spline.changed.emit(gui.spline.value)  # initialize


@setup_function_gui(CylindraMainWidget.rename_molecules)
def _setup_rename_molecules(self: CylindraMainWidget, gui: FunctionGui):
    label = Label(value="No change.")

    @gui.changed.connect
    def _on_change(*_):
        if self.parent_viewer is None:
            return
        old: str = gui.old.value
        new: str = gui.new.value
        if old == "" or new == "":
            label.value = "Not enough arguments."
            return
        include: str = gui.include.value
        exclude: str = gui.exclude.value
        pattern: str = gui.pattern.value
        rename_list = list[str]()
        for layer in self.mole_layers.list(
            include=include, exclude=exclude, pattern=pattern
        ):
            old_name = layer.name
            new_name = old_name.replace(old, new)
            if old_name != new_name:
                rename_list.append((old_name, new_name))
        if len(rename_list) == 0:
            label.value = "No change."
        else:
            label.value = "\n".join(f"{old} --> {new}" for old, new in rename_list)

    gui.insert(-1, label)
    label.reset_choices = _on_change  # hack


@setup_function_gui(CylindraMainWidget.delete_molecules)
def _setup_delete_molecules(self: CylindraMainWidget, gui: FunctionGui):
    label = Label(value="No change.")

    @gui.changed.connect
    def _on_change():
        if self.parent_viewer is None:
            return
        include: str = gui.include.value
        exclude: str = gui.exclude.value
        pattern: str = gui.pattern.value
        delete_list = list[str]()
        for layer in self.mole_layers.list(
            include=include, exclude=exclude, pattern=pattern
        ):
            delete_list.append(layer.name)
        if len(delete_list) == 0:
            label.value = "No change."
        else:
            if len(delete_list) < 8:
                label.value = "\n".join(f"{name}" for name in delete_list)
            else:
                label.value = (
                    "\n".join(f"{name}" for name in delete_list[:8]) + "\n    ..."
                )

    gui.insert(-1, label)
    label.reset_choices = _on_change  # hack


@contextmanager
def _temp_layer_colors(layer: MoleculesLayer):
    """Temporarily change the colors of a layer and restore them afterwards."""
    fc = layer.face_color
    ec = layer.edge_color
    fcmap = layer.face_colormap
    ecmap = layer.edge_colormap
    fclim = layer.face_contrast_limits
    eclim = layer.edge_contrast_limits
    info = layer.colormap_info
    try:
        yield
    finally:
        layer.face_color = fc
        layer.edge_color = ec
        layer.face_colormap = fcmap
        layer.edge_colormap = ecmap
        layer.face_contrast_limits = fclim
        layer.edge_contrast_limits = eclim
        layer._colormap_info = info


def _update_existing_layer(self: CylindraMainWidget, out: Molecules) -> MoleculesLayer:
    viewer = self.parent_viewer
    if PREVIEW_LAYER_NAME in viewer.layers:
        layer: MoleculesLayer = viewer.layers[PREVIEW_LAYER_NAME]
        layer.molecules = out
    else:
        layer = self.add_molecules(out, name=PREVIEW_LAYER_NAME)
    return layer
