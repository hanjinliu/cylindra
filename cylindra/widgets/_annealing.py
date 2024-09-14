from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from acryo import Molecules
from magicclass import get_function_gui
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas
from magicgui.widgets import FunctionGui

from cylindra._napari import LandscapeSurface, MoleculesLayer
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import nm
from cylindra.core import ACTIVE_WIDGETS

if TYPE_CHECKING:
    from cylindra._cylindra_ext import CylindricAnnealingModel
    from cylindra.components import CylSpline
    from cylindra.components.landscape import AnnealingResult
    from cylindra.widgets.sta import SubtomogramAveraging


def get_annealing_model(
    molecules: Molecules,
    spl: CylSpline,
    max_shifts: tuple[nm, nm, nm],
    scale_factor: float,  # scale / upsample_factor
) -> CylindricAnnealingModel:
    from cylindra._cylindra_ext import CylindricAnnealingModel

    if spl:
        cyl = spl.cylinder_model()
        _nrise, _npf = cyl.nrise, cyl.shape[1]
    else:
        raise ValueError("Layer does not have a valid source spline.")

    _max_shifts = np.asarray(max_shifts, dtype=np.float32)
    _max_shifts_clipped = (_max_shifts / scale_factor).astype(np.int32) * scale_factor
    m0 = molecules.translate_internal(-_max_shifts_clipped)
    return (
        CylindricAnnealingModel()
        .construct_graph(
            indices=molecules.features.select([Mole.nth, Mole.pf])
            .to_numpy()
            .astype(np.int32),
            npf=_npf,
            nrise=_nrise,
        )
        .set_graph_coordinates(
            origin=m0.pos,
            zvec=m0.z.astype(np.float32) * scale_factor,
            yvec=m0.y.astype(np.float32) * scale_factor,
            xvec=m0.x.astype(np.float32) * scale_factor,
        )
    )


def get_distances(
    molecules: Molecules,
    spl: CylSpline,
    scale_factor: nm,
):
    annealing = get_annealing_model(molecules, spl, (0, 0, 0), scale_factor)
    data_lon = annealing.longitudinal_distances()
    data_lat = annealing.lateral_distances()
    return data_lon, data_lat


def preview_single(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    range_long: tuple[str, str],
    range_lat: tuple[str, str],
    upsample_factor: int,
):
    fgui = get_function_gui(self.align_all_rma)
    scale = self._get_main().tomogram.scale
    yield from _preview_function(
        widget=self,
        fgui=fgui,
        molecules=layer.molecules,
        spline=layer.source_spline,
        range_long=range_long,
        range_lat=range_lat,
        scale_factor=scale / upsample_factor,
    )


def preview_landscape_function(
    self: SubtomogramAveraging,
    landscape_layer: LandscapeSurface,
    range_long: tuple[str, str],
    range_lat: tuple[str, str],
):
    fgui = get_function_gui(self.run_rma_on_landscape)
    yield from _preview_function(
        widget=self,
        fgui=fgui,
        molecules=landscape_layer.landscape.molecules,
        spline=landscape_layer.source_spline,
        range_long=range_long,
        range_lat=range_lat,
        scale_factor=landscape_layer.landscape.scale_factor,
    )


def _preview_function(
    widget: SubtomogramAveraging,
    fgui: FunctionGui,
    molecules: Molecules,
    spline: CylSpline,
    range_long: tuple[str, str],
    range_lat: tuple[str, str],
    scale_factor: nm,
):
    parent = widget._get_main()
    data_lon, data_lat = get_distances(molecules, spline, scale_factor)

    canvas = QtMultiPlotCanvas(ncols=2)
    ACTIVE_WIDGETS.add(canvas)
    min_lon = _eval_dist_like(range_long[0], data_lon)
    max_lon = _eval_dist_like(range_long[1], data_lon)
    min_lat = _eval_dist_like(range_lat[0], data_lat)
    max_lat = _eval_dist_like(range_lat[1], data_lat)
    lon_hist = canvas[0].add_hist(data_lon, bins=24, density=False, name="Longitudinal")
    color_ok = "yellow"
    color_ng = "red"
    if min_lon is None:
        min_lon = data_lon.min()
    if max_lon is None:
        max_lon = data_lon.max()
    if min_lat is None:
        min_lat = data_lat.min()
    if max_lat is None:
        max_lat = data_lat.max()
    lon_low = canvas[0].add_infline((min_lon, 0), 90, color=color_ok, ls=":")
    lon_high = canvas[0].add_infline((max_lon, 0), 90, color=color_ok, ls=":")
    canvas[0].add_infline((0, 0), 0, color="gray")
    canvas[0].title = "Longitudinal distances"
    lat_hist = canvas[1].add_hist(data_lat, bins=24, density=False, name="Lateral")
    lat_low = canvas[1].add_infline((min_lat, 0), 90, color=color_ok, ls=":")
    lat_high = canvas[1].add_infline((max_lat, 0), 90, color=color_ok, ls=":")
    canvas[1].add_infline((0, 0), 0, color="gray")
    canvas[1].title = "Lateral distances"
    canvas.native.setParent(parent.native, canvas.native.windowFlags())

    # connect value change signals
    @fgui[0].changed.connect
    def _layer_changed(val: MoleculesLayer | LandscapeSurface):
        # When new image is opened, the source spline may be garbage collected before
        # this callback is called. Therefore, we need to check if the spline is still
        # alive.
        if val.source_spline is None:
            lon_hist.set_hist(np.zeros(0))
            lat_hist.set_hist(np.zeros(0))

        data_lon, data_lat = get_distances(
            val.molecules, val.source_spline, scale_factor
        )
        lon_hist.set_hist(data_lon)
        lat_hist.set_hist(data_lat)

    @fgui.range_long.changed.connect
    def _long_changed(val: tuple[float, float]):
        min_lon = _eval_dist_like(val[0], data_lon)
        max_lon = _eval_dist_like(val[1], data_lon)
        if min_lon is None:
            lon_low.color = color_ng
        else:
            lon_low.pos = (min_lon, 0)
            lon_low.color = color_ok
        if max_lon is None:
            lon_high.color = color_ng
        else:
            lon_high.pos = (max_lon, 0)
            lon_high.color = color_ok
        if min_lon is not None and max_lon is not None:
            if min_lon >= max_lon:
                lon_low.color = color_ng
                lon_high.color = color_ng
            else:
                lon_low.color = color_ok
                lon_high.color = color_ok

    @fgui.range_lat.changed.connect
    def _lat_changed(val: tuple[float, float]):
        min_lat = _eval_dist_like(val[0], data_lat)
        max_lat = _eval_dist_like(val[1], data_lat)
        if min_lat is None:
            lat_low.color = color_ng
        else:
            lat_low.pos = (min_lat, 0)
            lat_low.color = color_ok
        if max_lat is None:
            lat_high.color = color_ng
        else:
            lat_high.pos = (max_lat, 0)
            lat_high.color = color_ok
        if min_lat is not None and max_lat is not None:
            if min_lat >= max_lat:
                lat_low.color = color_ng
                lat_high.color = color_ng
            else:
                lat_low.color = color_ok
                lat_high.color = color_ok

    canvas.show()
    canvas.width = 600
    canvas.height = 400

    is_active = yield
    if not is_active:
        fgui[0].changed.disconnect(_layer_changed)
        fgui.range_long.changed.disconnect(_long_changed)
        fgui.range_lat.changed.disconnect(_lat_changed)
        canvas.close()
    return None


def _eval_dist_like(val: str, data: np.ndarray) -> float | None:
    try:
        out = eval(val, {"d": data, "np": np, "__builtins__": {}})
    except Exception:
        out = None
    if not isinstance(out, (int, float, np.number)):
        return None
    else:
        return float(out)


def plot_annealing_result(results: list[AnnealingResult]):
    for i, r in enumerate(results):
        _x = np.arange(r.energies.size) * 1e-6 * r.batch_size
        plt.plot(_x, -r.energies, label=f"{i}", alpha=0.5)
    plt.xlabel("Repeat (x10^6)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
