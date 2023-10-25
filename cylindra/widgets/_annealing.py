from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

from magicgui.widgets import FunctionGui
from magicclass import get_function_gui
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas

import numpy as np

from cylindra.types import MoleculesLayer
from cylindra.const import MoleculesHeader as Mole, nm

if TYPE_CHECKING:
    from cylindra._cylindra_ext import CylindricAnnealingModel
    from cylindra.widgets.sta import SubtomogramAveraging
    from cylindra.components.landscape import AnnealingResult


def get_annealing_model(
    layer: MoleculesLayer,
    max_shifts: tuple[nm, nm, nm],
    scale_factor: float,  # scale / upsample_factor
) -> CylindricAnnealingModel:
    from cylindra._cylindra_ext import CylindricAnnealingModel

    molecules = layer.molecules
    if spl := layer.source_spline:
        cyl = spl.cylinder_model()
        _nrise, _npf = cyl.nrise, cyl.shape[1]
    else:
        raise ValueError(f"{layer!r} does not have a valid source spline.")

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


def get_distances(layer: MoleculesLayer, scale: nm, upsample_factor: int):
    annealing = get_annealing_model(layer, (0, 0, 0), scale / upsample_factor)
    data_lon = annealing.longitudinal_distances()
    data_lat = annealing.lateral_distances()
    return data_lon, data_lat


def preview_single(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    range_long: tuple[nm, nm],
    range_lat: tuple[nm, nm],
    upsample_factor: int,
):
    fgui = get_function_gui(self.align_all_annealing)
    yield from _preview_function(
        widget=self,
        fgui=fgui,
        layer=layer,
        range_long=range_long,
        range_lat=range_lat,
        upsample_factor=upsample_factor,
    )


def _preview_function(
    widget: SubtomogramAveraging,
    fgui: FunctionGui,
    layer: MoleculesLayer,
    range_long: tuple[nm, nm],
    range_lat: tuple[nm, nm],
    upsample_factor: int,
):
    parent = widget._get_main()
    scale = parent.tomogram.scale
    data_lon, data_lat = get_distances(layer, scale, upsample_factor)

    canvas = QtMultiPlotCanvas(ncols=2)
    parent._active_widgets.add(canvas)
    lon_hist = canvas[0].add_hist(data_lon, bins=24, density=False, name="Longitudinal")
    lon_low = canvas[0].add_infline((range_long[0], 0), 90, color="yellow", ls=":")
    lon_high = canvas[0].add_infline((range_long[1], 0), 90, color="yellow", ls=":")
    canvas[0].add_infline((0, 0), 0, color="gray")
    canvas[0].title = "Longitudinal distances"
    lat_hist = canvas[1].add_hist(data_lat, bins=24, density=False, name="Lateral")
    lat_low = canvas[1].add_infline((range_lat[0], 0), 90, color="yellow", ls=":")
    lat_high = canvas[1].add_infline((range_lat[1], 0), 90, color="yellow", ls=":")
    canvas[1].add_infline((0, 0), 0, color="gray")
    canvas[1].title = "Lateral distances"
    canvas.native.setParent(parent.native, canvas.native.windowFlags())

    # connect value change signals
    @fgui.layer.changed.connect
    def _layer_changed(val: MoleculesLayer):
        data_lon, data_lat = get_distances(val, scale, upsample_factor)
        lon_hist.set_hist(data_lon)
        lat_hist.set_hist(data_lat)

    @fgui.range_long.changed.connect
    def _long_changed(val: tuple[float, float]):
        lon_low.pos = (val[0], 0)
        lon_high.pos = (val[1], 0)

    @fgui.range_lat.changed.connect
    def _lat_changed(val: tuple[float, float]):
        lat_low.pos = (val[0], 0)
        lat_high.pos = (val[1], 0)

    canvas.show()

    is_active = yield
    if not is_active:
        fgui.layer.changed.disconnect(_layer_changed)
        fgui.range_long.changed.disconnect(_long_changed)
        fgui.range_lat.changed.disconnect(_lat_changed)
        canvas.close()
    return None


def plot_annealing_result(results: list[AnnealingResult]):
    for i, r in enumerate(results):
        _x = np.arange(r.energies.size) * 1e-6 * r.batch_size
        plt.plot(_x, -r.energies, label=f"{i}", alpha=0.5)
    plt.xlabel("Repeat (x10^6)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
