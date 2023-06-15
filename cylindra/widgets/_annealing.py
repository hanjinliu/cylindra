from __future__ import annotations

from typing import Iterable, TYPE_CHECKING, NamedTuple
import matplotlib.pyplot as plt

from magicgui.widgets import FunctionGui
from magicclass import get_function_gui
from magicclass.logging import getLogger
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas

import numpy as np

from cylindra.types import MoleculesLayer
from cylindra.const import MoleculesHeader as Mole, nm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from cylindra._cylindra_ext import CylindricAnnealingModel
    from .sta import SubtomogramAveraging

_Logger = getLogger("cylindra")


def get_annealing_model(
    layer: MoleculesLayer,
    max_shifts: tuple[nm, nm, nm],
    scale: nm,
    upsample_factor: int = 5,
) -> CylindricAnnealingModel:
    from cylindra._cylindra_ext import CylindricAnnealingModel

    molecules = layer.molecules
    scale_factor = scale / upsample_factor
    if spl := layer.source_spline:
        cyl = spl.cylinder_model()
        _nrise, _npf = cyl.nrise, cyl.shape[1]
    else:
        raise ValueError(f"{layer!r} does not have a valid source spline.")

    _max_shifts = np.asarray(max_shifts, dtype=np.float32)
    max_shifts_px = (_max_shifts / scale_factor).astype(np.int32) * scale_factor
    m0 = molecules.translate_internal(-max_shifts_px)

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


def _get_distance_data(layer, scale, upsample_factor: int):
    annealing = get_annealing_model(layer, (0, 0, 0), scale, upsample_factor)
    data_lon = annealing.longitudinal_distances()
    data_lat = annealing.lateral_distances()
    return data_lon, data_lat


def preview_single(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    distance_range_long: tuple[nm, nm],
    distance_range_lat: tuple[nm, nm],
    upsample_factor: int,
):
    fgui = get_function_gui(self.align_all_annealing)
    parent = self._get_parent()
    scale = parent.tomogram.scale
    data_lon, data_lat = _get_distance_data(layer, scale, upsample_factor)

    canvas = QtMultiPlotCanvas(ncols=2)
    parent._active_widgets.add(canvas)
    lon_hist = canvas[0].add_hist(data_lon, bins=24, density=False, name="Longitudinal")
    lon_low = canvas[0].add_infline(
        (distance_range_long[0], 0), 90, color="yellow", ls=":"
    )
    lon_high = canvas[0].add_infline(
        (distance_range_long[1], 0), 90, color="yellow", ls=":"
    )
    canvas[0].add_infline((0, 0), 0, color="gray")
    canvas[0].title = "Longitudinal distances"
    lat_hist = canvas[1].add_hist(data_lat, bins=24, density=False, name="Lateral")
    lat_low = canvas[1].add_infline(
        (distance_range_lat[0], 0), 90, color="yellow", ls=":"
    )
    lat_high = canvas[1].add_infline(
        (distance_range_lat[1], 0), 90, color="yellow", ls=":"
    )
    canvas[1].add_infline((0, 0), 0, color="gray")
    canvas[1].title = "Lateral distances"
    canvas.native.setParent(parent.native, canvas.native.windowFlags())

    # connect value change signals
    @fgui.layer.changed.connect
    def _layer_changed(val: MoleculesLayer):
        data_lon, data_lat = _get_distance_data(val, scale, upsample_factor)
        lon_hist.set_hist(data_lon)
        lat_hist.set_hist(data_lat)

    @fgui.distance_range_long.changed.connect
    def _long_changed(val: tuple[float, float]):
        lon_low.pos = (val[0], 0)
        lon_high.pos = (val[1], 0)

    @fgui.distance_range_lat.changed.connect
    def _lat_changed(val: tuple[float, float]):
        lat_low.pos = (val[0], 0)
        lat_high.pos = (val[1], 0)

    canvas.show()

    is_active = yield
    if not is_active:
        fgui.layer.changed.disconnect(_layer_changed)
        fgui.distance_range_long.changed.disconnect(_long_changed)
        fgui.distance_range_lat.changed.disconnect(_lat_changed)
        canvas.close()
    return None


def preview_multiple(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    distance_range_long: tuple[nm, nm],
    distance_range_lat: tuple[nm, nm],
    upsample_factor: int,
):
    fgui = get_function_gui(self.align_all_annealing_multi_template)
    parent = self._get_parent()
    scale = parent.tomogram.scale
    data_lon, data_lat = _get_distance_data(layer, scale, upsample_factor)

    canvas = QtMultiPlotCanvas(ncols=2)
    parent._active_widgets.add(canvas)
    lon_hist = canvas[0].add_hist(data_lon, bins=24, density=False, name="Longitudinal")
    lon_low = canvas[0].add_infline(
        (distance_range_long[0], 0), 90, color="yellow", ls=":"
    )
    lon_high = canvas[0].add_infline(
        (distance_range_long[1], 0), 90, color="yellow", ls=":"
    )
    canvas[0].add_infline((0, 0), 0, color="gray")
    canvas[0].title = "Longitudinal distances"
    lat_hist = canvas[1].add_hist(data_lat, bins=24, density=False, name="Lateral")
    lat_low = canvas[1].add_infline(
        (distance_range_lat[0], 0), 90, color="yellow", ls=":"
    )
    lat_high = canvas[1].add_infline(
        (distance_range_lat[1], 0), 90, color="yellow", ls=":"
    )
    canvas[1].add_infline((0, 0), 0, color="gray")
    canvas[1].title = "Lateral distances"
    canvas.native.setParent(parent.native, canvas.native.windowFlags())

    # connect value change signals
    @fgui.layer.changed.connect
    def _layer_changed(val: MoleculesLayer):
        data_lon, data_lat = _get_distance_data(val, scale, upsample_factor)
        lon_hist.set_hist(data_lon)
        lat_hist.set_hist(data_lat)

    @fgui.distance_range_long.changed.connect
    def _long_changed(val: tuple[float, float]):
        lon_low.pos = (val[0], 0)
        lon_high.pos = (val[1], 0)

    @fgui.distance_range_lat.changed.connect
    def _lat_changed(val: tuple[float, float]):
        lat_low.pos = (val[0], 0)
        lat_high.pos = (val[1], 0)

    canvas.show()

    is_active = yield
    if not is_active:
        fgui.layer.changed.disconnect(_layer_changed)
        fgui.distance_range_long.changed.disconnect(_long_changed)
        fgui.distance_range_lat.changed.disconnect(_lat_changed)
        canvas.close()
    return None


def get_annealing_results(
    annealing: CylindricAnnealingModel,
    initial_temperature: float,
    seeds: Iterable[int],
    batch_size: int,
) -> list[AnnealingResult]:
    from dask import array as da, delayed

    @delayed
    def _run(seed: int) -> AnnealingResult:
        _model = annealing.with_seed(seed)
        rng = np.random.default_rng(seed)
        loc_shape = _model.local_shape()
        shifts = np.stack(
            [rng.integers(0, s0, _model.node_count()) for s0 in loc_shape], axis=1
        )
        _model.set_shifts(shifts)
        energies = [_model.energy()]
        while (
            _model.temperature() > initial_temperature * 1e-4
            and _model.optimization_state() == "not_converged"
        ):
            _model.simulate(batch_size)
            energies.append(_model.energy())
        return AnnealingResult(_model, np.array(energies), energies[-1])

    tasks = [_run(s) for s in seeds]
    results: list[AnnealingResult] = da.compute(tasks)[0]
    if all(result.model.optimization_state() == "failed" for result in results):
        raise RuntimeError(
            "Failed to optimize for all trials. You may check the distance range."
        )
    elif not any(
        result.model.optimization_state() == "converged" for result in results
    ):
        _Logger.print("Optimization did not converge for any trial.")

    _Logger.print_table(
        {
            "Iteration": [r.model.iteration() for r in results],
            "Score": [-r.energy for r in results],
            "State": [r.model.optimization_state() for r in results],
        }
    )
    return results


class AnnealingResult(NamedTuple):
    model: CylindricAnnealingModel
    energies: NDArray[np.float32]
    energy: float


def plot_annealing_result(results: list[AnnealingResult], batch_size):
    for i, r in enumerate(results):
        _x = np.arange(r.energies.size) * 1e-6 * batch_size
        plt.plot(_x, -r.energies, label=f"{i}", alpha=0.5)
    plt.xlabel("Repeat (x10^6)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
