from typing import NamedTuple
import numpy as np
from cylindra.widgets.sta import SubtomogramAveraging
from cylindra._custom_layers import MoleculesLayer
from acryo import Molecules

from magicgui import magicgui
from magicclass.ext.pyqtgraph import plot_api as plt


class DemoResult(NamedTuple):
    molecules: list[Molecules]
    energies: np.ndarray
    temps: np.ndarray


def mesh_annealing_demo(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    template_path,
    mask_params=None,
    max_shifts=(0.6, 0.6, 0.6),
    cutoff: float = 0.5,
    interpolation: int = 3,
    distance_range_long: tuple[float, float] = (3.9, 4.4),
    distance_range_lat: tuple[float, float] = (4.7, 5.3),
    upsample_factor: int = 3,
):
    parent = self._get_parent()
    molecules = layer.molecules

    landscape = self.construct_landscape(
        layer=layer,
        template_path=template_path,
        mask_params=mask_params,
        max_shifts=max_shifts,
        cutoff=cutoff,
        interpolation=interpolation,
        upsample_factor=upsample_factor,
    )
    spl = layer.source_spline
    annealing = landscape.annealing_model(spl, distance_range_long, distance_range_lat)
    _model = annealing.with_seed(seed=0)
    energies = [_model.energy()]

    all_molecules = [molecules]
    temps = []
    for _ in range(1000):
        _model.simulate(10000)
        energies.append(_model.energy())

        offset = landscape.offset
        all_shifts_px = ((_model.shifts() - offset) / upsample_factor).reshape(-1, 3)
        all_shifts = all_shifts_px * parent.tomogram.scale

        all_molecules.append(molecules.translate_internal(all_shifts))
        temps.append(_model.temperature())

    result = DemoResult(all_molecules, np.array(energies), np.array(temps))

    pltw = plt.figure()
    plt.plot(result.energies, name="Energy")
    pltw.xlabel = "Iteration (x10^4)"
    pltw.ylabel = "Energy"
    layer0 = pltw.add_infline((0, 0), 90)

    plt_temp = plt.figure()
    plt.plot(result.temps, name="Temperature")
    plt_temp.xlabel = "Iteration (x10^4)"
    plt_temp.ylabel = "Temperature"
    layer1 = plt_temp.add_infline((0, 0), 90)

    viewer = self.parent_viewer
    viewer.window.add_dock_widget(pltw, area="right", name="Energy")
    viewer.window.add_dock_widget(plt_temp, area="right", name="Temperature")

    @magicgui(auto_call=True, x={"max": 1000, "widget_type": "Slider"})
    def fn(x: int):
        layer0.pos = (x, 0)
        layer1.pos = (x, 0)
        viewer.layers[-1].molecules = result.molecules[x]

    viewer.window.add_dock_widget(fn, area="right")
    return result
