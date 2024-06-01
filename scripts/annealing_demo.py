import warnings
from typing import NamedTuple

import napari
import numpy as np
from acryo import Molecules
from magicclass.ext.pyqtgraph import plot_api as plt
from magicgui import magicgui

from cylindra import start
from cylindra._napari import MoleculesLayer
from cylindra.widgets.sta import SubtomogramAveraging
from scripts.user_consts import TEMPLATE_X


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
    angle_max: float = 5.0,
    upsample_factor: int = 5,
    batch_size: int | None = None,
    **kwargs,
):
    assert isinstance(self, SubtomogramAveraging)
    parent = self._get_main()
    molecules = layer.molecules

    landscape = self._construct_landscape(
        molecules=layer.molecules,
        template_path=template_path,
        mask_params=mask_params,
        max_shifts=max_shifts,
        cutoff=cutoff,
        order=interpolation,
        upsample_factor=upsample_factor,
    )
    spl = layer.source_spline
    annealing = landscape.annealing_model(
        spl,
        distance_range_long,
        distance_range_lat,
        angle_max=angle_max,
    )
    _model = annealing.with_seed(seed=0)
    energies = [_model.energy()]

    all_molecules = [molecules]
    temps = []
    if batch_size is None:
        batch_size = np.prod(landscape.energies.shape) // 25
    for _ in range(1000):
        _model.simulate(batch_size)
        energies.append(_model.energy())

        offset = landscape.offset
        all_shifts_px = ((_model.shifts() - offset) / upsample_factor).reshape(-1, 3)
        all_shifts = all_shifts_px * parent.tomogram.scale

        all_molecules.append(molecules.translate_internal(all_shifts))
        temps.append(_model.temperature())
        if temps[-1] / temps[0] < 1e-4:
            break

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

    mole_layer = parent.mole_layers.last()
    mole_layer.point_size = 5.0
    mole_layer.view_ndim = 2

    @magicgui(
        auto_call=True, x={"max": len(result.molecules) - 1, "widget_type": "Slider"}
    )
    def fn(x: int):
        with warnings.catch_warnings(), parent.macro.blocked():
            warnings.simplefilter("ignore")
            layer0.pos = (x, 0)
            layer1.pos = (x, 0)
            mole_layer.molecules = result.molecules[x]
            parent.calculate_lattice_structure(mole_layer)
            parent.convolve_feature(
                mole_layer, "spacing", footprint=[[1, 1, 1], [1, 1, 1], [0, 0, 0]]
            )
            parent.paint_molecules(
                mole_layer, color_by="spacing_mean", limits=(4.0, 4.28)
            )
            mole_layer.edge_color = "black"

    viewer.window.add_dock_widget(fn, area="right")
    return result


if __name__ == "__main__":
    ui = start()
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.2615)
    ui.simulator.create_straight_line(start=(30.0, 15.0, 30.0), end=(30.0, 165.0, 30.0))
    ui.simulator.generate_molecules(
        spline=0,
        spacing=4.08,
        twist=0.04,
        start=3,
        npf=13,
        radius=11.0,
        offsets=(0.0, 0.0),
        update_glob=True,
    )
    ui.simulator.expand(
        layer="Mole(Sim)-0", by=0.1, yrange=(6, 16), arange=(0, 6), allev=True
    )
    ui.simulator.expand(
        layer="Mole(Sim)-0", by=0.1, yrange=(22, 32), arange=(7, 13), allev=True
    )
    ui.simulator.simulate_tomogram_and_open(
        components=[("Mole(Sim)-0", TEMPLATE_X)],
        nsr=1.5,
        bin_size=[4],
        tilt_range=(-60.0, 60.0),
        n_tilt=41,
        interpolation=3,
        seed=36,
    )
    mesh_annealing_demo(
        ui.sta,
        ui.mole_layers.last(),
        template_path=TEMPLATE_X,
        mask_params=(0.3, 0.8),
        distance_range_long=(4.00, 4.28),
        distance_range_lat=("-0.1", "+0.1"),
    )
    ui.parent_viewer.show(block=True)
    napari.run()
