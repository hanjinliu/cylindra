import warnings
from typing import NamedTuple

import napari
import numpy as np
from magicclass.ext.pyqtgraph import plot_api as plt
from magicgui import magicgui

from cylindra import start
from cylindra._napari import MoleculesLayer
from cylindra.components import CylSpline
from cylindra.components.landscape import Landscape
from cylindra.widgets.sta import SubtomogramAveraging
from scripts.user_consts import TEMPLATE_CYL, TEMPLATE_X


class DemoResult(NamedTuple):
    splines: list[CylSpline]
    energies: np.ndarray
    temps: np.ndarray


def run_rfa_demo(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    template_path,
    mask_params=None,
    max_shifts=(4.5, 3, 4.5),
    cutoff: float = 0.5,
    interpolation: int = 3,
    distance_range_long: tuple[float, float] = (8, 8.5),
    angle_max: float | None = 2.0,
    upsample_factor: int = 2,
    batch_size: int | None = None,
    bin_size: int = 2,
    **kwargs,
):
    assert isinstance(self, SubtomogramAveraging)
    parent = self._get_main()
    molecules = layer.molecules

    self.construct_landscape(
        layer,
        template_path=template_path,
        mask_params=mask_params,
        max_shifts=max_shifts,
        rotations=[(3, 3), (9, 4.5), (3, 3)],
        cutoff=cutoff,
        interpolation=interpolation,
        upsample_factor=upsample_factor,
        bin_size=bin_size,
    )
    landscape: Landscape = self.parent_viewer.layers[-1].landscape
    annealing = landscape.filamentous_annealing_model(
        distance_range_long,
        angle_max=angle_max,
        temperature_time_const=3,
    )
    _model = annealing.with_seed(seed=0)
    energies = [_model.energy()]

    all_splines = [CylSpline().fit(molecules.pos, err_max=0.5)]
    temps = []
    if batch_size is None:
        batch_size = int(np.prod(landscape.energies.shape) / (bin_size**3) / 2)
    for _ in range(500):
        _model.simulate(batch_size)
        energies.append(_model.energy())

        offset = landscape.offset
        all_shifts_px = ((_model.shifts() - offset) / upsample_factor).reshape(-1, 3)
        all_shifts = all_shifts_px * parent.tomogram.scale * bin_size

        mole_trans = molecules.translate_internal(all_shifts)
        all_splines.append(CylSpline().fit(mole_trans.pos, err_max=0.5))
        temps.append(_model.temperature())
        if temps[-1] / temps[0] < 1e-4:
            break

    result = DemoResult(all_splines, np.array(energies), np.array(temps))

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

    _u_default = np.linspace(0, 1, 100)
    shapes_layer = viewer.add_shapes(
        all_splines[0].map(_u_default),
        edge_color="lime",
        shape_type="path",
        blending="translucent_no_depth",
    )

    @magicgui(
        auto_call=True, x={"max": len(result.splines) - 1, "widget_type": "Slider"}
    )
    def fn(x: int):
        with warnings.catch_warnings(), parent.macro.blocked():
            warnings.simplefilter("ignore")
            layer0.pos = (x, 0)
            layer1.pos = (x, 0)
            shapes_layer.data = result.splines[x].map(_u_default)

    viewer.window.add_dock_widget(fn, area="right")
    viewer.layers["Mole(Sim)-0"].visible = False
    ui.filter_reference_image()
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
    ui.simulator.simulate_tomogram_and_open(
        components=[("Mole(Sim)-0", TEMPLATE_X)],
        nsr=1.5,
        bin_size=[2, 4],
        tilt_range=(-60.0, 60.0),
        n_tilt=41,
        interpolation=3,
        seed=36,
    )
    ui.register_path(
        [
            [29.4, 152, 30.9],
            [29.4, 123, 30.2],
            [29.4, 94, 29.5],
            [29.4, 65, 28.8],
            [29.4, 35, 28.1],
        ]
    )
    ui.map_along_spline(
        1, molecule_interval=8.1, orientation=None, rotate_molecules=False
    )
    run_rfa_demo(
        ui.sta,
        ui.mole_layers.last(),
        template_path=TEMPLATE_CYL,
    )
    ui.parent_viewer.show(block=True)
    napari.run()
