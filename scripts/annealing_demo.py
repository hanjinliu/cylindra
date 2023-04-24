from typing import NamedTuple
import numpy as np
from cylindra.widgets.sta import SubtomogramAveraging
from cylindra._molecules_layer import MoleculesLayer
from acryo import alignment, Molecules

from magicgui import magicgui
from magicclass.ext.pyqtgraph import plot_api as plt


class DemoResult(NamedTuple):
    molecules: list[Molecules]
    energies: np.ndarray
    temps: np.ndarray


def boltzmann_demo(
    self: SubtomogramAveraging,
    layer: MoleculesLayer,
    template_path,
    mask_params=None,
    tilt_range=None,
    max_shifts=(0.6, 0.6, 0.6),
    cutoff: float = 0.5,
    interpolation: int = 3,
    distance_range_long: tuple[float, float] = (3.9, 4.4),
    distance_range_lat: tuple[float, float] = (4.7, 5.3),
    upsample_factor: int = 3,
):
    from cylindra._cpp_ext import CylindricAnnealingModel

    parent = self._get_parent()
    molecules = layer.molecules
    shape_nm = self._get_shape_in_nm()
    loader = parent.tomogram.get_subtomogram_loader(
        molecules, shape=shape_nm, order=interpolation
    )
    template, mask = loader.normalize_input(
        template=self.params._get_template(path=template_path),
        mask=self.params._get_mask(params=mask_params),
    )
    max_shifts_px = tuple(s / parent.tomogram.scale for s in max_shifts)
    search_size = tuple(int(px * upsample_factor) * 2 + 1 for px in max_shifts_px)
    model = alignment.ZNCCAlignment.with_params(
        cutoff=cutoff,
        tilt_range=tilt_range,
    )

    score_dsk = loader.construct_landscape(
        template,
        mask=mask,
        max_shifts=max_shifts,
        upsample=upsample_factor,
        alignment_model=model,
    )

    score: np.ndarray = score_dsk.compute()
    scale = parent.tomogram.scale
    m0 = molecules.translate_internal(-(np.array(max_shifts) - scale) / 2)

    dist_lon = np.array(distance_range_long) / scale * upsample_factor
    dist_lat = np.array(distance_range_lat) / scale * upsample_factor
    spl = layer.source_component
    _cyl_model = spl.cylinder_model()
    _grid_shape = _cyl_model.shape
    _vec_shape = _grid_shape + (3,)
    energy = -score

    time_const = molecules.pos.size * np.product(search_size)
    initial_temperature = np.std(energy) * 4

    # construct the annealing model
    annealing = (
        CylindricAnnealingModel()
        .set_graph(
            energy.reshape(_grid_shape + search_size),
            (m0.pos / scale * upsample_factor).reshape(_vec_shape),
            m0.z.reshape(_vec_shape),
            m0.y.reshape(_vec_shape),
            m0.x.reshape(_vec_shape),
            _cyl_model.nrise,
        )
        .set_reservoir(
            temperature=initial_temperature,
            time_constant=time_const,
        )
        .set_box_potential(
            *dist_lon,
            *dist_lat,
        )
    )

    _model = annealing.with_seed(seed=0)
    energies = [_model.energy()]

    all_molecules = []
    temps = []
    for _ in range(1000):
        _model.simulate(10000)
        energies.append(_model.energy())

        offset = (np.array(max_shifts_px) * upsample_factor).astype(np.int32)
        all_shifts_px = ((_model.shifts() - offset) / upsample_factor).reshape(-1, 3)
        all_shifts = all_shifts_px * scale

        all_molecules.append(molecules.translate_internal(all_shifts))
        temps.append(_model.reservoir().temperature())

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
