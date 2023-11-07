import tempfile
from matplotlib import pyplot as plt

import napari

from cylindra import start  # NOTE: Set ApplicationAttributes
from timeit import default_timer
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
from acryo import Molecules
from cylindra.widgets import CylindraMainWidget
from cylindra.const import MoleculesHeader as Mole
from cylindra.cylstructure import calc_lateral_interval
from cylindra.types import MoleculesLayer

import polars as pl
from scripts.user_consts import TEMPLATE_X

SPACING = Mole.spacing
SPACING_MEAN = f"{Mole.spacing}_mean"


def initialize_molecules(ui: CylindraMainWidget):
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 165.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.04, start=3, npf=13, radius=11.0, offsets=(0.0, 0.0)
    )


def create_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.25)
    initialize_molecules(ui)
    layer = ui.mole_layers.last()
    ui.simulator.expand(layer=layer, by=0.1, yrange=(6, 16), arange=(0, 6), allev=True)
    ui.simulator.expand(
        layer=layer, by=0.1, yrange=(22, 32), arange=(7, 13), allev=True
    )
    ui.calculate_lattice_structure(layer=layer, props=["spacing"])
    return layer.molecules


def save_tilt_series(ui: CylindraMainWidget, path: Path):
    path = Path(path)
    layer_name = ui.mole_layers.last().name
    ui.simulator.simulate_tilt_series(
        components=[(layer_name, TEMPLATE_X)],
        save_dir=path,
        tilt_range=(-60, 60),
        n_tilt=61,
    )


def rmsd_pos(est: Molecules, ans: Molecules) -> float:
    return np.sqrt(np.mean((est.pos - ans.pos) ** 2))


def rmsd_spacing(est: Molecules, ans: Molecules) -> float:
    val_a = est.features[SPACING]
    val_b = ans.features[SPACING]
    d = val_a - val_b
    d = d.filter(d.is_finite())
    return np.sqrt((d**2).mean())


def rmsd_spacing_mean(est: Molecules, ans: Molecules) -> float:
    val_a = est.features[SPACING_MEAN]
    val_b = ans.features[SPACING]
    d = val_a - val_b
    d = d.filter(d.is_finite())
    return np.sqrt((d**2).mean())


def post_process_layer(ui: CylindraMainWidget, layer: MoleculesLayer) -> MoleculesLayer:
    ui.calculate_lattice_structure(layer=layer, props=["spacing"])
    ui.convolve_feature(
        layer=layer,
        target=SPACING,
        method="mean",
        footprint=[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
    )
    return layer


def run_one(
    ui: CylindraMainWidget, image_path: Path, mole_truth: Molecules, seed: int = 0
):
    ui.simulator.simulate_tomogram_from_tilt_series(
        image_path,
        nsr=2.5,
        bin_size=2,
        tilt_range=(-60, 60),
        height=60.0,
        seed=seed,
    )
    initialize_molecules(ui)
    ui.global_cft_analysis(splines=[0], bin_size=2)

    layer = ui.mole_layers.last()
    nth, npf = layer.regular_shape()
    ui.simulator.expand(
        layer=layer,
        by=0.02,
        yrange=(0, nth),
        arange=(0, npf),
        allev=True,
    )
    mole_init = post_process_layer(ui, layer).molecules
    t0 = default_timer()
    ui.sta.construct_landscape(
        layer=layer,
        template_path=TEMPLATE_X,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
    )
    t0 = default_timer() - t0
    land_layer = ui.parent_viewer.layers[-1]

    # conventional alignment
    t1 = default_timer()
    ui.sta.run_align_on_landscape(land_layer)
    t1 = default_timer() - t1
    mole_cnv = post_process_layer(ui, ui.mole_layers.last()).molecules

    # Viterbi alignment
    t2 = default_timer()
    ui.sta.run_viterbi_on_landscape(land_layer, range_long=(3.98, 4.28), angle_max=5.0)
    t2 = default_timer() - t2
    mole_vit = post_process_layer(ui, ui.mole_layers.last()).molecules

    # RMA alignment
    intervs = calc_lateral_interval(mole_init, ui.splines[0])
    interv_mean = intervs.filter(intervs.is_finite()).mean()
    dx = 0.1
    t3 = default_timer()
    ui.sta.run_annealing_on_landscape(
        land_layer,
        range_long=(3.98, 4.28),
        range_lat=(interv_mean - dx, interv_mean + dx),
        angle_max=5.0,
    )
    t3 = default_timer() - t3
    mole_rma = post_process_layer(ui, ui.mole_layers.last()).molecules

    pos_values = [
        rmsd_pos(mole_init, mole_truth),
        rmsd_pos(mole_cnv, mole_truth),
        rmsd_pos(mole_vit, mole_truth),
        rmsd_pos(mole_rma, mole_truth),
    ]

    spacing_values = [
        rmsd_spacing(mole_init, mole_truth),
        rmsd_spacing(mole_cnv, mole_truth),
        rmsd_spacing_mean(mole_cnv, mole_truth),
        rmsd_spacing(mole_vit, mole_truth),
        rmsd_spacing_mean(mole_vit, mole_truth),
        rmsd_spacing(mole_rma, mole_truth),
        rmsd_spacing_mean(mole_rma, mole_truth),
    ]

    time_values = [t0 + t1, t0 + t2, t0 + t3]

    return pos_values, spacing_values, time_values


def show_dataframe(ui: CylindraMainWidget, df: pl.DataFrame):
    view = DataFrameView()
    view.value = df
    ui.parent_viewer.window.add_dock_widget(view)


def main():
    ui = start()
    mole_truth = create_microtubule(ui)
    pos_list = []
    spacing_list = []
    time_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tilt_series(ui, tmpdir)
        for i in range(10):
            pos, spacing, times = run_one(
                ui,
                Path(tmpdir) / "image.mrc",
                mole_truth,
                seed=i,
            )
            pos_list.append(pos)
            spacing_list.append(spacing)
            time_list.append(times)
            print("Done:", i)

    df_pos = pl.DataFrame(
        np.array(pos_list), schema=["initial", "conventional", "viterbi", "rma"]
    )
    df_spacing = pl.DataFrame(
        np.array(spacing_list),
        schema=[
            "initial",
            "conventional",
            "conventional_mean",
            "viterbi",
            "viterbi_mean",
            "rma",
            "rma_mean",
        ],
    )
    df_time = pl.DataFrame(
        np.array(time_list), schema=["conventional", "viterbi", "rma"]
    )

    show_dataframe(ui, df_pos)
    show_dataframe(ui, df_spacing)
    show_dataframe(ui, df_time)
    for i in [-3, -2, -1]:
        layer = ui.mole_layers.nth(i)
        ui.sta.calculate_fsc(layer, template_path=TEMPLATE_X, mask_params=(0.8, 0.8))
        ui.paint_molecules(layer, "spacing", limits=(4.0, 4.28))
        ui.MoleculesMenu.View.plot_molecule_feature(layer)
        plt.show()
        ui.paint_molecules(layer, "spacing_mean", limits=(4.0, 4.28))
        ui.MoleculesMenu.View.plot_molecule_feature(layer)
        plt.show()
    ui.add_molecules(mole_truth, name="truth", source=ui.splines[0])
    ui.calculate_lattice_structure(layer="truth", props=["spacing"])
    ui.paint_molecules("truth", "spacing", limits=(4.0, 4.28))
    ui.MoleculesMenu.View.plot_molecule_feature("truth")
    plt.show()
    ui.parent_viewer.show(block=True)


if __name__ == "__main__":
    main()
    napari.run()