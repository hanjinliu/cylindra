import tempfile

import napari

from cylindra import start  # NOTE: Set ApplicationAttributes

from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
from acryo import Molecules
from cylindra.widgets import CylindraMainWidget
from cylindra.const import MoleculesHeader as Mole
from cylindra.cylstructure import calc_lateral_interval
from cylindra.types import MoleculesLayer

import polars as pl

from .user_consts import TEMPLATE_A, TEMPLATE_B, TEMPLATE_X, WOBBLE_TEMPLATES


def create_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.25)
    initialize_molecules(ui)
    layer = ui.mole_layers.last()
    ui.simulator.displace(
        layer, twist=pl.when(pl.col("isotype-id")).then(-0.02).otherwise(0.02)
    )
    ui.calculate_lattice_structure(layer=layer, props=["twist", "skew"])
    ui.split_molecules(layer, by="isotype-id")
    return layer.molecules


def initialize_molecules(ui: CylindraMainWidget):
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 165.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.04, start=3, npf=13, radius=11.0, offsets=(0.0, 0.0)
    )
    ui.sta.seam_search_manually(ui.mole_layers.last(), location=0)


def save_tilt_series(ui: CylindraMainWidget, path: Path):
    path = Path(path)
    layer_name = ui.mole_layers.first().name
    ui.simulator.simulate_tilt_series(
        components=[
            (layer_name + "_0", TEMPLATE_A),
            (layer_name + "_1", TEMPLATE_B),
        ],
        save_dir=path,
        tilt_range=(-60, 60),
        n_tilt=61,
    )


def post_process_layer(ui: CylindraMainWidget, layer: MoleculesLayer) -> MoleculesLayer:
    ui.calculate_lattice_structure(layer=layer, props=["twist", "skew"])
    return layer


def finite_mean(series: pl.Series) -> float:
    return series.filter(series.is_finite()).mean()


def run_one(
    ui: CylindraMainWidget,
    image_path: Path,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    ui.simulator.simulate_tomogram_from_tilt_series(
        image_path,
        nsr=2.5,
        bin_size=2,
        tilt_range=(-60, 60),
        height=60.0,
        seed=seed,
    )
    initialize_molecules(ui)
    ui.global_ft_analysis(splines=[0], bin_size=2)

    layer = ui.mole_layers.last()
    mole_init = post_process_layer(ui, layer).molecules

    # start comparing alignment methods
    shared_kwargs = dict(
        template_paths=WOBBLE_TEMPLATES,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
    )

    # conventional alignment
    ui.sta.align_all_multi_template(layers=[layer], bin_size=1, **shared_kwargs)

    # RMA alignment
    interv_mean = finite_mean(calc_lateral_interval(mole_init, ui.splines[0]))
    dx = 0.1
    ui.sta.align_all_annealing_multi_template(
        layer=layer,
        **shared_kwargs,
        distance_range_long=(3.98, 4.28),
        distance_range_lat=(interv_mean - dx, interv_mean + dx),
        angle_max=5.0,
        upsample_factor=5,
    )
    mole_rma = post_process_layer(ui, ui.mole_layers.last()).molecules
    return flat_agg(mole_rma.features)


def flat_agg(df: pl.DataFrame) -> tuple[float, float, float, float]:
    agg = df.group_by("isotype-id").agg(
        pl.col("twist").filter(pl.col("twist").is_finite()).mean(),
        pl.col("skew").filter(pl.col("skew").is_finite()).mean(),
    )
    return agg["twist"][0], agg["twist"][1], agg["skew"][0], agg["skew"][1]


def show_dataframe(ui: CylindraMainWidget, df: pl.DataFrame):
    view = DataFrameView()
    view.value = df
    ui.parent_viewer.window.add_dock_widget(view)


def main():
    ui = start()
    create_microtubule(ui)
    ans = flat_agg(ui.mole_layers.first().molecules.features)
    df_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tilt_series(ui, tmpdir)
        for i in range(10):
            out = run_one(
                ui,
                Path(tmpdir) / "image.mrc",
                seed=i,
            )
            df_list.append(ans + out)

    df = pl.DataFrame(
        np.array(df_list),
        schema=[
            "twist_a_ans",
            "twist_b_ans",
            "skew_a_ans",
            "skew_b_ans",
            "twist_a",
            "twist_b",
            "skew_a",
            "skew_b",
        ],
    )
    show_dataframe(ui, df)


if __name__ == "__main__":
    main()
    napari.run()
