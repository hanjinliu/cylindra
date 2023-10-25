import tempfile

import napari

from scripts.user_consts import TEMPLATE_X, WOBBLE_TEMPLATES
from cylindra import start  # NOTE: Set ApplicationAttributes

from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
from cylindra.widgets import CylindraMainWidget
from cylindra.cylstructure import calc_lateral_interval
from cylindra.types import MoleculesLayer

import polars as pl


def create_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.25)
    initialize_molecules(ui)
    layer = ui.mole_layers.last()
    dtheta = 0.06
    ui.simulator.displace(
        layer, twist=pl.when(pl.col("isotype-id").eq(1)).then(-dtheta).otherwise(dtheta)
    )
    post_process_layer(ui, layer)
    # ui.split_molecules(layer, by="isotype-id")
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
        # components=[
        #     (layer_name + "_0", TEMPLATE_X),
        #     (layer_name + "_1", TEMPLATE_X),
        # ],
        components=[(layer_name, TEMPLATE_X)],
        save_dir=path,
        tilt_range=(-60, 60),
        n_tilt=61,
    )


def post_process_layer(ui: CylindraMainWidget, layer: MoleculesLayer) -> MoleculesLayer:
    ui.calculate_lattice_structure(layer=layer, props=["twist", "skew_angle"])
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
        nsr=0.1,
        bin_size=2,
        tilt_range=(-60, 60),
        height=60.0,
        seed=seed,
    )
    initialize_molecules(ui)
    ui.global_ft_analysis(splines=[0], bin_size=2)

    layer = ui.mole_layers.last()
    mole_init = post_process_layer(ui, layer).molecules

    # RMA alignment
    interv_mean = finite_mean(calc_lateral_interval(mole_init, ui.splines[0]))
    dx = 0.1
    ui.sta.align_all_annealing(
        layer=layer,
        template_path=WOBBLE_TEMPLATES,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        range_long=(3.98, 4.28),
        range_lat=(interv_mean - dx, interv_mean + dx),
        angle_max=5.0,
        upsample_factor=5,
    )
    mole_rma = post_process_layer(ui, ui.mole_layers.last()).molecules
    return flat_agg(mole_rma.features)


def flat_agg(df: pl.DataFrame) -> tuple[float, float, float, float]:
    agg = df.group_by("isotype-id").agg(
        pl.col("twist").filter(pl.col("twist").is_finite()).mean(),
        pl.col("skew_angle").filter(pl.col("skew_angle").is_finite()).mean(),
    )
    return agg["twist"][0], agg["twist"][1], agg["skew_angle"][0], agg["skew_angle"][1]


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
        for i in range(3):
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
