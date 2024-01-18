import tempfile

import napari
import polars as pl
from magicclass.ext.polars import DataFrameView
from magicclass.types import Path

from cylindra import start
from cylindra.types import MoleculesLayer
from cylindra.widgets import CylindraMainWidget
from scripts.user_consts import TEMPLATE_X


def create_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.2615)
    initialize_molecules(ui)
    layer = ui.mole_layers.last()
    dtheta = 0.04
    ui.simulator.displace(
        layer, twist=pl.when(pl.col("isotype-id").eq(1)).then(-dtheta).otherwise(dtheta)
    )
    post_process_layer(ui, layer)
    return layer.molecules


def initialize_molecules(ui: CylindraMainWidget):
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 165.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.05, start=3, npf=13, radius=10.5, offsets=(0.0, 0.0)
    )
    ui.sta.seam_search_manually(ui.mole_layers.last(), location=0)


def save_tilt_series(ui: CylindraMainWidget, path: Path):
    path = Path(path)
    layer_name = ui.mole_layers.first().name
    ui.simulator.simulate_tilt_series(
        components=[(layer_name, TEMPLATE_X)],
        save_dir=path,
        tilt_range=(-60, 60),
        n_tilt=61,
    )


def post_process_layer(ui: CylindraMainWidget, layer: MoleculesLayer) -> MoleculesLayer:
    ui.sta.seam_search_manually(layer, location=0)
    ui.calculate_lattice_structure(layer=layer, props=["twist", "skew_angle"])
    return layer


def finite_mean(series: pl.Series) -> float:
    return series.filter(series.is_finite()).mean()


def run_one(
    ui: CylindraMainWidget,
    image_path: Path,
    seed: int = 0,
) -> pl.DataFrame:
    ui.simulator.simulate_tomogram_from_tilt_series(
        image_path,
        nsr=3.5,
        bin_size=2,
        tilt_range=(-60, 60),
        height=60.0,
        seed=seed,
    )
    initialize_molecules(ui)
    ui.global_cft_analysis(splines=[0], bin_size=2)

    layer = ui.mole_layers.last()
    post_process_layer(ui, layer)

    # RMA alignment
    ui.sta.construct_landscape(
        layer=layer,
        template_path=TEMPLATE_X,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        upsample_factor=5,
    )
    landscape_layer = ui.parent_viewer.layers[-1]
    ui.sta.run_align_on_landscape(landscape_layer)
    mole_cnv = post_process_layer(ui, ui.mole_layers.last()).molecules
    ui.sta.run_annealing_on_landscape(
        landscape_layer,
        range_long=(4.0, 4.28),
        range_lat=("-0.1", "+0.1"),
        angle_max=5.0,
    )
    mole_rma = post_process_layer(ui, ui.mole_layers.last()).molecules
    return pl.concat(
        [
            flat_agg(mole_cnv.features, "_conv"),
            flat_agg(mole_rma.features, "_RMA"),
        ],
        how="horizontal",
    )


def flat_agg(df: pl.DataFrame, suffix: str):
    agg = (
        df.group_by("isotype-id", maintain_order=True)
        .agg(pl.col("twist").filter(pl.col("twist").is_finite()).mean())
        .sort("isotype-id")
    )
    return pl.DataFrame(
        {f"twist_a{suffix}": agg["twist"][1], f"twist_b{suffix}": agg["twist"][0]}
    )


def show_dataframe(ui: CylindraMainWidget, df: pl.DataFrame):
    view = DataFrameView()
    view.value = df
    ui.parent_viewer.window.add_dock_widget(view)


def main():
    ui = start()
    create_microtubule(ui)
    mole_ans = ui.mole_layers.first().molecules
    ans = flat_agg(mole_ans.features, "_ans")
    df_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tilt_series(ui, tmpdir)
        for i in range(20):
            out = run_one(
                ui,
                Path(tmpdir) / "image.mrc",
                seed=i,
            )
            df_list.append(pl.concat([ans, out], how="horizontal"))
            print("Done: ", i)

    df = pl.concat(df_list, how="vertical")
    show_dataframe(ui, df)
    ui.add_molecules(mole_ans, name="GT", source=ui.splines[0])


if __name__ == "__main__":
    main()
    napari.run()
