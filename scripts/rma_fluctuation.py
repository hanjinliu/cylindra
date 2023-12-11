import tempfile
from cylindra import start  # NOTE: Set ApplicationAttributes
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
from cylindra.widgets import CylindraMainWidget
from cylindra.const import MoleculesHeader as Mole
from cylindra.types import MoleculesLayer
from cylindra.cylmeasure import RegionProfiler

import polars as pl
from scripts.user_consts import TEMPLATE_X

SPACING = Mole.spacing
SPACING_MEAN = f"{Mole.spacing}_mean"


def create_microtubule(ui: CylindraMainWidget, spacing: float = 4.08):
    ui.simulator.create_empty_image(size=(60.0, 180.0, 60.0), scale=0.2625)
    prep_molecules(ui, spacing)
    layer = ui.mole_layers.last()
    ui.calculate_lattice_structure(layer=layer, props=["spacing"])
    return layer.molecules


def prep_molecules(ui: CylindraMainWidget, spacing: float = 4.08):
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 165.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=spacing, twist=0.04, start=3, npf=13, radius=11.0, offsets=(0.0, 0.0)
    )
    layer = ui.mole_layers.last()
    return layer.molecules


def simulate_and_save_tilt_series(ui: CylindraMainWidget, path: Path):
    path = Path(path)
    layer_name = ui.mole_layers.last().name
    ui.simulator.simulate_tilt_series(
        components=[(layer_name, TEMPLATE_X)],
        save_dir=path,
        tilt_range=(-60, 60),
        n_tilt=61,
    )


def run_one(ui: CylindraMainWidget, image_path: Path, seed: int = 0):
    ui.simulator.simulate_tomogram_from_tilt_series(
        image_path,
        nsr=3.5,
        bin_size=2,
        tilt_range=(-60, 60),
        height=60.0,
        seed=seed,
    )
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 165.0, 30.0))
    ui.global_cft_analysis(splines=[0], bin_size=2)
    spacing_est = ui.splines[0].props.get_glob("spacing")
    ui.simulator.generate_molecules(
        spacing=spacing_est,
        twist=0.04,
        start=3,
        npf=13,
        radius=11.0,
        offsets=(0.0, 0.0),
    )

    layer = ui.mole_layers.last()
    ui.sta.align_all_annealing(
        layer=layer,
        template_path=TEMPLATE_X,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        range_long=(3.98, 4.28),
        range_lat=("-0.1", "+0.1"),
        temperature_time_const=0.8,
    )
    aligned = ui.mole_layers.last()
    ui.calculate_lattice_structure(layer=aligned, props=["spacing"])
    ui.convolve_feature(
        layer=aligned,
        target=SPACING,
        method="mean",
        footprint=[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
    )
    ui.binarize_feature(layer=aligned, target=SPACING_MEAN, threshold=4.15)
    ui.label_feature_clusters(layer=aligned, target=f"{SPACING_MEAN}_binarize")

    return run_regionprops(aligned, smallest=3.0)


def run_regionprops(layer: MoleculesLayer, smallest: int = 3):
    mole = layer.molecules
    spl = layer.source_spline
    nmole = mole.count()
    prof = RegionProfiler.from_components(
        mole, spl, target=SPACING_MEAN, label=f"{SPACING_MEAN}_binarize_label"
    )
    out = prof.calculate(["area", "length", "width"]).filter(pl.col("area") >= smallest)
    occ = out["area"].sum() / nmole
    freq = len(out) / spl.length()
    return occ, freq


def show_dataframe(ui: CylindraMainWidget, df: pl.DataFrame):
    view = DataFrameView()
    view.value = df
    ui.parent_viewer.window.add_dock_widget(view)


def main():
    ui = start()
    results = []
    for spacing in [4.08, 4.1, 4.12]:
        create_microtubule(ui, spacing=spacing)
        with tempfile.TemporaryDirectory() as tmpdir:
            simulate_and_save_tilt_series(ui, tmpdir)
            for i in range(10):
                out = run_one(
                    ui,
                    Path(tmpdir) / "image.mrc",
                    seed=i,
                )
                results.append([spacing, *out])
                print(
                    f"Done: {spacing:.3f}, ({i})",
                )
    df = pl.DataFrame(np.array(results), schema=["spacing", "occ", "freq"])
    show_dataframe(ui, df)
    ui.parent_viewer.show(block=True)


if __name__ == "__main__":
    main()