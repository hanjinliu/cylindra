import tempfile
from cylindra import start  # NOTE: Set ApplicationAttributes
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cylindra.widgets import CylindraMainWidget
from cylindra.const import MoleculesHeader as Mole

import polars as pl
from scripts.user_consts import TEMPLATE_A, TEMPLATE_B

SPACING = Mole.spacing
SPACING_MEAN = f"{Mole.spacing}_mean"


def save_tilt_series(ui: CylindraMainWidget, path: Path):
    path = Path(path)
    ui.simulator.create_empty_image(size=(60.0, 240.0, 60.0), scale=0.2615)
    ui.simulator.create_straight_line((30.0, 0.0, 30.0), (30.0, 240.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.04, start=3, npf=13, radius=10.5, offsets=(0.0, 0.4)
    )
    ui.sta.seam_search_manually(ui.mole_layers.last())
    ui.split_molecules(ui.mole_layers.last(), by="isotype-id")
    layer0 = ui.mole_layers.nth(0).name
    layer1 = ui.mole_layers.nth(1).name
    ui.simulator.simulate_tilt_series(
        components=[(layer0, TEMPLATE_A), (layer1, TEMPLATE_B)],
        save_dir=path,
        tilt_range=(-60, 60),
        n_tilt=61,
    )


def run_one(
    ui: CylindraMainWidget,
    image_path: Path,
    nsr: float,
    seed: int = 0,
) -> pl.DataFrame:
    ui.simulator.simulate_tomogram_from_tilt_series(
        image_path,
        nsr=nsr,
        bin_size=2,
        tilt_range=(-60, 60),
        height=60.0,
        seed=seed,
    )
    ui.simulator.create_straight_line((30.0, 20.0, 30.0), (30.0, 220.0, 30.0))
    ui.fit_splines(bin_size=2, edge_sigma=2.0, max_shift=5.0, err_max=1.0)
    ui.refine_splines(max_interval=24.0, bin_size=2, err_max=0.8)
    ui.measure_radius(bin_size=2, min_radius=8.0)
    ui.global_cft_analysis(bin_size=2)
    ui.infer_polarity(bin_size=2)
    assert (npf := ui.splines[0].props.get_glob("npf")) == 13, npf
    assert (st := ui.splines[0].props.get_glob("start")) == 3, st
    assert (ori := ui.splines[0].props.get_glob("orientation")) == "MinusToPlus", ori
    out = ui.run_workflow("07-estimate_nsr", i=0)
    return out


def show_dataframe(ui: CylindraMainWidget, df: pl.DataFrame):
    view = DataFrameView()
    view.value = df
    ui.parent_viewer.window.add_dock_widget(view)


def main():
    ui = start()
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tilt_series(ui, tmpdir)
        for nsr in [3.0, 3.5, 4.0]:
            for i in range(10):
                nsr3d = run_one(
                    ui,
                    Path(tmpdir) / "image.mrc",
                    nsr=nsr,
                    seed=i + int(nsr * 1000),
                )
                results.append((nsr, nsr3d))

                print(f"Done: {nsr:.2f}, {nsr3d:.2f}, ({i})")
    df = pl.DataFrame(np.array(results), schema=["nsr2d", "nsr3d"])
    show_dataframe(ui, df)
    plt.figure(figsize=(4, 3))
    sns.swarmplot(data=df, x="nsr2d", y="nsr3d")
    plt.show()


if __name__ == "__main__":
    main()
    import napari

    napari.run()
