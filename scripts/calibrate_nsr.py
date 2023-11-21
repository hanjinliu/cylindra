import tempfile
from cylindra import start  # NOTE: Set ApplicationAttributes
from magicclass.types import Path
from magicclass.ext.polars import DataFrameView
import numpy as np
from cylindra.widgets import CylindraMainWidget
from cylindra.const import MoleculesHeader as Mole
from cylindra.types import MoleculesLayer
import napari

import polars as pl
from scripts.user_consts import TEMPLATE_X

SPACING = Mole.spacing
SPACING_MEAN = f"{Mole.spacing}_mean"


def create_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 240.0, 60.0), scale=0.25)
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 225.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.04, start=3, npf=13, radius=10.5, offsets=(0.0, 0.0)
    )


def map_monomers(ui: CylindraMainWidget):
    ui.simulator.create_straight_line((30.0, 15.0, 30.0), (30.0, 225.0, 30.0))
    ui.simulator.generate_molecules(
        spacing=4.08, twist=0.04, start=3, npf=13, radius=10.5, offsets=(0.0, 0.0)
    )


def save_tilt_series(ui: CylindraMainWidget, path: Path):
    path = Path(path)
    layer_name = ui.mole_layers.first().name
    ui.simulator.simulate_tilt_series(
        components=[(layer_name, TEMPLATE_X)],
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
    map_monomers(ui)

    layer = ui.mole_layers.last()

    # RMA alignment
    ui.sta.align_all(
        layers=layer,
        template_path=TEMPLATE_X,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        rotations=[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    )

    dsk = ui.sta.get_subtomograms(ui.mole_layers.last(), shape=(5.4, 5.4, 5.4), order=3)
    dsk_mean = dsk.mean(axis=0)
    bg = dsk - dsk_mean[np.newaxis]
    return bg.std() / dsk_mean.max()


def show_dataframe(ui: CylindraMainWidget, df: pl.DataFrame):
    view = DataFrameView()
    view.value = df
    ui.parent_viewer.window.add_dock_widget(view)


def main():
    ui = start()
    create_microtubule(ui)
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        save_tilt_series(ui, tmpdir)
        for nsr in [2.0, 2.5, 3.0]:
            for i in range(10):
                nsr3d = run_one(
                    ui,
                    Path(tmpdir) / "image.mrc",
                    nsr=nsr,
                    seed=i,
                )
                results.append((nsr, nsr3d))

                print(f"Done: {nsr:.2f}, ({i})")
    df = pl.DataFrame(np.array(results), schema=["nsr2d", "nsr3d"])
    show_dataframe(ui, df)


if __name__ == "__main__":
    main()
    napari.run()
