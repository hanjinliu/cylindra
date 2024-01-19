import napari
import numpy as np
import polars as pl

from cylindra import start
from cylindra.const import MoleculesHeader as Mole
from cylindra.widgets import CylindraMainWidget
from scripts.user_consts import WOBBLE_TEMPLATES


def rmsd(x: pl.Series, y: pl.Series) -> float:
    d = x - y
    d = d.filter(d.is_finite())
    return np.sqrt((d**2).mean())


SPACING = Mole.spacing
SPACING_MEAN = f"{Mole.spacing}_mean"


def main(ui: CylindraMainWidget):
    ui.open_image(
        path=r"D:\simulated_tomograms\230925_two_patches\image-2.mrc",
        scale=0.2625,
        tilt_range=(-60.0, 60.0),
        bin_size=[2, 4],
        filter="Lowpass",
        eager=False,
    )
    ui.load_molecules(
        paths=(r"D:\simulated_tomograms\230925_two_patches\molecules.csv",)
    )
    ui.load_splines(paths=["D:/simulated_tomograms/230925_two_patches/spline.json"])
    ui.set_source_spline(layer="molecules", spline=0)
    ui.sta.align_all(
        layers=["molecules"],
        template_paths=WOBBLE_TEMPLATES,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        rotations=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff=0.5,
        interpolation=3,
        method="zncc",
        bin_size=1,
    )
    ui.measure_radius(splines=[0], bin_size=2, min_radius=1.6)
    ui.global_cft_analysis(splines=[0], bin_size=2)
    ui.mole_layers["molecules-ALN1"].name = "Conventional"
    ui.sta.align_all_annealing(
        layer="molecules",
        template_path=WOBBLE_TEMPLATES,
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        rotations=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff=0.5,
        interpolation=3,
        range_long=(4.0, 4.28),
        range_lat=(5.6, 5.8),
        angle_max=5.0,
        upsample_factor=5,
        random_seeds=[0, 1, 2, 3, 4],
    )
    ui.calculate_lattice_structure(layer="molecules", props=["spacing"])
    ui.calculate_lattice_structure(layer="Conventional", props=["spacing"])
    ui.calculate_lattice_structure(layer="molecules-ALN1", props=["spacing"])
    ui.paint_molecules(
        layer="molecules",
        color_by=SPACING,
        limits=(4.0, 4.28),
    )
    ui.paint_molecules(
        layer="Conventional",
        color_by=SPACING,
        limits=(4.0, 4.28),
    )
    ui.paint_molecules(
        layer="molecules-ALN1",
        color_by=SPACING,
        limits=(4.0, 4.28),
    )
    ui.set_multiscale(bin_size=2)
    ui.convolve_feature(
        layer="molecules-ALN1",
        target=SPACING,
        method="mean",
        footprint=[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
    )
    ui.convolve_feature(
        layer="Conventional",
        target=SPACING,
        method="mean",
        footprint=[[1, 1, 1], [1, 1, 1], [0, 0, 0]],
    )

    ans = ui.mole_layers["molecules"].molecules.features[SPACING]
    conv = ui.mole_layers["Conventional"].molecules.features[SPACING]
    conv_filt = ui.mole_layers["Conventional"].molecules.features[SPACING_MEAN]
    rma = ui.mole_layers["molecules-ALN1"].molecules.features[SPACING]
    rma_filt = ui.mole_layers["molecules-ALN1"].molecules.features[SPACING_MEAN]

    for data in [conv, conv_filt, rma, rma_filt]:
        print(rmsd(data, ans))


if __name__ == "__main__":
    main(start())
    napari.run()
