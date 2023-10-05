from cylindra import start
from cylindra.widgets import CylindraMainWidget
import napari
import polars as pl
import numpy as np


def rmsd(x: pl.Series, y: pl.Series) -> float:
    d = x - y
    d = d.filter(d.is_finite())
    return np.sqrt((d**2).mean())


def main(ui: CylindraMainWidget):
    ui.open_image(
        path=r"D:\simulated_tomograms\230925_two_patches\image-2.mrc",
        scale=0.25,
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
    ui.sta.align_all_multi_template(
        layers=["molecules"],
        template_paths=[
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_05.mrc",
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_10.mrc",
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_15.mrc",
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_20.mrc",
        ],
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        rotations=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff=0.5,
        interpolation=3,
        method="zncc",
        bin_size=1,
    )
    ui.measure_radius(splines=[0], bin_size=2, min_radius=1.6)
    ui.global_ft_analysis(splines=[0], bin_size=2)
    ui.parent_viewer.layers["molecules-ALN1"].name = "Conventional"
    ui.sta.align_all_annealing_multi_template(
        layer="molecules",
        template_paths=[
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_05.mrc",
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_10.mrc",
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_15.mrc",
            r"E:\EMDB\tubulin_spaced\x-tubulin-comp-4_20.mrc",
        ],
        mask_params=(0.3, 0.8),
        max_shifts=(0.8, 0.8, 0.8),
        rotations=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff=0.5,
        interpolation=3,
        distance_range_long=(3.98, 4.28),
        distance_range_lat=(5.6, 5.8),
        angle_max=5.0,
        upsample_factor=5,
        random_seeds=[0, 1, 2, 3, 4],
        return_all=False,
    )
    ui.calculate_lattice_structure(layer="molecules", props=["interv"])
    ui.calculate_lattice_structure(layer="Conventional", props=["interv"])
    ui.calculate_lattice_structure(layer="molecules-ALN1", props=["interv"])
    ui.paint_molecules(
        layer="molecules",
        color_by="interval-nm",
        cmap={
            0.0: (0.043, 0.0, 0.0, 1.0),
            0.3: (0.529, 0.176, 0.616, 1.0),
            0.68: (1.0, 0.0, 0.0, 1.0),
            1.0: (1.0, 1.0, 0.0, 1.0),
        },
        limits=(3.98, 4.28),
    )
    ui.paint_molecules(
        layer="Conventional",
        color_by="interval-nm",
        cmap={
            0.0: (0.043, 0.0, 0.0, 1.0),
            0.3: (0.529, 0.176, 0.616, 1.0),
            0.68: (1.0, 0.0, 0.0, 1.0),
            1.0: (1.0, 1.0, 0.0, 1.0),
        },
        limits=(3.98, 4.28),
    )
    ui.paint_molecules(
        layer="molecules-ALN1",
        color_by="interval-nm",
        cmap={
            0.0: (0.043, 0.0, 0.0, 1.0),
            0.3: (0.529, 0.176, 0.616, 1.0),
            0.68: (1.0, 0.0, 0.0, 1.0),
            1.0: (1.0, 1.0, 0.0, 1.0),
        },
        limits=(3.98, 4.28),
    )
    ui.set_multiscale(bin_size=2)
    ui.convolve_feature(
        layer="molecules-ALN1",
        target="interval-nm",
        method="mean",
        footprint=[[1, 1, 1], [1, 1, 1]],
    )
    ui.convolve_feature(
        layer="Conventional",
        target="interval-nm",
        method="mean",
        footprint=[[1, 1, 1], [1, 1, 1]],
    )

    ans = ui.mole_layers["molecules"].molecules.features["interval-nm"]
    conv = ui.mole_layers["Conventional"].molecules.features["interval-nm"]
    conv_filt = ui.mole_layers["Conventional"].molecules.features["interval-nm_mean"]
    rma = ui.mole_layers["molecules-ALN1"].molecules.features["interval-nm"]
    rma_filt = ui.mole_layers["molecules-ALN1"].molecules.features["interval-nm_mean"]

    for data in [conv, conv_filt, rma, rma_filt]:
        print(rmsd(data, ans))


if __name__ == "__main__":
    main(start())
    napari.run()
