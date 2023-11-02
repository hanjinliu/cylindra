import sys
from pathlib import Path
import timeit
import numpy as np
from cylindra import start
from scripts.user_consts import TEMPLATE_X

IMAGE_PATH = Path(__file__).parent.parent / "tests" / "13pf_MT.tif"


def main(method="viterbi"):
    ui = start()
    ui.open_image(IMAGE_PATH, tilt_range=(-60, 60))
    ui.register_path([[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    ui.fit_splines(
        splines=[0],
        max_interval=30,
        bin_size=1,
        err_max=1.0,
        degree_precision=0.5,
        edge_sigma=2.0,
        max_shift=5.0,
    )
    ui.refine_splines(
        splines=[0], max_interval=50.0, err_max=0.8, corr_allowed=0.9, bin_size=1
    )
    ui.measure_radius(splines=[0], bin_size=1, min_radius=1.0)
    ui.infer_polarity(splines=[0], depth=40, bin_size=1)
    ui.global_cft_analysis(splines=[0], bin_size=1)
    ui.map_monomers(splines=[0], orientation="MinusToPlus")
    ui.sta.align_averaged(layers=["Mole-0"], template_path=TEMPLATE_X, mask_params=None)
    ui.sta.construct_landscape(
        layer="Mole-0-ALN1",
        template_path=TEMPLATE_X,
        mask_params=(0.3, 0.8),
        max_shifts=(2.2, 2.2, 2.2),
        upsample_factor=3,
    )

    _times = []
    for _ in range(10):
        t0 = timeit.default_timer()
        match method:
            case "viterbi":
                ui.sta.run_viterbi_on_landscape(
                    landscape_layer="Landscape_Mole-0-ALN1",
                    range_long=(4.0, 4.28),
                    angle_max=5.0,
                )
            case "rma":
                ui.sta.run_annealing_on_landscape(
                    "Landscape_Mole-0-ALN1",
                    range_lat=("-0.1", "+0.1"),
                    random_seeds=[0, 1, 2, 3],
                )
        _times.append(timeit.default_timer() - t0)

    print(
        "landscape shape =",
        ui.parent_viewer.layers["Landscape_Mole-0-ALN1"].landscape.energies.shape,
    )
    print(f"Time: {np.mean(_times):.3f} ± {np.std(_times):.3f} sec")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
