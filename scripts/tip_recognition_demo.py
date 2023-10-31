from __future__ import annotations

from pathlib import Path
from cylindra import start

import napari
from scripts.user_consts import TEMPLATE_B


def main():
    ui = start()
    ui.simulator.create_image_with_straight_line(
        scale=0.25, size=(60.0, 240.0, 60.0), length=210.0
    )
    ui.simulator.generate_molecules(
        0, spacing=4.08, twist=0.04, start=3, radius=11.2, npf=13
    )
    ui.simulator.expand(0.12, (0, 18), (0, 13), allev=False)
    ui.simulator.simulate_tomogram_and_open(
        template_path=TEMPLATE_B,
        nsr=2.5,
        bin_size=2,
        n_tilt=61,
    )
    ui.clip_spline(0, (24.5, 24.5))
    ui.measure_radius([0], bin_size=2)
    ui.local_cft_analysis([0], depth=49, bin_size=2, interval=8.16)


if __name__ == "__main__":
    main()
    napari.run()
