from __future__ import annotations

from typing import NamedTuple
from pathlib import Path
import tempfile
from cylindra import start
from cylindra.const import PropertyNames as H

import polars as pl

TEMPLATE_PATH = Path(__file__).parent.parent / "tests" / "beta-tubulin.mrc"


class CftResult(NamedTuple):
    nsr: float
    rep: int
    values: list[float]


def simulate_local_expansion(
    n_tilt: int = 21,
    nsr: list[float] = [0.2, 1.2],
    nrepeat: int = 5,
    output_dir: Path | None = None,
):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    ui = start()
    ui.cylinder_simulator.create_straight_line(
        scale=0.5, size=(60, 200, 60), length=200
    )
    ui.cylinder_simulator.update_model(
        spacing=4.12, skew=0.08, start=3, radius=11.4, npf=13
    )
    ui.cylinder_simulator.expand(exp=-0.07, yrange=(0, 15), arange=(0, 13), allev=False)
    ui.cylinder_simulator.expand(
        exp=-0.02, yrange=(15, 30), arange=(0, 13), allev=False
    )
    ui.cylinder_simulator.expand(exp=0.03, yrange=(30, 45), arange=(0, 13), allev=False)
    ui.cylinder_simulator.expand(exp=0.08, yrange=(45, 60), arange=(0, 13), allev=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ui.cylinder_simulator.simulate_tomogram(
            template_path=TEMPLATE_PATH,
            save_dir=tmpdir,
            nsr=list(nsr) * nrepeat,
            n_tilt=n_tilt,
            seed=41298764,
        )
        results = list[CftResult]()
        for _rep in range(nrepeat):
            for _idx, _nsr in enumerate(nsr):
                fname = f"image-{len(nsr) * _rep + _idx}.mrc"
                ui.open_image(tmpdir / fname, tilt_range=(-60, 60), eager=True)
                ui.register_path([[30, 30, 30], [30, 210, 30]])
                ui.measure_radius(splines=[0])
                ui.add_anchors(splines=[0], interval=65, how="equal")
                ui.local_ft_analysis(splines=[0], depth=32.64, interval=None)
                lprops = ui.tomogram.splines[0].localprops
                if output_dir is not None:
                    lprops.write_csv(output_dir / f"n={_rep}_nsr={_nsr:3f}.csv")
                results.append(CftResult(_nsr, _rep, lprops[H.spacing].to_list()))

        results = pl.DataFrame(results).sort(by="nsr")
        print(results.write_csv(separator="\t", float_precision=3))


if __name__ == "__main__":
    simulate_local_expansion()
