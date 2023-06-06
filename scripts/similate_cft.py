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
    low0: float
    high: float
    low1: float


def simulate_local_expansion(
    n_tilt: int = 21,
    nsr: list[float] = [0.2, 1.2],
    nrepeat: int = 5,
    exp=0.12,
    output_dir: Path | None = None,
):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    ui = start()
    print(f"n_tilt={n_tilt}")
    print(f"spacing: low=4.080, high={4.08 + exp:.3f} ({exp / 4.08 * 100:+.1f}%)")
    ui.cylinder_simulator.create_straight_line(
        scale=0.5, size=(60, 180, 60), length=160
    )
    ui.cylinder_simulator.update_model(
        spacing=4.08, skew=0.08, rise=9.6, radius=10.4, npf=13
    )
    ui.cylinder_simulator.expand(exp=exp, yrange=(15, 25), arange=(0, 13), allev=False)
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
                ui.register_path([[30, 25, 30], [30, 155, 30]])
                ui.refine_splines(splines=[0])
                ui.measure_radius(splines=[0])
                ui.local_ft_analysis(splines=[0], depth=32.64, interval=7.7)
                lprops = ui.get_spline(0).localprops  # shape (17, x)
                if output_dir is not None:
                    lprops.select([H.spacing, H.skew]).write_csv(
                        output_dir / f"n={_rep}_nsr={_nsr:3f}.csv"
                    )
                sp = lprops[H.spacing]
                results.append(CftResult(_nsr, _rep, sp[0], sp[8], sp[14]))

        results = pl.DataFrame(results).sort(by="nsr")
        print(results.write_csv(separator="\t", float_precision=3))


if __name__ == "__main__":
    simulate_local_expansion()
