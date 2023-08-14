from __future__ import annotations

import argparse
import ast
from typing import NamedTuple
from pathlib import Path
import tempfile
from cylindra import start
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import timer
from cylindra.const import PropertyNames as H

import polars as pl

TEMPLATE_PATH = Path(__file__).parent.parent / "tests" / "beta-tubulin.mrc"


class CftResult(NamedTuple):
    nsr: float
    rep: int
    val0: float
    val1: float
    val2: float
    val3: float


POSITIONS = [(0, 15), (15, 30), (30, 45), (45, 60)]


def local_expansions(ui: CylindraMainWidget):
    ui.cylinder_simulator.update_model(
        spacing=4.12, skew=0.08, start=3, radius=11.4, npf=13
    )
    for exp, yrange in zip([-0.07, -0.02, 0.03, 0.08], POSITIONS):
        ui.cylinder_simulator.expand(
            exp=exp, yrange=yrange, arange=(0, 13), allev=False
        )


def local_skew(ui: CylindraMainWidget):
    ui.cylinder_simulator.update_model(
        spacing=4.1, skew=0.0, start=3, radius=11.4, npf=13
    )
    for sk, yrange in zip([-0.15, -0.05, 0.05, 0.15], POSITIONS):
        ui.cylinder_simulator.screw(skew=sk, yrange=yrange, arange=(0, 13), allev=False)


def simulate(
    func=local_expansions,
    n_tilt: int = 61,
    nsr: list[float] = [0.7072, 0.8840, 1.061],
    nrepeat: int = 5,
    scale: float = 0.5,
    binsize: int = 1,
    output_dir: Path | None = None,
):
    if isinstance(func, str):
        func = globals()[func]
    if output_dir is not None:
        output_dir = Path(output_dir)
        assert output_dir.parent.exists()
        if not output_dir.exists():
            output_dir.mkdir()
    ui = start()
    t0 = timer()
    ui.cylinder_simulator.create_straight_line(
        scale=scale, size=(60.0, 240.0, 60.0), length=245.0
    )
    func(ui)  # simulate cylinder
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
                ui.open_image(
                    tmpdir / fname, tilt_range=(-60, 60), eager=True, bin_size=binsize
                )
                ui.register_path([[30, 30, 30], [30, 210, 30]])
                ui.measure_radius(splines=[0])
                ui.add_anchors(splines=[0], interval=65, how="equal")
                ui.local_ft_analysis(
                    splines=[0], depth=32.64, interval=None, bin_size=binsize
                )
                lprops = ui.tomogram.splines[0].localprops
                if output_dir is not None:
                    lprops.write_csv(output_dir / f"n={_rep}_nsr={_nsr:3f}.csv")
                results.append(CftResult(_nsr, _rep, *lprops[H.spacing]))

        results = pl.DataFrame(results).sort(by="nsr")
        print(results.write_csv(separator="\t", float_precision=3))
        agg_df = results.groupby("nsr").agg(
            [
                pl.format(
                    "{} +/- {}", pl.col(x).mean().round(3), pl.col(x).std().round(3)
                ).alias(x)
                for x in ["val0", "val1", "val2", "val3"]
            ]
        )
        print(agg_df)
    t0.toc(log=False)


class Namespace(argparse.Namespace):
    func: str
    n_tilt: int
    nsr: list[float]
    nrepeat: int
    scale: float
    binsize: int
    output_dir: str | None


class Args(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--func", type=str, default="local_expansions")
        self.add_argument("--n_tilt", type=int, default=61)
        self.add_argument("--nsr", type=str, default="[0.7072, 0.8840, 1.061]")
        self.add_argument("--nrepeat", type=int, default=5)
        self.add_argument("--scale", type=float, default=0.5)
        self.add_argument("--binsize", type=int, default=1)
        self.add_argument("--output_dir", type=str, default=None)

    @classmethod
    def from_args(cls) -> Namespace:
        return cls().parse_args()


if __name__ == "__main__":
    args = Args.from_args()

    simulate(
        func=args.func,
        n_tilt=args.n_tilt,
        nsr=ast.literal_eval(args.nsr),
        nrepeat=args.nrepeat,
        scale=args.scale,
        binsize=args.binsize,
        output_dir=args.output_dir,
    )
