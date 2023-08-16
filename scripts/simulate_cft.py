from __future__ import annotations

import argparse
import ast
from typing import NamedTuple
from pathlib import Path
import tempfile

import numpy as np
from cylindra import start
from cylindra.components import CylSpline
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


class Simulator:
    def __init__(self, ui: CylindraMainWidget, scale: float):
        self.ui = ui
        self.scale = scale

    def prepare(self) -> np.ndarray:
        raise NotImplementedError

    def results(self):
        raise NotImplementedError

    def columns(self) -> list[str]:
        return ["val0", "val1", "val2", "val3"]


class local_expansions(Simulator):
    """Vertical MT with spacing=4.05, 4.10, 4.15, 4.20 nm."""

    def prepare(self):
        self.ui.cylinder_simulator.create_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.cylinder_simulator.update_model(
            spacing=4.12, skew=0.08, start=3, radius=11.4, npf=13
        )
        for exp, yrange in zip([-0.07, -0.02, 0.03, 0.08], POSITIONS):
            self.ui.cylinder_simulator.expand(
                exp=exp, yrange=yrange, arange=(0, 13), allev=False
            )
        return np.array([[30, 30, 30], [30, 210, 30]])

    def results(self):
        return self.ui.tomogram.splines[0].props.loc[H.spacing]


class local_skew(Simulator):
    """Vertical MT with skew=-0.15, -0.05, 0.05, 0.15 deg."""

    def prepare(self):
        self.ui.cylinder_simulator.create_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.cylinder_simulator.update_model(
            spacing=4.1, skew=0.0, start=3, radius=11.4, npf=13
        )
        for sk, yrange in zip([-0.15, -0.05, 0.05, 0.15], POSITIONS):
            self.ui.cylinder_simulator.screw(
                skew=sk, yrange=yrange, arange=(0, 13), allev=False
            )
        return np.array([[30, 30, 30], [30, 210, 30]])

    def results(self):
        return self.ui.tomogram.splines[0].props.loc[H.skew]


class local_orientation(Simulator):
    """Cureved MT with orientation=60, 40, 20, 0 deg."""

    def prepare(self):
        length = 52.0
        curve_length = 16.0

        def get_vec(l: float, deg: float, dup: int = 1) -> np.ndarray:
            rad = np.deg2rad(deg)
            l0 = l / dup
            return np.array([[0, l0 * np.cos(rad), l0 * np.sin(rad)]] * dup)

        dup0 = 12
        vecs = np.concatenate(
            [
                np.array([[30, 20, 0]]),
                get_vec(length, 60.0, dup=dup0),
                get_vec(curve_length, 50.0, dup=3),
                get_vec(length, 40.0, dup=dup0),
                get_vec(curve_length, 30.0, dup=3),
                get_vec(length, 20.0, dup=dup0),
                get_vec(curve_length, 10.0, dup=3),
                get_vec(length, 0.0, dup=dup0),
            ],
            axis=0,
        )

        coords = np.cumsum(vecs, axis=0)
        spl = CylSpline().fit(coords, err_max=1e-8)
        self.ui.cylinder_simulator.create_empty_image(
            size=(60, 228, 144), scale=self.scale
        )
        self.ui.cylinder_simulator.set_spline(spl)
        self.ui.cylinder_simulator.update_model(
            spacing=4.1, skew=0.0, start=3, radius=11.4, npf=13
        )
        return coords[dup0 // 2 : -dup0 // 2]

    def results(self):
        df = self.ui.tomogram.splines[0].props.loc
        return df[H.spacing].to_list() + df[H.skew].to_list() + df[H.rise].to_list()

    def columns(self):
        return [f"{n}{i}" for n in ["spacing", "skew", "rise"] for i in range(4)]


def simulate(
    func=local_expansions,
    n_tilt: int = 61,
    nsr: list[float] = [0.7072, 0.8840, 1.061],
    nrepeat: int = 5,
    scale: float = 0.5,
    binsize: int = 1,
    output: Path | None = None,
    seed: int = 41298764,
):
    if isinstance(func, str):
        func = globals()[func]
    if output is not None:
        output = Path(output)
        assert output.parent.exists()

    ui = start()
    t0 = timer(name=func.__name__)
    simulator = func(ui, scale)  # simulate cylinder
    coords = simulator.prepare()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ui.cylinder_simulator.simulate_tomogram(
            template_path=TEMPLATE_PATH,
            save_dir=tmpdir,
            nsr=list(nsr) * nrepeat,
            n_tilt=n_tilt,
            scale=scale,
            seed=seed,
        )
        results = list[CftResult]()
        for _rep in range(nrepeat):
            for _idx, _nsr in enumerate(nsr):
                fname = f"image-{len(nsr) * _rep + _idx}.mrc"
                ui.open_image(
                    tmpdir / fname, tilt_range=(-60, 60), eager=True, bin_size=binsize
                )
                ui.register_path(coords)
                ui.measure_radius(splines=[0])
                ui.tomogram.splines[0].anchors = np.linspace(0, 1, 4)
                ui.local_ft_analysis(
                    splines=[0], depth=32.64, interval=None, bin_size=binsize
                )
                results.append([_nsr, _rep, *simulator.results()])

        columns = ["nsr", "rep"] + simulator.columns()
        results = pl.DataFrame(results, schema=columns).sort(by="nsr")
        if output is None:
            print(results.write_csv(separator="\t", float_precision=3))
        else:
            results.write_csv(output)
        agg_df = results.groupby("nsr").agg(
            [
                pl.format(
                    "{}±{}", pl.col(x).mean().round(3), pl.col(x).std().round(3)
                ).alias(x)
                for x in results.columns[2:]
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
    output: str | None
    seed: int


class Args(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--func", type=str, default="local_expansions")
        self.add_argument("--n_tilt", type=int, default=61)
        self.add_argument("--nsr", type=str, default="[0.7072, 0.8840, 1.061]")
        self.add_argument("--nrepeat", type=int, default=5)
        self.add_argument("--scale", type=float, default=0.5)
        self.add_argument("--binsize", type=int, default=1)
        self.add_argument("--output", type=str, default=None)
        self.add_argument("--seed", type=int, default=41298764)

    @classmethod
    def from_args(cls) -> Namespace:
        return cls().parse_args()


if __name__ == "__main__":
    args = Args.from_args()
    nsr = ast.literal_eval(args.nsr)
    if isinstance(nsr, (int, float)):
        nsr = [nsr]
    simulate(
        func=args.func,
        n_tilt=args.n_tilt,
        nsr=nsr,
        nrepeat=args.nrepeat,
        scale=args.scale,
        binsize=args.binsize,
        output=args.output,
        seed=args.seed,
    )
