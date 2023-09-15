from enum import Enum
from typing import Annotated
from pathlib import Path
import tempfile
from tqdm import tqdm, trange

from cylindra import start
from magicgui import magicgui
from magicgui.widgets import PushButton
from magicclass.types import Optional
from magicclass.ext.polars import DataFrameView
import numpy as np
from numpy.typing import NDArray
from cylindra.components import CylSpline
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import timer
from cylindra.const import PropertyNames as H

import polars as pl

TEMPLATE_PATH = Path(__file__).parent.parent / "tests" / "beta-tubulin.mrc"


POSITIONS = [(0, 15), (15, 30), (30, 45), (45, 60)]


class Simulator:
    def __init__(self, ui: CylindraMainWidget, scale: float):
        self.ui = ui
        self.scale = scale

    def prepare(self) -> NDArray[np.float32]:
        raise NotImplementedError

    def results(self):
        raise NotImplementedError

    def anchors(self) -> NDArray[np.float32]:
        return np.linspace(0, 1, 4)

    def columns(self) -> list[str]:
        return ["val0", "val1", "val2", "val3"]


class local_expansion(Simulator):
    """Vertical MT with spacing=4.05, 4.10, 4.15, 4.20 nm."""

    def prepare(self):
        self.ui.cylinder_simulator.create_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.cylinder_simulator.update_model(
            spacing=4.12, dimer_twist=0.08, start=3, radius=11.2, npf=13
        )
        for exp, yrange in zip([-0.07, -0.02, 0.03, 0.08], POSITIONS):
            self.ui.cylinder_simulator.expand(
                exp=exp, yrange=yrange, arange=(0, 13), allev=False
            )
        return np.array([[30, 30, 30], [30, 210, 30]])

    def results(self):
        return self.ui.tomogram.splines[0].props.loc[H.spacing]

    def columns(self) -> list[str]:
        return [f"spacing{i}" for i in range(4)]


class local_skew(Simulator):
    """Vertical MT with dimer_twist=-0.15, -0.05, 0.05, 0.15 deg."""

    def prepare(self):
        self.ui.cylinder_simulator.create_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.cylinder_simulator.update_model(
            spacing=4.1, dimer_twist=0.0, start=3, radius=11.2, npf=13
        )
        for sk, yrange in zip([-0.15, -0.05, 0.05, 0.15], POSITIONS):
            self.ui.cylinder_simulator.twist(
                dimer_twist=sk, yrange=yrange, arange=(0, 13), allev=False
            )
        return np.array([[30, 30, 30], [30, 210, 30]])

    def results(self):
        return self.ui.tomogram.splines[0].props.loc[H.dimer_twist]

    def columns(self) -> list[str]:
        return [f"dimer_twist{i}" for i in range(4)]


class local_orientation(Simulator):
    """Cureved MT with orientation=60, 40, 20, 0 deg."""

    def get_coords(self):
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

        return np.cumsum(vecs, axis=0)

    def prepare(self):
        coords = self.get_coords()
        spl = CylSpline().fit(coords, err_max=1e-8)
        self.ui.cylinder_simulator.create_empty_image(
            size=(60, 228, 144), scale=self.scale
        )
        self.ui.cylinder_simulator.set_spline(spl)
        self.ui.cylinder_simulator.update_model(
            spacing=4.1, dimer_twist=0.0, start=3, radius=11.2, npf=13
        )
        return coords

    def results(self):
        df = self.ui.tomogram.splines[0].props.loc
        return (
            df[H.spacing].to_list() + df[H.dimer_twist].to_list() + df[H.rise].to_list()
        )

    def anchors(self) -> np.ndarray:
        coords = self.get_coords()
        spl = CylSpline().fit(coords, err_max=1e-8)
        clip = 30 / spl.length()
        return np.linspace(clip, 1 - clip, 4)

    def columns(self):
        return [f"{n}{i}" for n in ["spacing", "dimer_twist", "rise"] for i in range(4)]


class local_curvature(Simulator):
    N_ANCHORS = 24

    def get_coords(self):
        # curvature is 5e-4 ~ 3e-3 = 3% expansion of the outer PF
        return np.array(
            [
                [30, 10, 16],
                [30, 30, 16.1],
                [30, 50, 16.3],
                [30, 90, 17.2],
                [30, 122, 19.6],
                [30, 225, 47],
            ],
            dtype=np.float32,
        )

    def prepare(self) -> NDArray[np.float32]:
        coords = self.get_coords()
        spl = CylSpline().fit(coords, err_max=1e-8)
        self.ui.cylinder_simulator.create_empty_image(
            size=(60, 230, 60), scale=self.scale
        )
        self.ui.cylinder_simulator.set_spline(spl)
        self.ui.cylinder_simulator.update_model(
            spacing=4.1, dimer_twist=0.00, start=3, radius=11.2, npf=13
        )
        return coords

    def results(self):
        spl = self.ui.tomogram.splines[0]
        df = spl.props.loc
        # average curvatures
        length = spl.length()
        curvatures: list[float] = []
        for anc in self.anchors():
            u = np.linspace(anc - 24.5 / length, anc - 24.5 / length, 100)
            curvatures.append(spl.curvature(u).mean())
        return (
            df[H.spacing].to_list()
            + df[H.dimer_twist].to_list()
            + df[H.rise].to_list()
            + curvatures
        )

    def anchors(self) -> NDArray[np.float32]:
        coords = self.get_coords()
        spl = CylSpline().fit(coords)
        clip = 30 / spl.length()
        return np.linspace(clip, 1 - clip, self.N_ANCHORS)

    def columns(self):
        return [
            f"{n}{i}"
            for n in ["spacing", "dimer_twist", "rise", "curvature"]
            for i in range(self.N_ANCHORS)
        ]


class Funcs(Enum):
    local_expansion = local_expansion
    local_skew = local_skew
    local_orientation = local_orientation
    local_curvature = local_curvature


@magicgui
def simulate(
    function: Funcs = Funcs.local_expansion,
    n_tilt: Annotated[int, {"min": 1}] = 61,
    nsr: list[float] = [0.1, 2.5],
    nrepeat: int = 5,
    scale: float = 0.5,
    binsize: Annotated[int, {"min": 1}] = 1,
    output: Optional[Path] = None,
    seed: Annotated[int, {"max": 1e8}] = 12345,
):
    func: type[Simulator] = Funcs(function).value
    if output is not None:
        output = Path(output)
        assert output.parent.exists()

    ui = start()
    t0 = timer(name=func.__name__)
    simulator: Simulator = func(ui, scale)  # simulate cylinder
    coords = simulator.prepare()
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "image.mrc"
        ui.cylinder_simulator.simulate_tilt_series(
            template_path=TEMPLATE_PATH,
            save_path=tmpfile,
            n_tilt=n_tilt,
            scale=scale,
        )
        for _rep in trange(nrepeat):
            for _idx, _nsr in enumerate(tqdm(nsr)):
                ui.cylinder_simulator.simulate_tomogram_from_tilt_series(
                    path=tmpfile,
                    nsr=_nsr,
                    bin_size=binsize,
                    tilt_range=(-60, 60),
                    height=60.0,
                    seed=seed + _rep * len(nsr) + _idx,
                )
                ui.register_path(coords, err_max=1e-8)
                ui.measure_radius(splines=[0])
                ui.tomogram.splines[0].anchors = simulator.anchors()
                ui.local_ft_analysis(
                    splines=[0], depth=49.0, interval=None, bin_size=binsize
                )
                results.append([_nsr, _rep, *simulator.results()])

        columns = ["nsr", "rep"] + simulator.columns()
        results = pl.DataFrame(results, schema=columns).sort(by="nsr")
        if output is None:
            view = DataFrameView()
            view.value = results
            view.show()
        else:
            results.write_csv(output)
        agg_df = results.group_by("nsr").agg(
            [
                pl.format(
                    "{}±{}", pl.col(x).mean().round(3), pl.col(x).std().round(3)
                ).alias(x)
                for x in results.columns[2:]
            ]
        )
        print(agg_df)
    t0.toc(log=False)
    ui.parent_viewer.close()


def preview(
    function: Funcs = Funcs.local_expansion,
    scale: float = 0.5,
):
    func: type[Simulator] = Funcs(function).value
    ui = start()
    simulator: Simulator = func(ui, scale)
    simulator.prepare()
    spl = ui.cylinder_simulator.spline
    ui.tomogram.splines.append(spl)
    ui._add_spline_instance(spl)


btn = PushButton(text="Preview")
btn.clicked.connect(lambda: preview(simulate.function.value, simulate.scale.value))
simulate.append(btn)

if __name__ == "__main__":
    print(" --- starting simulation --- ")
    simulate.show(run=True)
