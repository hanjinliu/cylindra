import tempfile
from enum import Enum
from timeit import default_timer
from typing import Annotated, Any

import napari
import numpy as np
import polars as pl
from magicclass import MagicTemplate, magicclass, vfield
from magicclass.ext.polars import DataFrameView
from magicclass.types import Optional, Path
from numpy.typing import NDArray

from cylindra import start  # NOTE: Set ApplicationAttributes
from cylindra.components import CylSpline
from cylindra.const import PropertyNames as H
from cylindra.widgets import CylindraMainWidget
from scripts.user_consts import TEMPLATE_X

POSITIONS = [(0, 15), (15, 30), (30, 45), (45, 60)]


class timer:
    def __init__(self, name: str):
        self.name = name
        self.start = default_timer()

    def toc(self):
        dt = default_timer() - self.start
        print(f"{self.name} ({dt:.1f} sec)")


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
        self.ui.simulator.create_image_with_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.simulator.generate_molecules(
            0, spacing=4.125, twist=0.08, start=3, radius=11.2, npf=13
        )
        for exp, yrange in zip([-0.075, -0.025, 0.025, 0.075], POSITIONS, strict=False):
            self.ui.simulator.expand(
                layer=self.ui.mole_layers.last(),
                by=exp,
                yrange=yrange,
                arange=(0, 13),
                allev=False,
            )
        return np.array([[30, 30, 30], [30, 210, 30]])

    def results(self):
        return self.ui.tomogram.splines[0].props.loc[H.spacing]

    def columns(self) -> list[str]:
        return ["4.05", "4.10", "4.15", "4.20"]


class _local_skew_base(Simulator):
    _VALUES: list[str]

    def results(self):
        return self.ui.tomogram.splines[0].props.loc[H.twist]

    def columns(self) -> list[str]:
        return self._VALUES

    def values(self) -> list[float]:
        return [float(v) for v in self._VALUES]


class local_skew_13_3(_local_skew_base):
    """Vertical MT with twists."""

    _VALUES = ["-0.12", "-0.04", "0.04", "0.12"]

    def prepare(self):
        self.ui.simulator.create_image_with_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.simulator.generate_molecules(
            0, spacing=4.1, twist=0.0, start=3, radius=11.2, npf=13
        )
        for sk, yrange in zip(self.values(), POSITIONS, strict=False):
            self.ui.simulator.twist(
                self.ui.mole_layers.last(),
                by=sk,
                yrange=yrange,
                arange=(0, 13),
                allev=False,
            )
        return np.array([[30, 30, 30], [30, 210, 30]])


class local_skew_14_3(_local_skew_base):
    """Vertical MT with twists."""

    _VALUES = ["-0.37", "-0.29", "-0.21", "-0.13"]

    def prepare(self):
        self.ui.simulator.create_image_with_straight_line(
            scale=self.scale, size=(60.0, 240.0, 60.0), length=245.0
        )
        self.ui.simulator.generate_molecules(
            0, spacing=4.1, twist=0.0, start=3, radius=12.1, npf=14
        )
        for sk, yrange in zip(self.values(), POSITIONS, strict=False):
            self.ui.simulator.twist(
                self.ui.mole_layers.last(),
                by=sk,
                yrange=yrange,
                arange=(0, 13),
                allev=False,
            )
        return np.array([[30, 30, 30], [30, 210, 30]])


class local_orientation(Simulator):
    """Cureved MT with orientation=60, 40, 20, 0 deg."""

    def get_coords(self):
        length = 60.0
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
        self.ui.simulator.create_empty_image(size=(60, 256, 152), scale=self.scale)
        self.ui.register_path(coords, err_max=1e-8)
        self.ui.simulator.generate_molecules(
            0, spacing=4.1, twist=0.0, start=3, radius=11.2, npf=13
        )
        return coords

    def results(self):
        df = self.ui.tomogram.splines[0].props.loc
        return df[H.spacing].to_list() + df[H.twist].to_list() + df[H.rise].to_list()

    def anchors(self) -> np.ndarray:
        coords = self.get_coords()
        spl = CylSpline().fit(coords, err_max=1e-8)
        clip = 30 / spl.length()
        return np.linspace(clip, 1 - clip, 4)

    def columns(self):
        return [f"{n}{i}" for n in ["spacing", "twist", "rise"] for i in range(4)]


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
        self.ui.simulator.create_empty_image(size=(60, 230, 60), scale=self.scale)
        self.ui.register_path(coords, err_max=1e-8)
        self.ui.simulator.generate_molecules(
            0, spacing=4.1, twist=0.00, start=3, radius=11.2, npf=13
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
            + df[H.twist].to_list()
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
            for n in ["spacing", "twist", "rise", "curvature"]
            for i in range(self.N_ANCHORS)
        ]


class Funcs(Enum):
    local_expansion = local_expansion
    local_skew_13_3 = local_skew_13_3
    local_skew_14_3 = local_skew_14_3
    local_orientation = local_orientation
    local_curvature = local_curvature


@magicclass(widget_type="scrollable")
class Main(MagicTemplate):
    function = vfield(Funcs)
    n_tilt = vfield(61).with_options(min=1)
    nsr = vfield([0.1, 3.5])
    nrepeat = vfield(5)
    scale = vfield(0.5)
    binsize = vfield(1).with_options(min=1)
    output = vfield(Optional[Path.Save["*.csv"]])
    seed = vfield(12345).with_options(max=1e8)

    def __post_init__(self):
        self._ui = start()
        self.min_width = 320
        self._ui.parent_viewer.window.add_dock_widget(
            self,
            area="left",
            tabify=True,
        ).setFloating(True)

    def simulate(
        self,
        function: Annotated[Funcs, {"bind": function}] = Funcs.local_expansion,
        n_tilt: Annotated[int, {"bind": n_tilt}] = 21,
        nsr: Annotated[Any, {"bind": nsr}] = [0.1],
        nrepeat: Annotated[int, {"bind": nrepeat}] = 1,
        scale: Annotated[float, {"bind": scale}] = 0.5,
        binsize: Annotated[int, {"bind": binsize}] = 1,
        output: Annotated[str, {"bind": output}] = None,
        seed: Annotated[int, {"bind": seed}] = 12345,
    ):
        if isinstance(function, str):
            function = getattr(Funcs, function)
        func: type[Simulator] = Funcs(function).value
        if output is not None:
            assert Path(output).parent.exists()

        ui = self._ui
        t0 = timer(name=func.__name__)
        simulator: Simulator = func(ui, scale)  # simulate cylinder
        coords = simulator.prepare()
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            ui.simulator.simulate_tilt_series(
                components=[(ui.mole_layers.last().name, TEMPLATE_X)],
                save_dir=tmpdir,
                tilt_range=(-60, 60),
                n_tilt=n_tilt,
            )
            for _rep in range(nrepeat):
                for _idx, _nsr in enumerate(nsr):
                    t1 = timer(name=f"repeat={_rep}, nsr={_nsr:.2f} done")
                    ui.simulator.simulate_tomogram_from_tilt_series(
                        path=Path(tmpdir) / "image.mrc",
                        nsr=_nsr,
                        bin_size=binsize,
                        tilt_range=(-60, 60),
                        height=60.0,
                        seed=seed + _rep * len(nsr) + _idx,
                    )
                    ui.register_path(coords, err_max=1e-8)
                    ui.measure_radius(splines=[0])
                    ui.tomogram.splines[0].anchors = simulator.anchors()
                    ui.local_cft_analysis(
                        [0], depth=49.0, interval=None, bin_size=binsize
                    )
                    results.append([_nsr, _rep, *simulator.results()])
                    t1.toc()

            columns = ["nsr", "rep"] + simulator.columns()
            results = pl.DataFrame(results, schema=columns).sort(by="nsr")

        if output is None:
            view = DataFrameView()
            view.value = results
            ui.parent_viewer.window.add_dock_widget(view)
        else:
            results.write_csv(output)
        t0.toc()
        return

    def preview(
        self,
        function: Annotated[Funcs, {"bind": function}] = Funcs.local_expansion,
        scale: Annotated[float, {"bind": scale}] = 0.5,
    ):
        func: type[Simulator] = Funcs(function).value
        ui = self._ui
        simulator: Simulator = func(ui, scale)
        simulator.prepare()


if __name__ == "__main__":
    # Example
    # -------
    # $ python .\scripts\simulate_cft.py "function='local_expansion', n_tilt=61, nsr=[0.1, 3.0, 3.5, 4.0], nrepeat=20, scale=0.2615, binsize=2, output=r'C:\Users\Uemura-Lab\Desktop\simulate_CFT_results\cft_expansion.csv', seed=111"
    # $ python .\scripts\simulate_cft.py "function='local_skew_13_3', n_tilt=61, nsr=[0.1, 3.0, 3.5, 4.0], nrepeat=20, scale=0.2615, binsize=2, output=r'C:\Users\Uemura-Lab\Desktop\simulate_CFT_results\cft_twist_13_3.csv', seed=222"
    # $ python .\scripts\simulate_cft.py "function='local_skew_14_3', n_tilt=61, nsr=[0.1, 3.0, 3.5, 4.0], nrepeat=20, scale=0.2615, binsize=2, output=r'C:\Users\Uemura-Lab\Desktop\simulate_CFT_results\cft_twist_14_3.csv', seed=333"
    # $ python .\scripts\simulate_cft.py "function='local_orientation', n_tilt=61, nsr=[0.1, 3.0, 3.5, 4.0], nrepeat=20, scale=0.2615, binsize=2, output=r'C:\Users\Uemura-Lab\Desktop\simulate_CFT_results\cft_orientation.csv', seed=444"
    # $ python .\scripts\simulate_cft.py "function='local_curvature', n_tilt=61, nsr=[0.1, 3.0, 3.5, 4.0], nrepeat=20, scale=0.2615, binsize=2, output=r'C:\Users\Uemura-Lab\Desktop\simulate_CFT_results\cft_curvature.csv', seed=555"
    import inspect
    import sys

    if args := sys.argv[1:]:
        code = f"dict({args[0]})"
        kwargs = eval(code, {}, {})
        inspect.signature(Main.simulate).bind(dict(self=None, **kwargs))
    else:
        kwargs = None
    ui = Main()
    if kwargs is not None:
        ui.simulate(**kwargs)
    else:
        napari.run()
