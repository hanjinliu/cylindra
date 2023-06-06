from pathlib import Path
import tempfile
from cylindra import start
from cylindra.const import PropertyNames as H
from tqdm import tqdm

TEMPLATE_PATH = Path(__file__).parent.parent / "tests" / "beta-tubulin.mrc"


def simulate_local_expansion(
    n_tilt: int = 21,
    nsr: list[float] = [1.5],
    nrepeat: int = 5,
    output_dir: Path | None = None,
):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    ui = start()
    ui.cylinder_simulator.create_straight_line(
        scale=0.5, size=(60, 180, 60), length=160
    )
    ui.cylinder_simulator.update_model(
        spacing=4.08, skew=0.08, rise=9.6, radius=10.4, npf=13
    )
    ui.cylinder_simulator.expand(exp=0.12, yrange=(15, 25), arange=(0, 13), allev=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ui.cylinder_simulator.simulate_tomogram(
            template_path=TEMPLATE_PATH,
            save_dir=tmpdir,
            nsr=list(nsr) * nrepeat,
            n_tilt=n_tilt,
            seed=41298764,
        )
        results = []
        for _rep in tqdm(range(nrepeat)):
            for _idx, _nsr in enumerate(tqdm(nsr)):
                fname = f"image-{len(nsr) * _rep + _idx}.mrc"
                ui.open_image(tmpdir / fname, tilt_range=(-60, 60))
                ui.register_path([[30, 25, 30], [30, 155, 30]])
                ui.refine_splines(splines=[0])
                ui.measure_radius(splines=[0])
                ui.local_ft_analysis(splines=[0], depth=32.64, interval=8.16)
                lprops = ui.get_spline(0).localprops  # shape (16, x)
                if output_dir is not None:
                    lprops.select([H.spacing, H.skew]).write_csv(
                        output_dir / f"n={_rep}_nsr={_nsr:3f}.csv"
                    )
                lprops[H.spacing]
                lprops[H.skew]
