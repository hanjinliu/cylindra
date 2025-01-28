from pathlib import Path

from cylindra.utils._test_utils import (
    PROJECT_DIR_14PF,
    TEST_DIR,
    assert_molecules_equal,
    assert_splines_close,
)
from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import imod


def test_load_and_save_mod_files(ui: CylindraMainWidget, tmpdir):
    ui.load_project(PROJECT_DIR_14PF)
    ui.splines.pop()
    ui.mole_layers.pop()
    tmpdir = Path(tmpdir)
    spline_path = tmpdir / "splines.mod"
    imod.save_molecules(ui, tmpdir, ui.mole_layers.nth(0))
    imod.load_molecules(ui, tmpdir / "coordinates.mod", tmpdir / "angles.csv")
    assert_molecules_equal(ui.mole_layers.nth(0), ui.mole_layers.nth(1))
    imod.save_splines(ui, spline_path, interval=1.0)
    imod.load_splines(ui, spline_path)
    assert_splines_close(ui.splines[0], ui.splines[1], tol=0.06)
    imod.export_project(
        ui,
        ui.mole_layers.nth(0),
        tmpdir,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
    )
