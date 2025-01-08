from pathlib import Path

from cylindra.utils._test_utils import (
    PROJECT_DIR_14PF,
    assert_molecules_equal,
    assert_splines_close,
)
from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import relion


def test_load_and_save_starfiles(ui: CylindraMainWidget, tmpdir):
    ui.load_project(PROJECT_DIR_14PF)
    mole_path = Path(tmpdir) / "mole.star"
    spline_path = Path(tmpdir) / "spl.star"
    relion.save_molecules(ui, mole_path, ui.mole_layers)
    relion.save_splines(ui, spline_path, interval=1.0)
    all_mole = list(ui.mole_layers)
    all_spl = list(ui.splines)
    ui.mole_layers.clear()
    ui.splines.clear()
    relion.load_molecules(ui, mole_path)
    assert len(ui.mole_layers) == len(all_mole)
    for mole, mole_new in zip(ui.mole_layers, all_mole, strict=True):
        assert_molecules_equal(mole, mole_new, atol=1e-5)
    relion.load_splines(ui, spline_path)
    assert len(ui.splines) == len(all_spl)
    for spl, spl_new in zip(ui.splines, all_spl, strict=True):
        assert_splines_close(spl, spl_new)
