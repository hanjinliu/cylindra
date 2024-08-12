import tempfile
from pathlib import Path

from cylindra.utils.testing import (
    PROJECT_DIR_14PF,
    assert_molecules_equal,
    assert_splines_close,
)
from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import relion


def test_load_and_save_starfiles(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF)
    with tempfile.TemporaryDirectory() as tmpdir:
        starpath = Path(tmpdir) / "mole.star"
        spline_path = Path(tmpdir) / "spl.star"
        relion.save_molecules(ui, starpath, ui.mole_layers)
        relion.save_splines(ui, spline_path, interval=1.0)
        all_mole = list(ui.mole_layers)
        all_spl = list(ui.splines)
        ui.mole_layers.clear()
        ui.splines.clear()
        relion.load_molecules(ui, starpath)
        relion.load_splines(ui, spline_path)
        assert len(ui.mole_layers) == len(all_mole)
        assert len(ui.splines) == len(all_spl)
        for mole, mole_new in zip(ui.mole_layers, all_mole, strict=True):
            assert_molecules_equal(mole, mole_new, atol=1e-5)
        for spl, spl_new in zip(ui.splines, all_spl, strict=True):
            assert_splines_close(spl, spl_new)
