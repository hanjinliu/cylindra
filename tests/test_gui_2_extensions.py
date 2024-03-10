import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from cylindra.ext import CommandNotFound
from cylindra.types import MoleculesLayer
from cylindra.widgets import CylindraMainWidget

from ._const import PROJECT_DIR_14PF, TEST_DIR


def test_IMOD(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF)
    imod = ui.FileMenu.IMOD
    with tempfile.TemporaryDirectory() as tmpdir:
        with suppress(CommandNotFound):
            imod.save_molecules(Path(tmpdir), ui.mole_layers.nth(0))
        with suppress(CommandNotFound):
            imod.export_project(
                ui.mole_layers.nth(0),
                Path(tmpdir),
                template_path=TEST_DIR / "beta-tubulin.mrc",
                mask_params=(0.3, 0.8),
            )


def test_RELION(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF)
    relion = ui.FileMenu.RELION
    with tempfile.TemporaryDirectory() as tmpdir:
        starpath = Path(tmpdir) / "mole.star"
        spline_path = Path(tmpdir) / "spl.star"
        relion.save_molecules(starpath, ui.mole_layers)
        relion.save_splines(spline_path, interval=1.0)
        all_mole = list(ui.mole_layers)
        all_spl = list(ui.splines)
        ui.mole_layers.clear()
        ui.splines.clear()
        relion.load_molecules(starpath)
        relion.load_splines(spline_path)
        assert len(ui.mole_layers) == len(all_mole)
        assert len(ui.splines) == len(all_spl)
        for mole, mole_new in zip(ui.mole_layers, all_mole, strict=True):
            _assert_molecules_equal(mole, mole_new)
        for spl, spl_new in zip(ui.splines, all_spl, strict=True):
            diff = np.sqrt(
                np.sum((spl.partition(n=100) - spl_new.partition(n=100)) ** 2)
            )
            assert diff < 1e-2


def _assert_molecules_equal(mole1: MoleculesLayer, mole2: MoleculesLayer):
    assert_allclose(mole1.molecules.pos, mole2.molecules.pos, rtol=1e-5, atol=1e-5)
    assert_allclose(
        mole1.molecules.quaternion(),
        mole2.molecules.quaternion(),
        rtol=1e-5,
        atol=1e-5,
    )
