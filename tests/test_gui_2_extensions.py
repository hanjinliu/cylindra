import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from cylindra.components import CylSpline
from cylindra.ext import CommandNotFound
from cylindra.types import MoleculesLayer
from cylindra.widgets import CylindraMainWidget

from ._const import PROJECT_DIR_14PF, TEST_DIR


def test_IMOD(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF)
    ui.splines.pop()
    ui.mole_layers.pop()
    imod = ui.FileMenu.IMOD
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        spline_path = tmpdir / "splines.mod"
        with suppress(CommandNotFound):
            imod.save_molecules(tmpdir, ui.mole_layers.nth(0))
            imod.load_molecules(tmpdir / "coordinates.mod", tmpdir / "angles.csv")
            _assert_molecules_equal(ui.mole_layers.nth(0), ui.mole_layers.nth(1))
        with suppress(CommandNotFound):
            imod.save_splines(spline_path, interval=1.0)
            imod.load_splines(spline_path)
            _assert_splines_close(ui.splines[0], ui.splines[1], tol=0.06)
        with suppress(CommandNotFound):
            imod.export_project(
                ui.mole_layers.nth(0),
                tmpdir,
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
            _assert_molecules_equal(mole, mole_new, atol=1e-5)
        for spl, spl_new in zip(ui.splines, all_spl, strict=True):
            _assert_splines_close(spl, spl_new)


def _assert_molecules_equal(
    mole1: MoleculesLayer, mole2: MoleculesLayer, rtol=1e-5, atol=1e-2
):
    assert_allclose(mole1.molecules.pos, mole2.molecules.pos, rtol=rtol, atol=atol)
    # comparing quaternion is not safe.
    assert_allclose(mole1.molecules.x, mole2.molecules.x, rtol=rtol, atol=atol)
    assert_allclose(mole1.molecules.y, mole2.molecules.y, rtol=rtol, atol=atol)
    assert_allclose(mole1.molecules.z, mole2.molecules.z, rtol=rtol, atol=atol)


def _assert_splines_close(spl0: CylSpline, spl1: CylSpline, tol=1e-2):
    diff = np.sqrt(np.sum((spl0.partition(n=100) - spl1.partition(n=100)) ** 2))
    assert diff < tol
