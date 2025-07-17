from pathlib import Path

import impy as ip
import pytest

from cylindra.utils._test_utils import (
    PROJECT_DIR_14PF,
    TEST_DIR,
    assert_molecules_equal,
    assert_splines_close,
)
from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import relion

TEST_JOB_DIR = Path(__file__).parent / "test_jobs"
JOB_TOMO_DIR = TEST_JOB_DIR / "Tomograms" / "job_tomo"
JOB_PICK_DIR = TEST_JOB_DIR / "Picks" / "job_picks"


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


def test_opening_jobs(ui: CylindraMainWidget):
    path_13pf = JOB_TOMO_DIR.joinpath("tomograms", "13pf_MT.mrc")
    path_14pf = JOB_TOMO_DIR.joinpath("tomograms", "14pf_MT.mrc")
    if not (tomo_dir := JOB_TOMO_DIR.joinpath("tomograms")).exists():
        tomo_dir.mkdir()
    if not path_13pf.exists():
        ip.imread(TEST_DIR / "13pf_MT.tif").imsave(path_13pf)
    if not path_14pf.exists():
        ip.imread(TEST_DIR / "14pf_MT.tif").imsave(path_14pf)
    relion.open_relion_job(ui, JOB_TOMO_DIR / "job.star")
    assert len(ui.batch.constructor.projects) == 2
    ui.batch.constructor.projects[0].send_to_viewer()
    assert ui.tomogram.scale == pytest.approx(1.052)
    assert not ui.tomogram.is_dummy

    relion.open_relion_job(ui, JOB_PICK_DIR / "job.star")
    assert len(ui.batch.constructor.projects) == 2
    ui.batch.constructor.projects[0].send_to_viewer()
    assert ui.tomogram.scale == pytest.approx(1.052)
    assert not ui.tomogram.is_dummy
