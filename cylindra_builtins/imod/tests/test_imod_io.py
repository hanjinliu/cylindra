from pathlib import Path

import impy as ip

from cylindra.utils._test_utils import (
    PROJECT_DIR_13PF,
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

    ui.batch.constructor.add_projects(
        paths=[PROJECT_DIR_13PF, PROJECT_DIR_14PF],
    )
    imod.export_project_batch(ui, tmpdir, ui.batch._get_loader_paths())


def test_import_project(ui: CylindraMainWidget, tmpdir):
    tmpdir = Path(tmpdir)
    root = Path(__file__).parent / "imod_test_projects"
    path1 = root / "position_1" / "position_1_rec.mrc"
    path2 = root / "position_2" / "position_2.rec"  # old version
    ip.imread(TEST_DIR / "13pf_MT.tif").imsave(path1)
    ip.imread(TEST_DIR / "14pf_MT.tif").imsave(path2)
    try:
        imod.open_image_from_imod_project(ui, root / "position_1" / "position_1.edf")
        imod.import_imod_projects(ui, root / "*/*.edf", project_root=tmpdir)
        assert len(ui.batch.constructor.projects) == 2
    finally:
        path1.unlink(missing_ok=True)
        path2.unlink(missing_ok=True)
