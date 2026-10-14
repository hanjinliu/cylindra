from pathlib import Path

import impy as ip
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from cylindra.utils._test_utils import (
    PROJECT_DIR_14PF,
    TEST_DIR,
    assert_molecules_equal,
    assert_splines_close,
)
from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import relion
from cylindra_builtins.relion.io import (
    _mole_to_star_df,
    _particles_to_molecules,
    _preview_open_relion_job,
)

TEST_JOB_DIR = Path(__file__).parent / "test_jobs"
JOB_TOMO_DIR = TEST_JOB_DIR / "Tomograms" / "job_tomo"
JOB_PICK_DIR = TEST_JOB_DIR / "Picks" / "job_picks"
JOB_CLASS3D_DIR = TEST_JOB_DIR / "Class3D" / "job_class3d"
JOB_REFINE_DIR = TEST_JOB_DIR / "Refine3D" / "job_refine"


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


@pytest.mark.parametrize("has_subtomo_angles", [False, True])
def test_particles_to_molecules_follows_relion(has_subtomo_angles: bool):
    rng = np.random.default_rng(0)
    n = 10
    center_zyx = np.array([20.0, 30.0, 40.0])
    df = pd.DataFrame(
        {
            "rlnTomoName": ["TS_01"] * n,
            "rlnCenteredCoordinateXAngst": rng.uniform(-300, 300, n),
            "rlnCenteredCoordinateYAngst": rng.uniform(-300, 300, n),
            "rlnCenteredCoordinateZAngst": rng.uniform(-100, 100, n),
            "rlnAngleRot": rng.uniform(-180, 180, n),
            "rlnAngleTilt": rng.uniform(0, 180, n),
            "rlnAnglePsi": rng.uniform(-180, 180, n),
            "rlnOriginXAngst": rng.uniform(-10, 10, n),
            "rlnOriginYAngst": rng.uniform(-10, 10, n),
            "rlnOriginZAngst": rng.uniform(-10, 10, n),
        }
    )
    if has_subtomo_angles:
        df["rlnTomoSubtomogramRot"] = rng.uniform(-180, 180, n)
        df["rlnTomoSubtomogramTilt"] = rng.uniform(0, 180, n)
        df["rlnTomoSubtomogramPsi"] = rng.uniform(-180, 180, n)

    mole = _particles_to_molecules(df.copy(), center_zyx)["Mole-0"]

    # same as relion/src/tomography_python_programs/view/particles.py
    xyz = df[[f"rlnCenteredCoordinate{a}Angst" for a in "XYZ"]].to_numpy()
    origin_xyz = df[[f"rlnOrigin{a}Angst" for a in "XYZ"]].to_numpy()
    rot_particle = Rotation.from_euler(
        "ZYZ", df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]], degrees=True
    ).inv()
    if has_subtomo_angles:
        cols = ["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"]  # fmt: skip
        rot_subtomo = Rotation.from_euler("ZYZ", df[cols], degrees=True).inv()
    else:
        rot_subtomo = Rotation.identity(n)
    xyz = xyz - rot_subtomo.apply(origin_xyz)
    mat = rot_subtomo.as_matrix() @ rot_particle.as_matrix()

    assert_allclose(mole.pos, xyz[:, ::-1] / 10 + center_zyx, atol=1e-6)
    assert_allclose(mole.z, mat[:, ::-1, 2], atol=1e-6)
    assert_allclose(mole.y, mat[:, ::-1, 1], atol=1e-6)
    assert_allclose(mole.x, mat[:, ::-1, 0], atol=1e-6)
    assert mole.features.columns == ["rlnTomoName"]


def test_particles_to_molecules_not_centered(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF)
    mole = ui.mole_layers.nth(0).molecules
    df = _mole_to_star_df(
        [mole], ui.tomogram, "TS_01", shift_by_origin=False, centered=False
    )
    mole_new = _particles_to_molecules(
        df, center_zyx=(10.0, 10.0, 10.0), scale=ui.tomogram.scale
    )["Mole-0"]
    assert_allclose(mole_new.pos, mole.pos, atol=1e-5)
    assert_allclose(mole_new.rotator.as_matrix(), mole.rotator.as_matrix(), atol=1e-5)


def test_opening_jobs(ui: CylindraMainWidget, tmpdir):
    tmpdir = Path(tmpdir)
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
    assert ui.tomogram.tilt_model.tilt_range == pytest.approx((-60.1, 59.9))

    relion.open_relion_job(ui, JOB_PICK_DIR / "job.star")
    assert len(ui.batch.constructor.projects) == 2
    ui.batch.constructor.projects[0].send_to_viewer()
    assert ui.tomogram.scale == pytest.approx(1.052)
    assert not ui.tomogram.is_dummy
    assert ui.tomogram.tilt_model.tilt_range == pytest.approx((-60.1, 59.9))
    relion.save_molecules_for_import(
        ui, tmpdir / "p.star", ui.batch._get_loader_paths()
    )
    ui.batch.save_batch_project(tmpdir / "prj.zip")
    ui.batch.load_batch_project(tmpdir / "prj.zip")
    relion.save_molecules_for_import(
        ui,
        tmpdir / "p.star",
        ui.batch._get_loader_paths(),
        centered=True,
    )
    ui.batch.constructor.projects[0].molecules[0].check = False
    relion.save_molecules_for_import(
        ui,
        tmpdir / "p.star",
        ui.batch._get_loader_paths(),
    )

    # Class3D
    relion.open_relion_job(ui, JOB_CLASS3D_DIR / "job.star")

    relion.save_molecules_for_extract(
        ui,
        tmpdir / "p.star",
        ui.batch._get_loader_paths(),
    )

    # Refine3D
    relion.open_relion_job(ui, JOB_REFINE_DIR / "job.star")

    relion.save_molecules_for_extract(
        ui,
        tmpdir / "p.star",
        ui.batch._get_loader_paths(),
        relion_scale=1.1,
    )

    # test preview
    relion.open_relion_job(
        ui,
        JOB_REFINE_DIR / "job.star",
        join_molecules_by_id=True,
        path_sets=ui.batch._get_loader_paths(),
    )
    _preview_open_relion_job(ui, JOB_REFINE_DIR / "job.star")
