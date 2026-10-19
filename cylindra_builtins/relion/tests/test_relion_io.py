from pathlib import Path
from typing import NamedTuple

import impy as ip
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from cylindra.project import CylindraProject
from cylindra.utils._test_utils import (
    PROJECT_DIR_14PF,
    TEST_DIR,
    assert_molecules_equal,
    assert_splines_close,
)
from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import relion
from cylindra_builtins.relion._relion_utils import read_loop_block
from cylindra_builtins.relion.io import (
    POS_CENTERED,
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
    df = pl.DataFrame(
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
        df = df.with_columns(
            rlnTomoSubtomogramRot=rng.uniform(-180, 180, n),
            rlnTomoSubtomogramTilt=rng.uniform(0, 180, n),
            rlnTomoSubtomogramPsi=rng.uniform(-180, 180, n),
        )

    mole = _particles_to_molecules(df, center_zyx)["Mole-0"]

    # same as relion/src/tomography_python_programs/view/particles.py
    xyz = df.select([f"rlnCenteredCoordinate{a}Angst" for a in "XYZ"]).to_numpy()
    origin_xyz = df.select([f"rlnOrigin{a}Angst" for a in "XYZ"]).to_numpy()
    rot_particle = Rotation.from_euler(
        "ZYZ",
        df.select(["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]).to_numpy(),
        degrees=True,
    ).inv()
    if has_subtomo_angles:
        cols = ["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"]  # fmt: skip
        rot_subtomo = Rotation.from_euler(
            "ZYZ", df.select(cols).to_numpy(), degrees=True
        ).inv()
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
    # scale of the batch analyzer is determined by the tomograms.star file
    assert ui.batch.constructor.scale.value == pytest.approx(1.052)

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


TS_SCALE = 1.052  # nm/pixel of the tilt series


class RelionTiltSeriesJobs(NamedTuple):
    align: Path  # Align tilt series job directory
    tomo: Path  # Reconstruct tomograms job directory
    pick: Path  # Pick job directory
    tilt_series_star: Path
    tomogram: Path  # bin-2 tomogram reconstructed from the tilt series
    particles_zyx: np.ndarray  # centered coordinates of the particles in Angstrom
    tomo_bin25: Path  # Reconstruct tomograms job directory (bin 2.5)


def _write_star(path: Path, name: str, columns: dict[str, list], loop: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# version 50001", "", f"data_{name}", ""]
    if loop:
        lines.append("loop_")
        lines.extend(f"_{col} #{i + 1}" for i, col in enumerate(columns))
        rows = zip(*columns.values(), strict=True)
        lines.extend(" ".join(map(str, row)) for row in rows)
    else:
        lines.extend(f"_{col} {values[0]}" for col, values in columns.items())
    path.write_text("\n".join(lines) + "\n")


def _write_job_star(job_dir: Path, job_type: str):
    _write_star(job_dir / "job.star", "job", {"rlnJobTypeLabel": [job_type]}, False)


def _project_around_y(vol: np.ndarray, degrees: np.ndarray, width: int) -> np.ndarray:
    """Projections of the volume around the y axis (same as tests/test_tilt_series)."""
    nz, ny, nx = vol.shape
    nt = int(np.ceil(np.hypot(nz, nx))) + 2
    jj, uu, tt = np.meshgrid(
        np.arange(ny),
        np.arange(width) - width // 2,
        np.arange(nt) - nt // 2,
        indexing="ij",
    )
    out = np.empty((degrees.size, ny, width), dtype=np.float32)
    for k, rad in enumerate(np.deg2rad(degrees)):
        zz = uu * np.sin(rad) + tt * np.cos(rad) + nz // 2
        xx = uu * np.cos(rad) - tt * np.sin(rad) + nx // 2
        out[k] = ndi.map_coordinates(vol, [zz, jj, xx], order=1).sum(axis=-1)
    return out


@pytest.fixture(scope="module")
def relion_tilt_series_jobs(tmp_path_factory: pytest.TempPathFactory):
    """A RELION project of a tilt series simulated from the test tomogram."""
    mrcfile = pytest.importorskip("mrcfile")
    acryo = pytest.importorskip("acryo")
    if not hasattr(acryo, "BackProjectionLoader"):
        pytest.skip("BackProjectionLoader is not available")
    from acryo.projection import TiltSeries

    rln_dir = tmp_path_factory.mktemp("relion")
    vol = np.asarray(ip.imread(TEST_DIR / "13pf_MT.tif"), dtype=np.float32)
    vol -= vol.mean()
    degrees = np.arange(-60.0, 61.0, 6.0)
    stack_path = rln_dir / "MotionCorr" / "job002" / "TS_01.mrc"
    stack_path.parent.mkdir(parents=True)
    # dark density on a light background, as in cryo-EM images
    with mrcfile.new(stack_path) as f:
        f.set_data(-_project_around_y(vol, degrees, width=104))
        f.voxel_size = TS_SCALE * 10

    # Align tilt series job
    align_dir = rln_dir / "AlignTiltSeries" / "job005"
    ts_star = align_dir / "tilt_series" / "TS_01.star"
    ntilts = degrees.size
    _write_star(
        ts_star,
        "TS_01",
        {
            "rlnMicrographName": [
                f"{i + 1}@MotionCorr/job002/TS_01.mrc" for i in range(ntilts)
            ],
            "rlnTomoNominalStageTiltAngle": degrees.tolist(),
            "rlnTomoXTilt": [0.0] * ntilts,
            "rlnTomoYTilt": degrees.tolist(),
            "rlnTomoZRot": [0.0] * ntilts,
            "rlnTomoXShiftAngst": [0.0] * ntilts,
            "rlnTomoYShiftAngst": [0.0] * ntilts,
        },
    )
    global_columns = {
        "rlnTomoName": ["TS_01"],
        "rlnTomoTiltSeriesStarFile": ["AlignTiltSeries/job005/tilt_series/TS_01.star"],
        "rlnTomoTiltSeriesPixelSize": [TS_SCALE * 10],
        "rlnTomoHand": [-1.0],
    }
    _write_star(align_dir / "aligned_tilt_series.star", "global", global_columns)
    _write_job_star(align_dir, "relion.aligntiltseries")

    # Reconstruct tomograms job (bin 2)
    tomo_dir = rln_dir / "Tomograms" / "job006"
    tomo_path = tomo_dir / "tomograms" / "rec_TS_01.mrc"
    tomo_path.parent.mkdir(parents=True)
    loader = acryo.BackProjectionLoader(
        TiltSeries.from_relion_tilt_series_star(ts_star), binsize=2
    )
    with mrcfile.new(tomo_path) as f:
        f.set_data(loader.reconstruct_tomogram((18, 103, 47), dtype=np.float32))
        f.voxel_size = TS_SCALE * 2 * 10
    nz, ny, nx = vol.shape
    tomo_columns = global_columns | {
        "rlnTomoTomogramBinning": [2.0],
        "rlnTomoSizeX": [nx],
        "rlnTomoSizeY": [ny],
        "rlnTomoSizeZ": [nz],
        "rlnTomoReconstructedTomogram": ["Tomograms/job006/tomograms/rec_TS_01.mrc"],
    }
    _write_star(tomo_dir / "tomograms.star", "global", tomo_columns)
    _write_job_star(tomo_dir, "relion.reconstructtomograms")

    # Reconstruct tomograms job of a non-integer binning (bin 2.5)
    tomo25_dir = rln_dir / "Tomograms" / "job008"
    tomo25_path = tomo25_dir / "tomograms" / "rec_TS_01.mrc"
    tomo25_path.parent.mkdir(parents=True)
    with mrcfile.new(tomo25_path) as f:
        f.set_data(
            loader.binning(1.25).reconstruct_tomogram((14, 82, 37), dtype=np.float32)
        )
        f.voxel_size = TS_SCALE * 2.5 * 10
    tomo25_columns = tomo_columns | {
        "rlnTomoTomogramBinning": [2.5],
        "rlnTomoReconstructedTomogram": ["Tomograms/job008/tomograms/rec_TS_01.mrc"],
    }
    _write_star(tomo25_dir / "tomograms.star", "global", tomo25_columns)
    _write_job_star(tomo25_dir, "relion.reconstructtomograms")

    # Pick job
    pick_dir = rln_dir / "Picks" / "job007"
    particles_zyx = np.array([[0.0, 0.0, 0.0], [52.0, -310.0, 105.0]])
    _write_star(
        pick_dir / "particles.star",
        "particles",
        {
            "rlnTomoName": ["TS_01"] * 2,
            "rlnTomoParticleName": ["TS_01/1", "TS_01/2"],
            "rlnCenteredCoordinateXAngst": particles_zyx[:, 2].tolist(),
            "rlnCenteredCoordinateYAngst": particles_zyx[:, 1].tolist(),
            "rlnCenteredCoordinateZAngst": particles_zyx[:, 0].tolist(),
        },
    )
    _write_star(
        pick_dir / "optimisation_set.star",
        "optimisation_set",
        {
            "rlnTomoTomogramsFile": ["Tomograms/job006/tomograms.star"],
            "rlnTomoParticlesFile": ["Picks/job007/particles.star"],
        },
        loop=False,
    )
    _write_job_star(pick_dir, "relion.picktomo")
    return RelionTiltSeriesJobs(
        align_dir, tomo_dir, pick_dir, ts_star, tomo_path, particles_zyx, tomo25_dir
    )


def _corr(a, b) -> float:
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))


def _batch_project(ui: CylindraMainWidget, idx: int = 0) -> CylindraProject:
    return CylindraProject.from_file(ui.batch.constructor.projects[idx].path)


def test_open_relion_job_from_tilt_series(
    ui: CylindraMainWidget,
    relion_tilt_series_jobs: RelionTiltSeriesJobs,
    tmp_path: Path,
):
    jobs = relion_tilt_series_jobs

    # Align tilt series job has no tomograms, so the coarse tomograms are
    # reconstructed from the tilt series
    relion.open_relion_job(
        ui,
        jobs.align / "job.star",
        project_root=tmp_path / "align",
        reference_scale=TS_SCALE * 2,
        tomo_height=42.0,  # nm, 20 slices of 2.104 nm
    )
    assert len(ui.batch.constructor.projects) == 1
    # scale of the batch analyzer is the pixel size of the tilt series
    assert ui.batch.constructor.scale.value == pytest.approx(TS_SCALE)
    prj = _batch_project(ui)
    assert prj.image is None
    assert prj.scale == pytest.approx(TS_SCALE)
    assert prj.multiscales == []
    assert not prj.invert
    assert prj.invert_reference  # reconstructed tomogram is light-background
    assert prj.missing_wedge.as_param() == {"kind": "y", "range": (-60.0, 60.0)}
    ts_info = prj.tilt_series
    assert Path(ts_info.path) == jobs.tilt_series_star
    assert ts_info.reference is None
    assert ts_info.reference_binsize == 2
    assert ts_info.reference_shape == (20, 103, 52)  # tilt images are (206, 104)
    assert ts_info.tomo_hand == 1
    ui.batch.constructor.projects[0].send_to_viewer()
    assert ui.tomogram.tilt_series is not None
    assert ui.tomogram.tilt_series.reference.shape == (20, 103, 52)
    assert ui.tomogram.scale == pytest.approx(TS_SCALE)
    assert str(ui.macro[-1]).startswith("ui.open_tilt_series(")

    # Reconstruct job: the RELION tomograms of an integer binning are the coarse
    # tomograms
    relion.open_relion_job(
        ui,
        jobs.tomo / "job.star",
        project_root=tmp_path / "tomo",
        use_tilt_series=True,
        bin_size=[1, 2],
        reference_scale=None,
    )
    prj = _batch_project(ui)
    ts_info = prj.tilt_series
    assert Path(ts_info.reference) == jobs.tomogram
    assert ts_info.reference_binsize == 2
    assert ts_info.reference_shape == (18, 103, 47)
    assert ts_info.binsizes == [1, 2]
    assert prj.invert_reference  # `invert` is True by default
    ui.batch.constructor.projects[0].send_to_viewer()
    tomo = ui.tomogram
    assert tomo.tilt_series.reference_binsize == 2
    assert tomo.shape == (36, 206, 94)
    assert_allclose(
        tomo.tilt_series.tomogram_center, np.array([9, 51, 23]) * 2 * TS_SCALE
    )
    # the coarse tomogram is aligned with the images reconstructed from the tilt series
    sl = (slice(4, 14), slice(20, 80), slice(10, 40))
    region = tomo.get_multiscale_or_original(2)[sl].compute()
    assert _corr(region, -np.asarray(tomo.tilt_series.reference)[sl]) > 0.8

    # RELION tomograms of a non-integer binning are also the coarse tomograms
    relion.open_relion_job(
        ui,
        jobs.tomo_bin25 / "job.star",
        project_root=tmp_path / "tomo-bin2.5",
        use_tilt_series=True,
        reference_scale=None,
    )
    ts_info = _batch_project(ui).tilt_series
    assert Path(ts_info.reference).parent.parent == jobs.tomo_bin25
    assert ts_info.reference_binsize == 2.5
    assert ts_info.reference_shape == (14, 82, 37)
    ui.batch.constructor.projects[0].send_to_viewer()
    tomo = ui.tomogram
    assert tomo.tilt_series.reference_binsize == 2.5
    assert tomo.tilt_series.reference.scale.x == pytest.approx(TS_SCALE * 2.5)
    assert tomo.shape == (35, 205, 92)

    # coarse tomograms of a different bin size are reconstructed
    relion.open_relion_job(
        ui,
        jobs.tomo / "job.star",
        project_root=tmp_path / "tomo-bin3",
        use_tilt_series=True,
        reference_scale=TS_SCALE * 3,
    )
    ts_info = _batch_project(ui).tilt_series
    assert ts_info.reference is None
    assert ts_info.reference_binsize == 3
    assert ts_info.reference_shape == (12, 68, 31)
    relion.open_relion_job(
        ui,
        jobs.tomo / "job.star",
        project_root=tmp_path / "tomo-bin2.5-rec",
        use_tilt_series=True,
        reference_scale=TS_SCALE * 2.5,
    )
    ts_info = _batch_project(ui).tilt_series
    assert ts_info.reference is None
    assert ts_info.reference_binsize == 2.5
    assert ts_info.reference_shape == (14, 82, 37)

    # tomograms are still available
    relion.open_relion_job(ui, jobs.tomo / "job.star", project_root=tmp_path / "rec")
    prj = _batch_project(ui)
    assert prj.tilt_series is None
    assert Path(prj.image) == jobs.tomogram
    # scale of the batch analyzer is the pixel size of the tomograms (bin 2)
    assert ui.batch.constructor.scale.value == pytest.approx(TS_SCALE * 2)

    # particles are in the tomogram space of the tilt series
    relion.open_relion_job(
        ui, jobs.pick / "job.star", project_root=tmp_path / "pick", use_tilt_series=True
    )
    prj = _batch_project(ui)
    mole = prj.load_molecules("Mole-0")
    assert_allclose(
        mole.pos - np.array(prj.tilt_series.tomogram_center),
        jobs.particles_zyx / 10,
        atol=1e-4,
    )

    # particles are saved back in the tomogram space of the tilt series
    path_sets = ui.batch._get_loader_paths()
    relion.save_molecules_for_import(ui, tmp_path / "import.star", path_sets)
    df = read_loop_block(tmp_path / "import_particles" / "TS_01_particles.star")
    assert_allclose(df.select(POS_CENTERED).to_numpy(), jobs.particles_zyx, atol=1e-3)
    relion.save_molecules_for_extract(ui, tmp_path / "extract.star", path_sets)
    df = read_loop_block(tmp_path / "extract.star")
    assert df["rlnTomoName"].to_list() == ["TS_01"] * 2
    assert_allclose(df.select(POS_CENTERED).to_numpy(), jobs.particles_zyx, atol=1e-3)

    # join RELION particles to the batch of tilt-series projects
    relion.open_relion_job(
        ui,
        jobs.pick / "job.star",
        project_root=tmp_path / "pick-joined",
        use_tilt_series=True,
        join_molecules_by_id=True,
        path_sets=path_sets,
    )
    mole_joined = _batch_project(ui).load_molecules("Mole-0")
    assert_allclose(mole_joined.pos, mole.pos, atol=1e-4)

    # save and load molecules of the tilt series in the viewer
    ui.batch.constructor.projects[0].send_to_viewer()
    relion.save_molecules(ui, tmp_path / "mole.star", list(ui.mole_layers))
    relion.load_molecules(ui, tmp_path / "mole.star")
    assert_allclose(ui.mole_layers.last().molecules.pos, mole.pos, atol=1e-4)

    # preview
    _preview_open_relion_job(
        ui, jobs.align / "job.star", reference_scale=TS_SCALE * 2, tomo_height=42.0
    )
    _preview_open_relion_job(ui, jobs.tomo / "job.star", use_tilt_series=True)
