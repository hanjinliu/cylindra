import json
from pathlib import Path
from typing import NamedTuple

import acryo
import impy as ip
import numpy as np
import polars as pl
import pytest
from acryo import Molecules, SubtomogramLoader
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from cylindra.components import CylTomogram
from cylindra.const import PropertyNames as H
from cylindra.widgets import CylindraMainWidget

from ._const import TEST_DIR

pytestmark = pytest.mark.skipif(
    not hasattr(acryo, "BackProjectionLoader"),
    reason="BackProjectionLoader is not available",
)
mrcfile = pytest.importorskip("mrcfile")

SCALE = 1.052
TOMO_HAND = 1  # handedness of the reference tomogram of the fixture
COORDS_13PF = np.array([[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])


class TiltSeriesFiles(NamedTuple):
    json: Path
    reference: Path
    offset: np.ndarray  # coordinates of the tomogram from the tilt series - original


def _project(vol: np.ndarray, degrees: np.ndarray, width: int) -> np.ndarray:
    """Projections of the volume around the y axis, as in acryo's TiltSeries."""
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
def tilt_series_files(tmp_path_factory: pytest.TempPathFactory) -> TiltSeriesFiles:
    """Tilt series simulated from the test tomogram and its bin-2 reconstruction."""
    from acryo import BackProjectionLoader
    from acryo.projection import TiltSeries

    tmpdir = tmp_path_factory.mktemp("tilt_series")
    vol = np.asarray(ip.imread(TEST_DIR / "13pf_MT.tif"), dtype=np.float32)
    vol -= vol.mean()
    degrees = np.arange(-60.0, 61.0, 3.0)
    # dark density on a light background, as in cryo-EM images
    with mrcfile.new(tmpdir / "ts.mrc") as f:
        f.set_data(-_project(vol, degrees, width=104))
    ts = TiltSeries(
        tilt_angles=degrees,
        tilt_axis_angle=0.0,
        scale=SCALE,
        image_path=tmpdir / "ts.mrc",
    )
    # relative path to the tilt images
    json_path = tmpdir / "ts.json"
    json_path.write_text(ts.replace(image_path=Path("ts.mrc")).dump_json())

    ref_shape = tuple(s // 2 for s in vol.shape)
    loader = BackProjectionLoader(ts, binsize=2)
    assert loader.scale == pytest.approx(SCALE * 2)
    with mrcfile.new(tmpdir / "ref.mrc") as f:
        f.set_data(loader.reconstruct_tomogram(ref_shape))  # light background
        f.voxel_size = SCALE * 2 * 10
    # Voxel `i` of the reference tomogram is at `(i - shape // 2) * scale`, and its
    # voxel `shape // 2` is at the bin-1 voxel `shape // 2 * 2` of the tomogram (see
    # `default_tomogram_center`).
    center = np.array(ref_shape) // 2 * 2 * SCALE
    offset = center - np.array(vol.shape) // 2 * SCALE
    return TiltSeriesFiles(json_path, tmpdir / "ref.mrc", offset)


def _corr(a, b) -> float:
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)))


def _cached_binsizes(tomo: CylTomogram) -> set[int]:
    """Bin sizes of which the preprocessed tilt images are cached."""
    # keys are (bin size, selection of the tilts)
    return {key[0] for key in tomo.tilt_series.loader._images.cache.images.keys()}


def test_from_tilt_series(tilt_series_files: TiltSeriesFiles):
    from acryo import BackProjectionLoader

    tomo = CylTomogram.from_tilt_series(
        tilt_series_files.json,
        ip.imread(tilt_series_files.reference),
        tomo_hand=TOMO_HAND,
    )
    src = tomo.tilt_series
    assert src is not None
    assert src.loader.tomo_hand == TOMO_HAND
    assert src.tilt_series.image_path == tilt_series_files.json.parent / "ts.mrc"
    assert tomo.scale == pytest.approx(SCALE)
    assert tomo.shape == (36, 206, 94)
    assert tomo.tilt == {"kind": "y", "range": (-60.0, 60.0)}
    np.testing.assert_allclose(
        src.tomogram_center,
        tilt_series_files.offset + np.array([18, 103, 47]) * SCALE,
    )
    # nothing is preprocessed until needed
    assert _cached_binsizes(tomo) == set()

    # the reference is only for visualization
    assert src.reference_binsize == 2
    assert src.reference.scale.x == pytest.approx(SCALE * 2)
    assert tomo.multiscaled == []
    assert tomo.available_binsizes() == [1]
    img2 = tomo.get_multiscale_or_original(2)
    assert isinstance(img2, ip.LazyImgArray)
    assert img2.shape == (18, 103, 47)
    assert _cached_binsizes(tomo) == set()

    # bin-1 image is lazy and is reconstructed from the tilt series
    img = tomo.get_multiscale_or_original(1)
    assert isinstance(img, ip.LazyImgArray)
    assert img.shape == tomo.shape
    assert img.scale.x == pytest.approx(SCALE)
    region = img[10:30, 60:140, 30:70].compute()
    assert region.shape == (20, 80, 40)
    assert len(src._chunks._data) > 0
    assert src.is_cache_enabled
    assert _cached_binsizes(tomo) == {1}

    mole = Molecules(np.array([[18.0, 100.0, 40.0], [19.0, 150.0, 35.0]]))
    loader = tomo.get_subtomogram_loader(mole, output_shape=(10.0, 10.0, 10.0))
    assert isinstance(loader, BackProjectionLoader)
    assert loader.scale == pytest.approx(SCALE)
    assert loader.asnumpy().shape == (2, 10, 10, 10)
    assert _cached_binsizes(tomo) == {1}
    loader = tomo.get_subtomogram_loader(mole, (10.0, 10.0, 10.0), binsize=2)
    assert isinstance(loader, BackProjectionLoader)
    assert loader.scale == pytest.approx(SCALE * 2)
    assert _cached_binsizes(tomo) == {1}
    assert loader.asnumpy().shape == (2, 5, 5, 5)
    assert _cached_binsizes(tomo) == {1, 2}

    # The reference has light background. The voxels of the binned image are shifted
    # by (binsize - 1) / 2 bin-1 voxels from those of the reference.
    assert _corr(src.reconstruct(2), src.reference) < -0.85

    with pytest.raises(ValueError):
        tomo.invert()
    with pytest.raises(ValueError):
        tomo.update_scale(1.0)

    # multiscale images are reconstructed from the tilt series
    img4 = tomo.add_multiscale(4)
    assert isinstance(img4, ip.ImgArray)
    assert img4.shape == (9, 51, 23)
    assert tomo.available_binsizes() == [1, 4]
    loader = tomo.get_subtomogram_loader(mole, (10.0, 10.0, 10.0), binsize=4)
    assert isinstance(loader, SubtomogramLoader)

    src.release_cache()
    assert not src.is_cache_enabled
    assert len(src._chunks._data) == 0
    assert _cached_binsizes(tomo) == set()
    # cache is enabled again if needed
    tomo.get_subtomogram_loader(mole, (10.0, 10.0, 10.0)).asnumpy()
    assert src.is_cache_enabled
    assert _cached_binsizes(tomo) == {1}
    src.release_cache()


def test_from_tilt_series_reference(tilt_series_files: TiltSeriesFiles):
    ref = ip.imread(tilt_series_files.reference)
    tomo = CylTomogram.from_tilt_series(tilt_series_files.json, ref, binsize=[1, 3])
    assert tomo.available_binsizes() == [1, 3]
    assert tomo.tilt_series.reference is not ref

    ref_wrong_scale = ref.set_scale(zyx=SCALE * 2.5)
    with pytest.raises(ValueError):
        CylTomogram.from_tilt_series(tilt_series_files.json, ref_wrong_scale)
    tomo = CylTomogram.from_tilt_series(
        tilt_series_files.json, ref_wrong_scale, reference_binsize=2
    )
    assert tomo.tilt_series.reference.scale.x == pytest.approx(SCALE * 2)
    assert ref_wrong_scale.scale.x == pytest.approx(SCALE * 2.5)  # not updated


def test_relative_reference_path(
    tilt_series_files: TiltSeriesFiles, monkeypatch: pytest.MonkeyPatch
):
    # the path of the reference is saved in projects, so it must be absolute
    monkeypatch.chdir(tilt_series_files.reference.parent)
    ref = ip.imread(tilt_series_files.reference.name)
    tomo = CylTomogram.from_tilt_series(tilt_series_files.json, ref)
    assert tomo.metadata["reference"] == tilt_series_files.reference.resolve()


@pytest.mark.parametrize("binsize", [2, 3])
@pytest.mark.parametrize("tomo_hand", [1, -1])
def test_reference_aligned_with_tilt_series(
    tilt_series_files: TiltSeriesFiles, binsize: int, tomo_hand: int
):
    from acryo import BackProjectionLoader

    from cylindra.components.tomogram._tilt_series import read_tilt_series

    # reference created as in `prep_reference_tomogram`
    ts = read_tilt_series(tilt_series_files.json)
    loader = BackProjectionLoader(ts, tomo_hand=tomo_hand)
    ref = loader.binning(binsize).reconstruct_tomogram((36 // binsize, -1, -1))
    ref = ip.asarray(ref, axes="zyx").set_scale(zyx=SCALE * binsize, unit="nm")
    tomo = CylTomogram.from_tilt_series(
        tilt_series_files.json, ref, tomo_hand=tomo_hand
    )
    src = tomo.tilt_series
    assert src.reconstruct(binsize).shape == ref.shape
    # Voxel `i` of the reference is at the bin-1 voxel `i * binsize` of the tomogram
    # (while that of the binned images is at `i * binsize + (binsize - 1) / 2`). The
    # subtomogram of an odd size at the tomogram center must be at the same position
    # as the center of the reference.
    size = tuple(n - 1 + n % 2 for n in ref.shape)
    mole = Molecules(src.tomogram_center[np.newaxis])
    cube = src.get_loader(mole, binsize=binsize, output_shape=size).asnumpy()[0]
    sl = tuple(
        slice(n // 2 - s // 2, n // 2 + s // 2 + 1)
        for n, s in zip(ref.shape, size, strict=True)
    )
    ref_center = -np.asarray(ref, dtype=np.float32)[sl]  # light background
    shift, *_ = phase_cross_correlation(ref_center, cube, upsample_factor=20)
    assert np.abs(shift).max() <= 0.05
    assert _corr(ref_center, cube) > 0.98
    src.release_cache()


def test_analysis_with_tilt_series(tilt_series_files: TiltSeriesFiles):
    tomo = CylTomogram.from_tilt_series(
        tilt_series_files.json,
        ip.imread(tilt_series_files.reference),
        tomo_hand=TOMO_HAND,
    )
    tomo.add_spline(COORDS_13PF + tilt_series_files.offset)
    tomo.make_anchors(0, max_interval=32.6)
    tomo.fit(0, binsize=1)
    tomo.refine(0, binsize=1)
    tomo.make_anchors(0, interval=32.6)
    tomo.measure_radius(0, binsize=1)
    tomo.local_cft_params(i=0, binsize=1)
    tomo.global_cft_params(i=0, binsize=1)
    spl = tomo.splines[0]
    assert 4.08 < spl.props.get_glob(H.spacing) < 4.12
    assert spl.props.get_glob(H.npf) == 13
    assert all(spl.props.loc[H.npf] == 13)
    assert _cached_binsizes(tomo) == {1}
    tomo.tilt_series.release_cache()


def test_open_tilt_series(
    ui: CylindraMainWidget,
    tilt_series_files: TiltSeriesFiles,
    tmp_path: Path,
):
    ref = np.asarray(ip.imread(tilt_series_files.reference))
    ui.open_tilt_series_with_reference(
        tilt_series_files.json,
        scale=SCALE,
        reference_path=tilt_series_files.reference,
        tomo_hand=TOMO_HAND,
        bin_size=[1],
        filter=None,
        invert_reference=True,
    )
    assert str(ui.macro[-1]).startswith("ui.open_tilt_series_with_reference(")
    assert not ui._need_save
    tomo = ui.tomogram
    src = tomo.tilt_series
    assert src is not None
    assert ui._current_binsize == 2
    assert [b for _, b in ui._get_available_binsize()] == [1]
    # the reference is shown as a binned image
    np.testing.assert_allclose(
        ui._reserved_layers.image.translate, [tomo.multiscale_translation(2)] * 3
    )
    np.testing.assert_allclose(ui._reserved_layers.image_data, -ref)
    assert ui._reserved_layers.ref_inverted
    assert _cached_binsizes(tomo) == set()

    ui.register_path(COORDS_13PF + tilt_series_files.offset)
    # projections in the spline control are sampled from the reference
    assert _cached_binsizes(tomo) == set()
    ui._runner.run(interval=32.6, bin_size=1)
    assert _cached_binsizes(tomo) == {1}
    spl = ui.splines[0]
    assert 4.08 < spl.props.get_glob(H.spacing) < 4.12
    assert spl.props.get_glob(H.npf) == 13

    ui.map_monomers(splines=[0])
    ui.AnalysisMenu.open_sta_widget()
    ui.sta.average_all("Mole-0", size=12.0, bin_size=1)
    assert ui.sub_viewer.layers[-1].data.shape == (11, 11, 11)
    ui.sta.align_all(
        layers=["Mole-0"],
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(1, 1),
        max_shifts=(1.0, 1.0, 1.0),
        bin_size=1,
    )
    assert "Mole-0-ALN1" in ui.mole_layers
    assert _cached_binsizes(tomo) == {1}

    # save and load the project
    old_spline = ui.splines[0].copy()
    ui.save_project(tmp_path / "project")
    ui.load_project(tmp_path / "project", filter=None)
    tomo = ui.tomogram
    assert tomo.tilt_series is not None
    assert tomo.tilt_series is not src
    # the old tilt series is released
    assert not src.is_cache_enabled
    assert len(src._chunks._data) == 0
    assert tomo.tilt_series.binsizes == [1]
    assert tomo.tilt_series.loader.tomo_hand == TOMO_HAND
    assert _cached_binsizes(tomo) == set()
    np.testing.assert_allclose(tomo.tilt_series.tomogram_center, src.tomogram_center)
    np.testing.assert_allclose(ui._reserved_layers.image_data, -ref)
    assert ui._reserved_layers.ref_inverted
    assert ui.splines[0].close_to(old_spline)
    assert "Mole-0" in ui.mole_layers

    # the coarse tomogram can be shown again after showing a multi-scale image
    assert [b for _, b in ui._get_displayable_binsize()] == [2]
    ui.add_multiscale(4)
    assert ui._current_binsize == 4
    assert [b for _, b in ui._get_displayable_binsize()] == [2, 4]
    ui.set_multiscale(2)
    assert ui._current_binsize == 2
    np.testing.assert_allclose(ui._reserved_layers.image_data, -ref)
    with pytest.raises(ValueError):
        ui.set_multiscale(1)  # not shown in the viewer

    # opening another tomogram releases the tilt series
    src = tomo.tilt_series
    ui.sta.average_all("Mole-0", size=12.0, bin_size=1)
    assert src.is_cache_enabled
    ui.open_image(TEST_DIR / "13pf_MT.tif", scale=SCALE, bin_size=[2], filter=None)
    assert ui.tomogram.tilt_series is None
    assert not src.is_cache_enabled

    # a project of a missing tilt series is opened with a dummy tomogram
    project_json = tmp_path / "project" / "project.json"
    js = json.loads(project_json.read_text())
    js["tilt_series"]["reference"] = str(tmp_path / "missing.mrc")
    project_json.write_text(json.dumps(js))
    ui.load_project(tmp_path / "project", filter=None)
    assert ui.tomogram.is_dummy
    assert ui.splines[0].close_to(old_spline)


def test_refine_tilt_series(
    ui: CylindraMainWidget,
    tilt_series_files: TiltSeriesFiles,
    tmp_path: Path,
):
    ui.open_tilt_series_with_reference(
        tilt_series_files.json,
        scale=SCALE,
        reference_path=tilt_series_files.reference,
        tomo_hand=TOMO_HAND,
        bin_size=[1],
        filter=None,
        invert_reference=True,
    )
    tomo = ui.tomogram
    src = tomo.tilt_series
    true_ts = src.tilt_series
    ref_before = np.asarray(src.reference)
    mole = Molecules.from_csv(TEST_DIR / "test_project_13pf" / "Mole-0.csv")
    layer = ui.add_molecules(mole.translate(tilt_series_files.offset), name="Mole-0")
    ui.sta.align_all(
        layers=[layer],
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(1, 1),
        max_shifts=(1.0, 1.0, 1.0),
        bin_size=1,
    )
    with pytest.raises(ValueError):  # no layer
        ui.sta.refine_tilt_series(layers=[], size=12.0, bin_size=1)
    # The averages of the current tilt series are the references. The molecules are
    # not exactly at their centers, which shifts the tilt images systematically. The
    # shifts are estimated from the true tilt series.
    ui.sta.refine_tilt_series(
        layers=["Mole-0-ALN1"], size=12.0, bin_size=1, update_coarse_tomogram=False
    )
    assert str(ui.macro[-1]).startswith(
        "ui.sta.refine_tilt_series(layers=['Mole-0-ALN1'], size=12.0,"
    )
    assert src.has_unsaved_tilt_series
    assert not src.has_unsaved_reference
    assert src.reference is not None
    np.testing.assert_allclose(src.reference, ref_before)
    expected = src.tilt_series.sample_translations

    rng = np.random.default_rng(0)
    shifts = rng.uniform(-1.0, 1.0, size=(true_ts.ntilts, 2))
    perturbed = true_ts.replace(
        sample_translations=true_ts.sample_translations + shifts
    )
    assert tomo.add_multiscale(4) is not None
    assert tomo.update_tilt_series(perturbed) == [4]  # stale images are removed
    assert tomo.multiscaled == []
    ui.sta.refine_tilt_series(layers=["Mole-0-ALN1"], size=12.0, bin_size=1)
    refined = src.tilt_series
    # the periodic lattice of microtubules is ambiguous in a few tilts
    error = np.abs(refined.sample_translations - expected).max(axis=1)  # nm
    assert np.median(error) < 0.15
    assert np.mean(error < 0.25) >= 0.85

    # the coarse tomogram is reconstructed from the refined tilt series, and shown
    # with the same inversion
    new_ref = src.reference
    assert new_ref.shape == ref_before.shape
    assert new_ref.scale.x == pytest.approx(SCALE * 2)
    assert src.has_unsaved_reference
    assert _corr(new_ref, ref_before) > 0.95
    np.testing.assert_allclose(ui._reserved_layers.image_data, -np.asarray(new_ref))
    assert ui._reserved_layers.ref_inverted

    # both the tilt series and the coarse tomogram are restored by undo
    ui.macro.undo()
    np.testing.assert_allclose(
        src.tilt_series.sample_translations, perturbed.sample_translations
    )
    np.testing.assert_allclose(src.reference, ref_before)
    np.testing.assert_allclose(ui._reserved_layers.image_data, -ref_before)
    ui.macro.redo()
    assert src.tilt_series is refined
    assert src.reference is new_ref

    # a new coarse tomogram is shown with the same filter
    ui.filter_reference_image("Lowpass", 2.0)
    filtered = np.asarray(ui._reserved_layers.image_data)
    ui.sta._show_coarse_tomogram(new_ref)
    np.testing.assert_allclose(ui._reserved_layers.image_data, filtered, atol=1e-5)
    assert ui._reserved_layers.ref_filter == ("Lowpass", 2.0)
    assert ui._reserved_layers.ref_inverted

    # the refined tilt series and the new reference are saved in the project
    for name in ["project", "project.tar", "project"]:
        ui.save_project(tmp_path / name)
        ui.load_project(tmp_path / name, filter=None)
        src = ui.tomogram.tilt_series
        np.testing.assert_allclose(
            src.tilt_series.sample_translations, refined.sample_translations
        )
        np.testing.assert_allclose(src.reference, new_ref, atol=1e-6)
        assert src.has_unsaved_tilt_series and src.has_unsaved_reference
        assert ui._reserved_layers.ref_inverted
    assert (tmp_path / "project" / "tilt_series.json").exists()
    assert (tmp_path / "project" / "coarse_tomogram.mrc").exists()
    src.release_cache()


def _project_files(path: Path) -> set[str]:
    """Names of the files in a project directory, tar or zip file."""
    import tarfile
    import zipfile

    if path.suffix == ".tar":
        with tarfile.open(path) as tar:
            return set(tar.getnames())
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return set(zf.namelist())
    return {p.name for p in path.iterdir()}


def test_save_project_with_tilt_series(
    ui: CylindraMainWidget,
    tilt_series_files: TiltSeriesFiles,
    tmp_path: Path,
):
    from cylindra.project import CylindraProject
    from cylindra.project._single import COARSE_TOMOGRAM_FILE, TILT_SERIES_FILE

    ui.open_tilt_series_with_reference(
        tilt_series_files.json,
        scale=SCALE,
        reference_path=tilt_series_files.reference,
        tomo_hand=TOMO_HAND,
        bin_size=[1],
        filter=None,
        invert_reference=True,
    )
    ui.register_path(COORDS_13PF + tilt_series_files.offset)
    old_spline = ui.splines[0].copy()
    src = ui.tomogram.tilt_series
    # the tilt series and the coarse tomogram that are not saved in any file
    src.set_reference(src.reconstruct_reference())
    src.mark_unsaved(tilt_series=True, reference=False)
    expected = np.asarray(src.reference)
    with mrcfile.mmap(tilt_series_files.json.parent / "ts.mrc", mode="r") as mrc:
        tilt_series_size = f"{mrc.data.nbytes / 1024**2:.2f} MB"

    for name in ["project", "project.tar", "project.zip"]:
        for save_coarse in [True, False]:
            path = tmp_path / f"coarse-{save_coarse}" / name
            path.parent.mkdir(exist_ok=True)
            ui.save_project(path, save_coarse_tomogram=save_coarse)
            files = _project_files(path)
            assert TILT_SERIES_FILE in files
            assert (COARSE_TOMOGRAM_FILE in files) == save_coarse

            # the coarse tomogram is reconstructed if not saved
            ui.load_project(path, filter=None)
            src = ui.tomogram.tilt_series
            assert src is not None, f"{name}, {save_coarse=}"
            assert src.has_unsaved_tilt_series and src.has_unsaved_reference
            np.testing.assert_allclose(src.reference, expected, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                ui._reserved_layers.image_data, -expected, rtol=1e-4, atol=1e-4
            )
            assert ui.splines[0].close_to(old_spline)

            # files in the project are shown as such, not as the temporary files
            info = ui.GeneralInfo.image_info.value
            if path.suffix in (".tar", ".zip"):
                prefix = f"{path.as_posix()}:"
            else:
                prefix = f"{path.resolve().as_posix()}/"
            assert f"{prefix}{TILT_SERIES_FILE}" in info
            assert (f"{prefix}{COARSE_TOMOGRAM_FILE}" in info) == save_coarse
            assert ("Reference:" in info) == save_coarse  # not if reconstructed
            assert f"Data size: {tilt_series_size} (tilt series)" in info

            # also without GUI
            tomo = CylindraProject.from_file(path).load_tomogram()
            assert tomo.tilt_series is not None
            assert tomo.tilt_series.reference.shape == expected.shape
            assert len(tomo.splines) == 1
            tomo.tilt_series.release_cache()

    # overwriting keeps whether the coarse tomogram is saved
    ui.overwrite_project()
    assert COARSE_TOMOGRAM_FILE not in _project_files(path)


def test_batch_with_tilt_series(
    ui: CylindraMainWidget,
    tilt_series_files: TiltSeriesFiles,
    tmp_path: Path,
):
    from acryo import BackProjectionLoader

    ui.open_tilt_series_with_reference(
        tilt_series_files.json,
        scale=SCALE,
        reference_path=tilt_series_files.reference,
        tomo_hand=TOMO_HAND,
        bin_size=[1, 2],
        filter=None,
        invert_reference=True,
    )
    mole = Molecules.from_csv(TEST_DIR / "test_project_13pf" / "Mole-0.csv")
    ui.add_molecules(mole.translate(tilt_series_files.offset), name="Mole-0")
    # projects of the tilt series in a directory and in a zip file
    paths = [tmp_path / "project-0", tmp_path / "project-1.zip"]
    for path in paths:
        ui.save_project(path)

    ui.batch.constructor.add_projects(paths, clear=True)
    assert len(ui.batch.constructor.projects) == 2

    # open a project in the viewer
    ui.batch.constructor.projects[1].send_to_viewer()
    assert ui.tomogram.tilt_series is not None
    assert "Mole-0" in ui.mole_layers

    # subtomograms are reconstructed from the tilt series
    ui.batch.construct_loader(
        ui.batch._get_loader_paths(), predicate="pl.col('nth') < 3", name="Loader"
    )
    loader = ui.batch.sta.get_loader("Loader")
    nmole = mole.filter(pl.col("nth") < 3).count()
    assert loader.molecules.count() == 2 * nmole
    sub_loaders = list(loader.loaders)
    assert len(sub_loaders) == 2
    assert all(isinstance(ldr, BackProjectionLoader) for ldr in sub_loaders)
    np.testing.assert_allclose(
        sub_loaders[0].tomogram_center, ui.tomogram.tilt_series.tomogram_center
    )
    # subtomograms are reconstructed at the binned scale, so the molecules are not
    # translated by binning
    binned = loader.binning(2)
    np.testing.assert_allclose(binned.molecules.pos, loader.molecules.pos)
    ui.batch.sta.average_all("Loader", size=6.0, bin_size=1)
    for binsize in [1, 2]:
        ui.batch.sta.align_all(
            "Loader",
            template_path=TEST_DIR / "beta-tubulin.mrc",
            mask_params=(2.0, 2.0),
            max_shifts=(1.0, 1.0, 1.0),
            bin_size=binsize,
        )
        aligned = ui.batch.sta.get_loader(ui.batch.loader_infos[-1].name)
        assert aligned.molecules.count() == 2 * nmole

    # loader of a single project
    single = ui.batch.constructor.projects[0].get_loader(order=1)
    assert isinstance(single, BackProjectionLoader)
    assert single.molecules.count() == mole.count()


def test_open_tilt_series_dialog(
    ui: CylindraMainWidget,
    tilt_series_files: TiltSeriesFiles,
    tmp_path: Path,
):
    import shutil

    from acryo.projection import TiltSeries

    # the default settings are written next to the tilt series
    shutil.copy(tilt_series_files.json.parent / "ts.mrc", tmp_path / "ts.mrc")
    json_path = shutil.copy(tilt_series_files.json, tmp_path / "ts.json")
    dialog = ui.FileMenu.open_tilt_series_loader()
    dialog.path = json_path
    dialog.scan_header_or_defaults()
    assert dialog.create_reference  # no default settings yet

    # the coarse tomogram is reconstructed and saved
    ui.open_tilt_series(
        json_path,
        scale=SCALE,
        reference_bin_size=2,
        tomo_hand=TOMO_HAND,
        tomo_height=18,
        bin_size=[1],
        filter=None,
        save_coarse_tomogram=True,
    )
    assert str(ui.macro[-1]).startswith("ui.open_tilt_series(")
    src = ui.tomogram.tilt_series
    assert src.reference_binsize == 2
    assert src.reference.shape[0] == 18
    ref_path = tmp_path / "ts_rec_bin2.mrc"
    assert ref_path.exists()
    assert not src.has_unsaved_reference
    # the coarse tomogram is shown with inverted contrast
    np.testing.assert_allclose(ui._reserved_layers.image_data, -np.asarray(src.reference))

    # the default settings are used next time
    dialog.scan_header_or_defaults()
    assert not dialog.create_reference
    assert Path(dialog.reference_path) == ref_path
    assert dialog.tomo_hand == TOMO_HAND
    assert dialog.tomo_height == 18
    assert dialog.reference_bin_size == 2
    assert float(dialog.scale.scale_value) == pytest.approx(SCALE, abs=1e-4)

    # the coarse tomogram is not saved
    ui.open_tilt_series(
        json_path,
        scale=SCALE,
        reference_bin_size=3,
        tomo_hand=TOMO_HAND,
        tomo_height=12,
        bin_size=[1],
        filter=None,
        save_coarse_tomogram=False,
    )
    src = ui.tomogram.tilt_series
    assert src.reference_binsize == 3
    assert src.has_unsaved_reference
    assert not (tmp_path / "ts_rec_bin3.mrc").exists()

    # size of the tilt series is unknown if the images are missing
    ts = TiltSeries.from_json(json_path.read_text())
    tomo = CylTomogram.from_tilt_series(
        ts.replace(image_path=tmp_path / "missing.mrc", scale=SCALE),
        ip.imread(ref_path),
    )
    ui.GeneralInfo._refer_tomogram(tomo)
    assert "Data size: Unknown (tilt series)" in ui.GeneralInfo.image_info.value


def test_tilt_series_source_errors(tilt_series_files: TiltSeriesFiles, tmp_path: Path):
    from cylindra.components.tomogram import _tilt_series as _ts

    tomo = CylTomogram.from_tilt_series(
        tilt_series_files.json,
        ip.imread(tilt_series_files.reference),
        tomo_hand=TOMO_HAND,
    )
    src = tomo.tilt_series
    assert "TiltSeriesSource" in repr(src)
    assert src.shape == (36, 206, 94)
    for kwargs in [{"chunk_size": 63}, {"chunk_margin": -1}, {"binsizes": [0]}]:
        with pytest.raises(ValueError):
            _ts.TiltSeriesSource(src.loader, src.shape, **kwargs)
    with pytest.raises(ValueError):
        src.image(0)

    # without a reference
    no_ref = _ts.TiltSeriesSource(src.loader, src.shape)
    assert no_ref.reference_binsize is None
    with pytest.raises(ValueError):
        no_ref.reconstruct_reference()

    # the reference must be of the same shape and bin size
    with pytest.raises(ValueError):
        src.set_reference(src.reference[:-1])
    other_bin = src.reference.copy()
    other_bin.set_scale(zyx=SCALE * 3, unit="nm")
    with pytest.raises(ValueError):
        src.set_reference(other_bin)
    with pytest.raises(ValueError):
        _ts.reconstruct_coarse_tomogram(src.tilt_series, 2, None, TOMO_HAND)
    with pytest.raises(ValueError):
        _ts.reconstruct_coarse_tomogram(
            src.tilt_series.replace(scale=None), 2, src.reference.shape, TOMO_HAND
        )

    # size of the tilt images
    ts = src.tilt_series
    assert src.nbytes == ts.ntilts * 206 * 104 * 4
    per_tilt = ts.replace(image_path=(ts.image_path,) * ts.ntilts)
    assert _ts.tilt_images_nbytes(per_tilt) == src.nbytes
    assert _ts.tilt_images_nbytes(ts.replace(image_path=None)) is None
    missing = ts.replace(image_path=tmp_path / "missing.mrc")
    assert _ts.tilt_images_nbytes(missing) is None

    # file types of the tilt series
    with pytest.raises(ValueError):
        _ts.read_tilt_series(tmp_path / "tilt_series.txt")
    with pytest.raises(ValueError):
        _ts.read_tilt_series(tmp_path / "tilt_series.edf")  # needs the pixel size
    # the paths of the tilt images are relative to the json file
    (tmp_path / "per_tilt.json").write_text(
        ts.replace(image_path=("ts.mrc",) * ts.ntilts).dump_json()
    )
    read = _ts.read_tilt_series(tmp_path / "per_tilt.json")
    assert read.image_path == (tmp_path / "ts.mrc",) * ts.ntilts
    src.release_cache()


def test_chunk_cache():
    from cylindra.components.tomogram._tilt_series import _ChunkCache

    arr = np.zeros(1, dtype=np.float32)  # 4 bytes
    cache = _ChunkCache(max_bytes=12)
    for key in "abc":
        cache.put(key, arr)
    assert cache.get("a") is arr  # "a" is used recently
    cache.put("d", arr)  # "b" is removed
    assert cache.get("b") is None
    assert cache.get("a") is arr
    cache.put("a", np.zeros(2, dtype=np.float32))  # "c" is removed
    assert cache.get("c") is None
    assert cache.get("a").size == 2
    cache.put("large", np.zeros(4, dtype=np.float32))  # too large to be cached
    assert cache.get("large") is None
    cache.clear()
    assert cache.get("a") is None


def test_infer_tilt_series_scale(tmp_path: Path):
    from cylindra.widgets.subwidgets.misc import (
        _format_nbytes,
        _infer_scale_for_tilt_series,
        _resolve_path_str,
    )

    image_path = tmp_path / "TS_01.mrc"
    with mrcfile.new(image_path) as f:
        f.set_data(np.zeros((3, 8, 8), dtype=np.float32))
        f.voxel_size = 2.5
    # IMOD
    (tmp_path / "TS_01.edf").write_text("")
    assert _infer_scale_for_tilt_series(tmp_path / "TS_01.edf") == pytest.approx(0.25)
    # RELION 5
    star_path = tmp_path / "TS_01.star"
    star_path.write_text(
        "data_\n\nloop_\n_rlnMicrographName #1\n_rlnTomoNominalStageTiltAngle #2\n"
        f"{image_path.as_posix()} 3.0\n{image_path.as_posix()} -3.0\n"
    )
    assert _infer_scale_for_tilt_series(star_path) == pytest.approx(0.25)
    assert _infer_scale_for_tilt_series(tmp_path / "tilt_series.json") is None

    assert _format_nbytes(3 * 1024**3) == "3.00 GB"
    assert _format_nbytes(3 * 1024**2) == "3.00 MB"
    assert _resolve_path_str(f"{image_path};{tmp_path / 'b.mrc'}") == image_path
    assert _resolve_path_str(tmp_path / "TS_*.mrc") == image_path
