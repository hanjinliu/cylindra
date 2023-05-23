from cylindra.widgets import CylindraMainWidget
import tempfile
from pathlib import Path
import pytest
from .utils import pytest_group
from ._const import TEST_DIR, PROJECT_DIR_13PF


def _load(ui: CylindraMainWidget):
    ui.batch.construct_loader(
        paths=[
            (
                TEST_DIR / "13pf_MT.tif",
                [
                    PROJECT_DIR_13PF / "Mono-0.csv",
                    PROJECT_DIR_13PF / "Mono-1.csv",
                ],
            )
        ],
        predicate="pl.col('nth') < 3",
        name="Loader",
    )


def test_project_io(ui: CylindraMainWidget):
    _load(ui)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = root / "test"
        ui.batch.save_batch_project(path)
        assert len(ui.batch._loaders) == 1
        ui.batch.load_batch_project(path)
        assert len(ui.batch._loaders) == 1


@pytest_group("batch.average")
@pytest.mark.parametrize("binsize", [1, 2])
def test_average(ui: CylindraMainWidget, binsize: int):
    _load(ui)
    ui.batch.sta.average_all("Loader", size=6.0, bin_size=binsize)


@pytest_group("batch.align")
@pytest.mark.parametrize("binsize", [1, 2])
def test_align(ui: CylindraMainWidget, binsize: int):
    _load(ui)
    ui.batch.sta.align_all(
        "Loader",
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(2.0, 2.0),
        bin_size=binsize,
    )


def test_calculate_fsc(ui: CylindraMainWidget):
    _load(ui)
    ui.batch.sta.calculate_fsc("Loader", mask_params=None, size=6.0)


@pytest_group("batch.classify")
@pytest.mark.parametrize("binsize", [1, 2])
def test_classify_pca(ui: CylindraMainWidget, binsize: int):
    _load(ui)
    ui.batch.sta.classify_pca(
        "Loader", mask_params=None, size=6.0, interpolation=1, bin_size=binsize
    )


def test_filter(ui: CylindraMainWidget):
    _load(ui)
    ui.batch.sta.filter_loader("Loader", "pl.col('pf-id') == 1")
    loader = ui.batch.sta.get_loader("Loader-Filt")
    assert all(loader.features["pf-id"] == 1)
