from cylindra.widgets import CylindraMainWidget
import tempfile
from pathlib import Path
import pytest
from .utils import pytest_group
from ._const import TEST_DIR, PROJECT_DIR_13PF, PROJECT_DIR_14PF


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


def test_view(ui: CylindraMainWidget):
    ui.batch.constructor.add_children_glob(TEST_DIR / "test*" / "project.json")
    ui.batch.constructor.clear_children()
    ui.batch.constructor.add_children([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    ui.batch.constructor.view_components()
    ui.batch.constructor.view_molecules()
    ui.batch.constructor.view_filtered_molecules()
    ui.batch.constructor.view_localprops()


@pytest_group("batch.average")
@pytest.mark.parametrize("binsize", [1, 2])
def test_average(ui: CylindraMainWidget, binsize: int):
    _load(ui)
    ui.batch.sta.average_all("Loader", size=6.0, bin_size=binsize)
    assert len(ui.sta.sub_viewer.layers) == 1
    ui.batch.sta.average_groups(
        "Loader", size=6.0, bin_size=binsize, by="pl.col('pf-id')"
    )
    assert len(ui.sta.sub_viewer.layers) == 2
    ui.batch.show_macro()
    ui.batch.show_native_macro()


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
    assert len(ui.sta.sub_viewer.layers) == 1


@pytest_group("batch.classify")
@pytest.mark.parametrize("binsize", [1, 2])
def test_classify_pca(ui: CylindraMainWidget, binsize: int):
    _load(ui)
    ui.batch.sta.classify_pca(
        "Loader", mask_params=None, size=6.0, interpolation=1, bin_size=binsize
    )
    assert len(ui.sta.sub_viewer.layers) == 1


def test_filter(ui: CylindraMainWidget):
    _load(ui)
    ui.batch.sta.filter_loader("Loader", "pl.col('pf-id') == 1")
    loader = ui.batch.sta.get_loader("Loader-Filt")
    assert all(loader.features["pf-id"] == 1)
