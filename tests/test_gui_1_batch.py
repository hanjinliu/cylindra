from cylindra.widgets import CylindraMainWidget
import tempfile
from pathlib import Path
import pytest

from .utils import pytest_group
from ._const import TEST_DIR, PROJECT_DIR_13PF, PROJECT_DIR_14PF

from magicclass.testing import FunctionGuiTester
from cylindra.widgets.sta import MaskChoice


def _load(ui: CylindraMainWidget, name="Loader"):
    ui.batch.construct_loader(
        paths=[
            (
                TEST_DIR / "13pf_MT.tif",
                ["Mole-0.csv", "Mole-1.csv"],
                PROJECT_DIR_13PF,
            )
        ],
        predicate="pl.col('nth') < 3",
        name=name,
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

    ui.batch.construct_loader(
        paths=[
            (
                TEST_DIR / "13pf_MT.tif",
                ["Mole-0.csv", "Mole-1.csv"],
                PROJECT_DIR_13PF,
            ),
            (
                TEST_DIR / "14pf_MT.tif",
                ["Mole-0.csv", "Mole-1.csv"],
                PROJECT_DIR_13PF,
            ),
        ],
        predicate="pl.col('npf_glob') == 13",
        name="Loader2",
    )

    loader = ui.batch.sta.get_loader("Loader2")
    assert loader.features["pf-id"].max() == 12

    ui.batch.constructor.select_all_projects()
    ui.batch.constructor.select_molecules_by_pattern("Mole-0*")
    ui.batch.constructor.select_projects_by_pattern("*13*")


def test_view(ui: CylindraMainWidget):
    ui.batch.constructor.add_projects_glob(TEST_DIR / "test*" / "project.json")
    tester = FunctionGuiTester(ui.batch.constructor.add_projects_glob)
    tester.update_parameters(pattern=TEST_DIR / "test*" / "project.json")
    tester.click_preview()
    ui.batch.constructor.clear_projects()
    ui.batch.constructor.add_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    ui.batch.constructor.view_components()
    ui.batch.constructor.view_molecules()
    ui.batch.constructor.view_filtered_molecules()
    ui.batch.constructor.view_selected_components().close()
    ui.batch.close()


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
    template_path = TEST_DIR / "beta-tubulin.mrc"
    ui.batch.sta.params.template_path.value = template_path
    ui.batch.sta.params.mask_choice = MaskChoice.blur_template
    ui.batch.sta.show_template()
    ui.batch.sta.show_template_original()
    ui.batch.sta.show_mask()
    ui.batch.show_macro()
    ui.batch.show_native_macro()
    ui.batch.sta.remove_loader("Loader")


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
    ui.batch.sta.calculate_fsc("Loader", mask_params=None, size=6.0)
    assert len(ui.sta.sub_viewer.layers) == 1
    ui.batch.sta.split_loader("Loader", by="pf-id", delete_old=True)
    ui.batch.sta.show_loader_info()


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
