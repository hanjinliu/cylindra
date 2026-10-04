from pathlib import Path

import impy as ip
import numpy as np
import pytest
from magicclass import testing as mcls_testing
from magicgui.application import use_app

from cylindra.components.landscape import Landscape
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.sta import MaskChoice

from ._const import PROJECT_DIR_13PF, PROJECT_DIR_14PF, TEST_DIR
from .utils import pytest_group


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


def _template_size_nm() -> float:
    img = ip.imread(TEST_DIR / "beta-tubulin.mrc")
    return img.shape.x * img.scale.x


def _assert_box_size(shape: tuple[int, ...], scale: float, size: float):
    """Assert that the subtomogram box size in nm does not depend on binning."""
    box = np.array(shape[-3:]) * scale
    assert np.all(np.abs(box - size) < scale), f"box size {box} nm != {size} nm"


def test_click_buttons(ui: CylindraMainWidget):
    mcls_testing.check_function_gui_buildable(ui.batch)


def test_tooltip(ui: CylindraMainWidget):
    mcls_testing.check_tooltip(ui.batch)


def test_project_io(ui: CylindraMainWidget, tmpdir):
    _load(ui)
    root = Path(tmpdir)
    path = root / "test"
    ui.batch.save_batch_project(path)
    assert len(ui.batch.loader_infos) == 1
    ui.batch.load_batch_project(path)
    assert len(ui.batch.loader_infos) == 1

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
                PROJECT_DIR_14PF,
            ),
        ],
        predicate="pl.col('npf_glob') == 13",
        name="Loader2",
    )

    loader = ui.batch.sta.get_loader("Loader2")
    assert loader.features["pf-id"].max() == 12  # only 13-PF

    # test loader property
    assert ui.batch.loader_infos[0].name == "Loader"
    assert ui.batch.loader_infos[1].name == "Loader2"
    ui.batch.loader_infos["Loader"]
    del ui.batch.loader_infos["Loader2"]

    # use absolute paths
    ui.batch.construct_loader(
        paths=[
            (
                TEST_DIR / "13pf_MT.tif",
                [PROJECT_DIR_13PF / "Mole-0.csv", PROJECT_DIR_13PF / "Mole-1.csv"],
            ),
            (
                TEST_DIR / "14pf_MT.tif",
                [PROJECT_DIR_14PF / "Mole-0.csv", PROJECT_DIR_14PF / "Mole-1.csv"],
            ),
        ],
        name="Loader_abs",
    )

    ui.batch.constructor.clear_projects()
    ui.batch.new_projects(
        [TEST_DIR / "13pf_MT.tif", TEST_DIR / "14pf_MT.tif"],
        save_root=root / "new_projects",
        strip_prefix="1",
        strip_suffix="_MT",
    )
    assert len(ui.batch.loader_infos) == 2
    assert (p13dir := root.joinpath("new_projects", "3pf")).exists()
    ui.load_project(p13dir)
    ui.register_path([[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]])
    assert len(ui.batch.constructor.projects[0].splines) == 0
    ui.save_project(p13dir)
    assert len(ui.batch.constructor.projects[0].splines) == 1
    ui.batch.new_projects(
        [TEST_DIR / "13pf_MT.tif", TEST_DIR / "14pf_MT.tif"],
        save_root=root / "new_projects",
        ref_paths=[TEST_DIR / "13pf_MT.tif", TEST_DIR / "14pf_MT.tif"],
    )


def test_view(ui: CylindraMainWidget):
    ui.batch.constructor.add_projects(TEST_DIR / "test*" / "project.json")
    tester = mcls_testing.FunctionGuiTester(ui.batch.constructor.add_projects)
    tester.update_parameters(pattern=[TEST_DIR / "test*" / "project.json"])
    tester.click_preview()

    ui.batch.constructor.select_all_projects()
    ui.batch.constructor.select_molecules_by_pattern("Mole-0*", op="or")
    ui.batch.constructor.select_projects_by_pattern("*13*", op="and")
    ui.batch.constructor.select_molecules_by_globalprops("col('npf') == 13", op="ignore")

    ui.batch.constructor.clear_projects()
    ui.batch.constructor.add_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    ui.batch.constructor.view_components()
    ui.batch.constructor.view_molecules()
    ui.batch.constructor.view_filtered_molecules()
    # retry with filter
    ui.batch.constructor.filter_expression.value = "pl.col('npf_glob') == 13"
    ui.batch.constructor.view_filtered_molecules()

    ui.batch.constructor.view_selected_components().close()
    ui.batch.close()
    use_app().process_events()


@pytest_group("batch.average")
@pytest.mark.parametrize("binsize", [1, 2])
def test_average(ui: CylindraMainWidget, binsize: int):
    _load(ui)
    ui.batch.sta.average_all("Loader", size=6.0, bin_size=binsize)
    assert len(ui.sta.sub_viewer.layers) == 1
    layer = ui.sta.sub_viewer.layers[-1]
    _assert_box_size(layer.data.shape, layer.scale[-1], 6.0)
    ui.batch.sta.average_groups(
        "Loader", size=6.0, bin_size=binsize, by="pl.col('pf-id')"
    )
    assert len(ui.sta.sub_viewer.layers) == 2
    layer = ui.sta.sub_viewer.layers[-1]
    _assert_box_size(layer.data.shape, layer.scale[-1], 6.0)
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
    # the template must be rescaled to the binned scale
    loader = ui.batch.sta.get_loader("Loader-ALN1")
    _assert_box_size(loader.output_shape, loader.scale, _template_size_nm())
    ui.batch.sta.calculate_fsc("Loader", mask_params=None, size=6.0)
    assert len(ui.sta.sub_viewer.layers) == 1
    ui.batch.sta.align_all_template_free(
        "Loader",
        mask_params={"kind": "spherical", "radius": 2.3, "sigma": 0.7},
        size=12.0,
        bin_size=binsize,
        max_num_iters=3,
    )
    assert len(ui.sta.sub_viewer.layers) == 2
    loader = ui.batch.sta.get_loader("Loader-ALN2")
    _assert_box_size(loader.output_shape, loader.scale, 12.0)
    ui.batch.sta.split_loader("Loader", by="pf-id", delete_old=True)
    ui.batch.sta.show_loader_info()


@pytest_group("batch.align")
@pytest.mark.parametrize("binsize", [1, 2])
def test_align_rma(
    ui: CylindraMainWidget, binsize: int, monkeypatch: pytest.MonkeyPatch
):
    # Molecules returned by RMA are in the coordinates of the loader used to build
    # the landscape, so record the scale of these loaders.
    landscape_scales = list[float]()
    _from_loader = Landscape.from_loader.__func__

    def _from_loader_and_record(cls, loader, *args, **kwargs):
        landscape_scales.append(loader.scale)
        return _from_loader(cls, loader, *args, **kwargs)

    monkeypatch.setattr(Landscape, "from_loader", classmethod(_from_loader_and_record))

    # RMA needs the source splines, which are looked up from the batch projects.
    ui.batch.constructor.add_projects([PROJECT_DIR_13PF])
    _load(ui)
    template_path = TEST_DIR / "beta-tubulin.mrc"
    ui.batch.sta.align_all(
        "Loader",
        template_path=template_path,
        mask_params=(2.0, 2.0),
        bin_size=binsize,
    )
    # max_shifts must be larger than the landscape precision of the binned loader.
    ui.batch.sta.align_all_rma(
        "Loader",
        template_path=template_path,
        mask_params=(2.0, 2.0),
        max_shifts=(1.2, 1.2, 1.2),
        bin_size=binsize,
        upsample_factor=2,
        num_trials=2,
        temperature_time_const=0.2,
    )
    ui.batch.sta.align_all_rma_template_free(
        "Loader",
        mask_params=(2.0, 2.0),
        size=8.0,
        max_shifts=(1.2, 1.2, 1.2),
        bin_size=binsize,
        upsample_factor=2,
        temperature_time_const=0.2,
        max_num_iters=3,
    )

    # RMA outputs must be binned loaders, as the output of `align_all` is.
    loader_aln = ui.batch.sta.get_loader("Loader-ALN1")
    shapes_aln = {k: img.shape for k, img in loader_aln.images.items()}
    for name in ["Loader-ALN2", "Loader-ALN3"]:
        loader_rma = ui.batch.sta.get_loader(name)
        assert loader_rma.scale == pytest.approx(loader_aln.scale)
        assert {k: img.shape for k, img in loader_rma.images.items()} == shapes_aln
    assert landscape_scales
    assert landscape_scales == pytest.approx([loader_aln.scale] * len(landscape_scales))

    # templates must be rescaled to the binned scale
    for name, size in [
        ("Loader-ALN1", _template_size_nm()),
        ("Loader-ALN2", _template_size_nm()),
        ("Loader-ALN3", 8.0),
    ]:
        loader = ui.batch.sta.get_loader(name)
        _assert_box_size(loader.output_shape, loader.scale, size)


def test_filter(ui: CylindraMainWidget):
    _load(ui)
    ui.batch.sta.filter_loader("Loader", "pl.col('pf-id') == 1")
    loader = ui.batch.sta.get_loader("Loader-Filt")
    assert all(loader.features["pf-id"] == 1)
