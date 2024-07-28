import tempfile
from itertools import product
from pathlib import Path

import impy as ip
import napari
import numpy as np
import polars as pl
import pytest
from acryo import Molecules
from magicclass import get_function_gui
from magicclass import testing as mcls_testing
from magicclass.utils import thread_worker
from magicgui.application import use_app
from numpy.testing import assert_allclose

from cylindra import _config, cylmeasure, view_project
from cylindra._config import get_config
from cylindra.const import (
    MoleculesHeader as Mole,
)
from cylindra.const import PropertyNames as H
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.sta import MaskChoice, TemplateChoice

from ._const import PROJECT_DIR_13PF, PROJECT_DIR_14PF, TEST_DIR
from .utils import ExceptionGroup, pytest_group

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5]]
coords = {13: coords_13pf, 14: coords_14pf}


def assert_canvas(ui: CylindraMainWidget, isnone: list[bool]):
    use_app().process_events()
    for i in range(3):
        if isnone[i]:
            assert ui.SplineControl.canvas[i].image is None, f"{i}-th canvas"
        else:
            assert ui.SplineControl.canvas[i].image is not None, f"{i}-th canvas"


def assert_molecule_equal(mole0: Molecules, mole1: Molecules):
    assert_allclose(mole0.pos, mole1.pos, atol=1e-4, rtol=1e-4)
    assert_allclose(mole0.x, mole1.x, atol=1e-4, rtol=1e-4)
    assert_allclose(mole0.y, mole1.y, atol=1e-4, rtol=1e-4)
    assert_allclose(mole0.z, mole1.z, atol=1e-4, rtol=1e-4)


def assert_orientation(ui: CylindraMainWidget, ori: str):
    assert ui.splines[ui.SplineControl.num].orientation == ori
    assert ui.GlobalProperties.params.params2.orientation_.txt == ori

    spec = ui._reserved_layers.prof.features["spline-id"] == ui.SplineControl.num
    arr = ui._reserved_layers.prof.symbol[spec]
    if ori == "MinusToPlus":
        assert (arr[0], arr[-1]) == ("hbar", "cross")
    elif ori == "PlusToMinus":
        assert (arr[0], arr[-1]) == ("cross", "hbar")


def test_click_buttons(ui: CylindraMainWidget):
    mcls_testing.check_function_gui_buildable(ui)
    mcls_testing.check_function_gui_buildable(ui.sta)
    mcls_testing.check_function_gui_buildable(ui.simulator)


def test_tooltip(ui: CylindraMainWidget):
    mcls_testing.check_tooltip(ui)
    mcls_testing.check_tooltip(ui.sta)
    mcls_testing.check_tooltip(ui.simulator)


def test_plugin(make_napari_viewer):
    from cylindra.core import start_as_plugin

    make_napari_viewer()
    start_as_plugin(run=False)


@pytest.mark.parametrize(
    "save_path,npf", [(PROJECT_DIR_13PF, 13), (PROJECT_DIR_14PF, 14)]
)
def test_io(ui: CylindraMainWidget, save_path: Path, npf: int):
    path = TEST_DIR / f"{npf}pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=[1])
    ui.add_multiscale(2)
    ui.set_multiscale(1)
    ui.register_path(coords=coords[npf])
    ui.register_path(coords=coords[npf][::-1])
    ui._runner.run(interval=24.0)
    ui.infer_polarity()
    ui.map_monomers(splines=[0, 1])
    ui.measure_local_radius(splines=[0, 1])

    # Save project
    old_splines = ui.tomogram.splines.copy()
    old_molecules = list(ui.mole_layers.iter_molecules())
    ui.save_project(save_path)
    ui.overwrite_project()
    ui.load_project(save_path, filter="DoG")
    assert len(ui.macro.undo_stack["undo"]) == 0
    new_splines = ui.tomogram.splines
    new_molecules = list(ui.mole_layers.iter_molecules())
    assert old_splines[0].close_to(new_splines[0])
    assert old_splines[1].close_to(new_splines[1])
    for mol0, mol1 in zip(old_molecules, new_molecules, strict=True):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt["range"] == (-60, 60)

    # try .tar file
    ui.save_project(save_path / "test_tar.tar")
    ui.overwrite_project()
    ui.load_project(save_path / "test_tar.tar", filter="DoG")
    assert len(ui.macro.undo_stack["undo"]) == 0
    new_splines = ui.tomogram.splines
    new_molecules = list(ui.mole_layers.iter_molecules())
    assert old_splines[0].close_to(new_splines[0])
    assert old_splines[1].close_to(new_splines[1])
    for mol0, mol1 in zip(old_molecules, new_molecules, strict=True):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt["range"] == (-60, 60)

    # try .zip file
    ui.save_project(save_path / "test_zip.zip")
    ui.overwrite_project()
    ui.load_project(save_path / "test_zip.zip", filter="DoG")
    assert len(ui.macro.undo_stack["undo"]) == 0
    new_splines = ui.tomogram.splines
    new_molecules = list(ui.mole_layers.iter_molecules())
    assert old_splines[0].close_to(new_splines[0])
    assert old_splines[1].close_to(new_splines[1])
    for mol0, mol1 in zip(old_molecules, new_molecules, strict=True):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt["range"] == (-60, 60)

    # operations on the local properties and projections
    ui.SplineControl.auto_contrast()
    ui.LocalProperties.edit_plots(["spacing", "twist"])
    ui.LocalProperties.edit_plots(["spacing", "twist", "rise"])

    ui.SplinesMenu.Show.show_splines()
    ui.SplinesMenu.Show.show_splines_as_meshes()
    ui.load_splines(save_path / "spline-0.json")
    ui.set_source_spline(ui.mole_layers["Mole-0"], 0)
    ui.invert_image()


def test_io_with_different_data(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    params = [
        {"local_props": False, "global_props": False},
        {"local_props": False, "global_props": True},
        {"local_props": True, "global_props": False},
        {"local_props": True, "global_props": True, "map_monomers": True},
    ]
    exc_group = ExceptionGroup()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for param in params:
            with exc_group.merging():
                ui.open_image(
                    path=path, scale=1.052, tilt_range=(-60, 60), bin_size=[1, 2]
                )
                ui.register_path(coords=coords_13pf)
                ui._runner.run(splines=[0], **param)
                ui.save_project(tmpdir)
                ui.load_project(tmpdir, filter="DoG")
    exc_group.raise_exceptions()
    ui.SplineControl.footer.highlight_subvolume = True
    ui.SplineControl.pos = 1
    ui.SplineControl.footer.highlight_subvolume = False

    ui.load_project(PROJECT_DIR_14PF, filter=None, read_image=False)
    ui.mole_layers.get("Mole-0")
    ui.mole_layers.get("Mole-100", None)
    list(ui.mole_layers.iter())
    ui.mole_layers.first()
    ui.mole_layers.count()
    assert "Mole-0" in ui.mole_layers


def test_picking_splines(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=[1, 2])
    ui._reserved_layers.work.add(coords_13pf[0])
    ui._reserved_layers.work.add(coords_13pf[1])
    ui.Toolbar.pick_next()
    ui.register_path()
    assert len(ui.tomogram.splines) == 1


def test_spline_deletion(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 1.0
    ui.clear_current()
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 0.0
    ui.register_path(coords=coords_13pf[::-1])
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 1.0
    ui.delete_spline(0)
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 0.0
    ui.Toolbar.undo()
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 1.0
    ui.Toolbar.undo()
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 0.0
    ui.Toolbar.redo()
    ui.Toolbar.redo()
    assert ui._reserved_layers.prof.features["spline-id"].values[0] == 0.0
    assert ui._reserved_layers.prof.features["spline-id"].values[-1] == 0.0


def test_serialize(ui: CylindraMainWidget):
    from magicclass.serialize import deserialize, serialize

    ui.load_project(PROJECT_DIR_13PF, filter=None)
    d = serialize(ui)
    deserialize(ui, d)


def test_workflow_with_many_input(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    exc_group = ExceptionGroup()
    with exc_group.merging():
        ui._runner.run([0], max_shift=-1)  # no fit
    with exc_group.merging():
        ui._runner.run([0], n_refine=0)  # no refine
    with exc_group.merging():
        ui._runner.run([0], max_shift=-1, n_refine=0)  # no fit/refine
    with exc_group.merging():
        ui._runner.run([0], local_props=False)  # no localprops
    with exc_group.merging():
        ui._runner.run([0], global_props=False, map_monomers=False)  # no globalprops

    exc_group.raise_exceptions()

    # toggle many widgets to check if they are working
    ui._runner.fit = False
    ui._runner.fit = True
    ui._runner.local_props = False
    ui._runner.local_props = True


def test_workflow_undo_redo(ui: CylindraMainWidget):
    path = TEST_DIR / "14pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_14pf)
    ui._runner.run([0])
    n_undo = len(ui.macro.undo_stack["undo"])
    for _ in range(n_undo):
        ui.macro.undo()
    for _ in range(n_undo):
        ui.macro.redo()


def test_config(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    ui.SplinesMenu.Config.update_default_config(npf_range=(13, 15))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with _config.patch_config_dir(tmpdir):
            ui.SplinesMenu.Config.save_default_config(tmpdir)
            ui.SplinesMenu.Config.load_default_config(tmpdir)
    ui.SplinesMenu.Config.view_config_presets()


def test_reanalysis(ui: CylindraMainWidget):
    ui.load_project_for_reanalysis(PROJECT_DIR_14PF)
    assert len(ui.macro.undo_stack["undo"]) == 0
    assert ui.tomogram.splines[0].orientation == "none"
    assert len(ui.macro) > 1
    assert str(ui.macro[-1]).startswith("ui.register_path(")
    ui.measure_radius()
    assert ui.splines[0].radius is not None
    assert len(ui.macro.undo_stack["undo"]) > 0
    ui.reanalyze_image()
    ui.reanalyze_image_config_updated()
    assert len(ui.macro.undo_stack["undo"]) == 0
    assert ui.splines[0].radius is None
    assert len(ui.macro) > 1
    assert str(ui.macro[-1]).startswith("ui.register_path(")


def test_map_molecules(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    assert ui.get_loader("Mole-0").molecules is ui.mole_layers["Mole-0"].molecules
    ui.map_monomers_with_extensions(0, {0: (1, 1), 1: (-1, -1)})
    ui.map_along_pf(0, molecule_interval=4.0)
    ui.map_along_spline(0, molecule_interval=4.0)
    ui.macro.undo()
    ui.macro.undo()
    ui.macro.undo()
    ui.macro.redo()
    ui.macro.redo()
    ui.macro.redo()
    # test expression input
    ui.map_along_pf(0, molecule_interval=pl.col("spacing") * 2)
    ui.map_along_spline(0, molecule_interval=pl.col("spacing") * 2)
    # this also tests coercing names
    ui.map_along_spline(0, molecule_interval=pl.col("spacing"))
    ui.clear_all()
    ui.register_path(coords_14pf)
    ui.map_along_spline(
        0, molecule_interval=4.08, rotate_molecules=False, orientation=None
    )  # test mapping without any measurement.


def test_napari_operations(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    name = ui.mole_layers.last().name
    ui.parent_viewer.layers[name].name = "new-name"
    assert str(ui.macro[-1]) == f"ui.parent_viewer.layers[{name!r}].name = 'new-name'"
    assert ui.mole_layers.last().name == "new-name"
    ui.macro.undo()
    assert not str(ui.macro[-1]).endswith("'new-name'")
    assert ui.mole_layers.last().name == name

    del ui.parent_viewer.layers[-1]
    assert str(ui.macro[-1]) == f"del ui.parent_viewer.layers[{name!r}]"
    ui.macro.undo()
    assert not str(ui.macro[-1]).startswith("del ")


def test_load_macro(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir) / "test_macro.py"
        fp.write_text("print(0)")
        ui.OthersMenu.Macro.load_macro_file(fp)


def test_spline_control(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.filter_reference_image()
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])

    # check canvas is updated correctly
    ui.add_anchors(interval=15.0)
    ui.add_anchors(interval=15.0, how="equal")
    ui.macro.undo()
    assert_canvas(ui, [True, True, True])
    with thread_worker.blocking_mode():
        ui.sample_subtomograms()
        assert_canvas(ui, [False, False, True])
        ui.SplineControl.num = 1
        assert_canvas(ui, [False, False, True])
        ui.SplineControl.num = 0
        assert_canvas(ui, [False, False, True])
        ui.SplineControl.pos = 1
        assert_canvas(ui, [False, False, True])
        ui.SplineControl.pos = 0
        assert_canvas(ui, [False, False, True])

        ui.fit_splines_by_centroid(splines=[0])
        ui._runner.run(interval=32.6)

        # check results
        for spl in ui.tomogram.splines:
            spacing_mean = spl.props.loc[H.spacing].mean()
            spacing_glob = spl.props.get_glob(H.spacing)
            # GDP-bound microtubule has lattice spacing in this range
            assert 4.08 < spacing_glob < 4.11
            assert spacing_glob == pytest.approx(spacing_mean, abs=0.02)
            assert all(spl.props.loc[H.npf] == 13)
            assert all(spl.props.loc[H.rise] > 8.3)

        # check canvas again
        assert_canvas(ui, [False, False, False])
        ui.SplineControl.num = 1
        assert_canvas(ui, [False, False, False])
        ui.SplineControl.num = 0
        assert_canvas(ui, [False, False, False])
        ui.SplineControl.pos = 1
        assert_canvas(ui, [False, False, False])
        ui.SplineControl.pos = 0
        assert_canvas(ui, [False, False, False])

        # check orientations
        # switch spline 0 and 1 and check if orientation is correctly set
        ui.SplineControl.num = 0
        ui.set_spline_props(spline=0, orientation="PlusToMinus")
        assert_orientation(ui, "PlusToMinus")
        ui.macro.undo()
        ui.macro.redo()
        assert_orientation(ui, "PlusToMinus")
        ui.SplineControl.num = 1
        ui.set_spline_props(spline=1, orientation="MinusToPlus")
        assert_orientation(ui, "MinusToPlus")
        ui.SplineControl.num = 0
        assert_orientation(ui, "PlusToMinus")
        ui.SplineControl.num = 1
        assert_orientation(ui, "MinusToPlus")
        assert_canvas(ui, [False, False, False])

        ui.SplinesMenu.Show.show_localprops()

        # Check align polarity.
        # Only spline 0 will get updated.
        ui.align_to_polarity(orientation="MinusToPlus")
        ui.SplineControl.num = 0
        ui.SplineControl.pos = 1
        assert_orientation(ui, "MinusToPlus")
        assert (
            ui.LocalProperties.params.spacing.txt
            == f" {ui.splines[ui.SplineControl.num].props.loc[H.spacing][1]:.2f} nm"
        )
        assert (
            ui.GlobalProperties.params.params1.spacing.txt
            == f" {ui.splines[ui.SplineControl.num].props.get_glob(H.spacing):.2f} nm"
        )

        ui.SplineControl.num = 1
        assert ui.SplineControl.pos == 1
        assert_orientation(ui, "MinusToPlus")
        assert (
            ui.LocalProperties.params.spacing.txt
            == f" {ui.splines[ui.SplineControl.num].props.loc[H.spacing][1]:.2f} nm"
        )
        assert (
            ui.GlobalProperties.params.params1.spacing.txt
            == f" {ui.splines[ui.SplineControl.num].props.get_glob(H.spacing):.2f} nm"
        )

        assert_canvas(ui, [False, False, False])
        ui.copy_spline(0)
        ui.copy_spline_new_config(0, npf_range=(4, 10))

        ui.clear_all()

        assert ui.LocalProperties.params.spacing.txt == " -- nm"
        assert ui.GlobalProperties.params.params1.spacing.txt == " -- nm"

    ui.OthersMenu.cylindra_info()
    ui.OthersMenu.open_command_palette()
    ui.LocalProperties.copy_screenshot()
    ui.SplineControl.copy_screenshot()
    with tempfile.TemporaryDirectory() as tmpdir:
        ui.SplineControl.save_screenshot(Path(tmpdir) / "img.png")
        ui.LocalProperties.save_screenshot(Path(tmpdir) / "img.png")
    ui.SplineControl.log_screenshot()
    ui.LocalProperties.log_screenshot()
    cfg = get_config()
    ui.OthersMenu.configure_cylindra(use_gpu=cfg.use_gpu)
    cfg.to_user_dir()


def test_preview(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    layer = ui.mole_layers["Mole-0"]
    tester = mcls_testing.FunctionGuiTester(ui.translate_molecules)
    tester.gui.layers.value = [ui.mole_layers["Mole-0"]]
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer + 1
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.rotate_molecules)
    tester.gui.layers.value = [ui.mole_layers["Mole-0"]]
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer + 1
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.split_molecules)
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.filter_molecules)
    nlayer = len(ui.parent_viewer.layers)
    tester.update_parameters(predicate="pl.col('nth') < 4")
    tester.click_preview()
    is_true = layer.face_color[:, 3] > 0.9
    assert np.all(layer.features["nth"][is_true] < 4)
    tester.click_preview()
    is_true = layer.face_color[:, 3] > 0.9
    assert is_true.sum() == layer.data.shape[0]

    tester = mcls_testing.FunctionGuiTester(ui.paint_molecules)
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.map_along_pf)
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer + 1
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.map_monomers_with_extensions)
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer + 1
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.load_project_for_reanalysis)
    tester.update_parameters(path=PROJECT_DIR_13PF)
    tester.click_preview()

    tester = mcls_testing.FunctionGuiTester(ui.load_molecules)
    tester.update_parameters(
        paths=[PROJECT_DIR_13PF / "Mole-0.csv", PROJECT_DIR_13PF / "Mole-1.csv"]
    )
    tester.click_preview()

    tester = mcls_testing.FunctionGuiTester(ui.load_project)
    tester.update_parameters(path=PROJECT_DIR_13PF)
    tester.click_preview()

    tester = mcls_testing.FunctionGuiTester(ui.clip_spline)
    tester.click_preview()
    tester.update_parameters(lengths=(3, 1))  # clip
    tester.update_parameters(lengths=(-3, -2))  # extend
    tester.click_preview()

    tester = mcls_testing.FunctionGuiTester(ui.convolve_feature)
    tester.click_preview()
    tester = mcls_testing.FunctionGuiTester(ui.binarize_feature)
    tester.click_preview()
    tester = mcls_testing.FunctionGuiTester(ui.sta.seam_search_manually)
    tester.click_preview()
    ui.binarize_feature(ui.mole_layers["Mole-0"], "nth", threshold=3)
    tester = mcls_testing.FunctionGuiTester(ui.label_feature_clusters)
    tester.click_preview()

    # two preview layers
    mcls_testing.FunctionGuiTester(ui.translate_molecules).click_preview()
    mcls_testing.FunctionGuiTester(ui.map_along_pf).click_preview()


def test_sub_widgets(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    ui.ImageMenu.open_slicer()
    with thread_worker.blocking_mode():
        ui.spline_slicer.refresh_widget_state()
        ui.spline_slicer.show_what = "CFT"
        ui.spline_slicer._update_canvas()
        ui.spline_slicer.show_what = "R-projection"
        ui.spline_slicer._update_canvas()
        ui.spline_slicer.show_what = "Y-projection"
        ui.spline_slicer._update_canvas()
        ui.spline_slicer.show_what = "Filtered-R-projection"
        ui.spline_slicer._update_canvas()
        ui.spline_slicer.params.binsize = 2
        ui.spline_slicer._next_pos()
        ui.spline_slicer._prev_pos()
        ui.spline_slicer.measure_cft_here()
        ui.spline_slicer._show_overlay_text("some text")

        # spline clipper
        len_old = ui.splines[0].length()
        ui.SplinesMenu.open_spline_clipper()
        ui.spline_clipper.clip_length = 1
        ui.spline_clipper.clip_here()
        assert ui.splines[0].length() == pytest.approx(len_old - 1, abs=0.01)
        ui.spline_clipper.the_other_side()
        ui.spline_clipper.clip_length = 1.4
        ui.spline_clipper.clip_here()
        assert ui.splines[0].length() == pytest.approx(len_old - 2.4, abs=0.02)

        # spectra inspector
        ui.local_cft_analysis(interval=25)
        ui.AnalysisMenu.open_spectra_inspector()
        ui.spectra_inspector.log_scale = True
        ui.spectra_inspector.log_scale = False
        ui.spectra_inspector.select_axial_peak()
        ui.spectra_inspector._click_at((20, 10))
        ui.spectra_inspector.select_angular_peak()
        ui.spectra_inspector._click_at((10, 20))
        ui.spectra_inspector.upsample_spectrum()
        ui.spectra_inspector._click_at((15, 25))
        ui.spectra_inspector.peak_viewer.show_what = "Local-CFT"
        ui.spectra_inspector._click_at((5, 5))

        # file iterator
        ui.FileMenu.open_file_iterator()
        ui._file_iterator.set_pattern(f"{TEST_DIR.as_posix()}/*.tif")
        ui._file_iterator.last_file()
        ui._file_iterator.first_file()
        ui._file_iterator.next_file()
        ui._file_iterator.prev_file()
        ui._file_iterator.open_image(ui._file_iterator.path)
        ui._file_iterator.preview_all().close()
        ui._file_iterator.set_pattern(f"{TEST_DIR.as_posix()}/*/project.json")
        ui._file_iterator.view_local_props()
        ui._file_iterator.send_to_batch_analyzer()
        ui._file_iterator.load_project(ui._file_iterator.path)
        ui._file_iterator.load_project_for_reanalysis(ui._file_iterator.path)


@pytest.mark.parametrize("bin_size", [1, 2])
def test_sta(ui: CylindraMainWidget, bin_size: int):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    ui.AnalysisMenu.open_sta_widget()
    ui.sta.average_all("Mole-0", size=12.0, bin_size=bin_size)
    ui.sta.average_all("Mole-0", size=12.0, bin_size=bin_size)  # check coerce name
    for method in ["steps", "first", "last", "random"]:
        ui.sta.average_subset(
            "Mole-0",
            size=12.0,
            method=method,
            bin_size=bin_size,
        )
    ui.sta.average_groups("Mole-0", size=12.0, bin_size=bin_size, by=pl.col("pf-id"))
    ui.sta.average_filtered(
        "Mole-0", size=12.0, bin_size=bin_size, predicate=pl.col("pf-id") < 5
    )
    ui.sta.calculate_fsc(
        "Mole-0",
        mask_params=None,
        size=8.0,
        seed=0,
        interpolation=1,
    )

    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        molepath = dirpath / "monomers.txt"
        ui.save_molecules(layer="Mole-0", save_path=molepath)
        ui.save_spline(0, dirpath / "spline-x.json")
        mole = ui.mole_layers["Mole-0"].molecules
        ui.load_molecules(molepath)
        mole_read = ui.mole_layers["monomers"].molecules
        assert_molecule_equal(mole, mole_read)

        ui.sta.save_last_average(dirpath)

    template_path = TEST_DIR / "beta-tubulin.mrc"
    ui.sta.params.template_path.value = template_path
    ui.sta.params.mask_choice = MaskChoice.from_file
    ui.sta.params.mask_choice = MaskChoice.blur_template
    ui.sta.show_template()
    ui.sta.show_template_original()
    ui.sta.show_mask()
    ui.sta.get_template(TEST_DIR / "beta-tubulin.mrc")
    ui.sta.get_mask((0.3, 0.5), template_path=TEST_DIR / "beta-tubulin.mrc")
    ui.sta.align_averaged(
        layers=["Mole-0"],
        template_path=template_path,
        mask_params=(1, 1),
        bin_size=bin_size,
    )
    assert "offset_axial" in ui.splines[0].props.glob.columns
    ui.macro.undo()
    assert "offset_axial" not in ui.splines[0].props.glob.columns
    ui.macro.redo()
    assert "offset_axial" in ui.splines[0].props.glob.columns
    ui.sta.split_and_average(layers=["Mole-0"], size=12.0, bin_size=bin_size)
    ui.sta.align_all(
        layers=["Mole-0"],
        template_path=template_path,
        mask_params=(1, 1),
        max_shifts=(1.0, 1.1, 1.0),
        rotations=((0.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
        interpolation=1,
        bin_size=bin_size,
    )
    ui.macro.undo()
    ui.macro.redo()
    ui.sta.align_all_template_free(
        layers=["Mole-0"],
        mask_params=(1, 1),
        size=12.0,
        bin_size=bin_size,
    )
    ui.sta.align_all(
        layers=["Mole-0"],
        template_path=[template_path, template_path],
        mask_params=(1, 1),
        bin_size=bin_size,
    )
    ui.sta.calculate_correlation(
        layers=["Mole-0-ALN1"],
        template_path=template_path,
        column_prefix="score",
    )
    ui.sta.get_subtomograms("Mole-0", shape=(5, 5, 5), bin_size=bin_size, order=1)
    assert "score_0" in ui.mole_layers["Mole-0-ALN1"].features
    ui.sta.params.template_choice = TemplateChoice.from_files


def test_seam_search(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    ui.filter_molecules(
        ui.parent_viewer.layers["Mole-0"], predicate="pl.col('nth') < 5"
    )
    ui.sta.params.template_path.value = TEST_DIR / "beta-tubulin.mrc"
    ui.sta.params.mask_choice = MaskChoice.blur_template
    layer = ui.mole_layers.last()
    ui.sta.seam_search(
        layer=layer,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(1, 1),
    )
    with tempfile.TemporaryDirectory() as dirpath:
        path = Path(dirpath).joinpath("seam_result.csv")
        ui.sta.save_seam_search_result(layer, path)
    layer.molecules = layer.molecules.with_features(
        (pl.col("nth") * pl.col("pf-id") % 3 < 2).cast(pl.UInt8).alias("seam-label")
    )
    ui.sta.seam_search_by_feature(layer, by="seam-label")
    ui.sta.seam_search_manually(layer, 3)

    image_layer_name = ui.parent_viewer.layers[0].name
    with pytest.raises(TypeError):
        ui.sta.seam_search_manually(image_layer_name, 3)


def test_classify_pca(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    ui.filter_molecules(
        ui.parent_viewer.layers["Mole-0"], predicate="pl.col('nth') < 3"
    )
    layer = ui.mole_layers.last()
    exc_group = ExceptionGroup()
    for binsize in [1, 2]:
        with exc_group.merging():
            ui.sta.classify_pca(
                layer,
                mask_params=None,
                size=12.0,
                interpolation=1,
                bin_size=binsize,
            )
    exc_group.raise_exceptions()


# TODO: not a good test
# def test_extend_filament(ui: CylindraMainWidget):
#     ui.load_project(PROJECT_DIR_13PF, filter=None)
#     ui.sta.extend_filaments(
#         "Mole-1",
#         TEST_DIR / "beta-tubulin.mrc",
#         min_score=0.8,
#     )


def test_clip_spline(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    length_old = ui.tomogram.splines[0].length()
    ui.clip_spline(0, (10, 5))
    length_new = ui.tomogram.splines[0].length()
    assert length_old - 15 == pytest.approx(length_new, abs=1e-2)

    length_old = ui.tomogram.splines[0].length()
    ui.clip_spline(0, (3, 1))
    length_new = ui.tomogram.splines[0].length()
    assert length_old - 4 == pytest.approx(length_new, abs=1e-2)


def test_radius_methods(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    mole = ui.mole_layers["Mole-0"].molecules
    shifts = np.zeros((mole.count(), 3), dtype=np.float32)
    shifts[:, 0] = np.linspace(-1, 1, mole.count())
    mole_tr = mole.translate_internal(shifts)

    ui.add_molecules(mole_tr, "Corn", source=ui.splines[0])
    layer = ui.parent_viewer.layers["Corn"]
    ui.measure_radius_by_molecules([layer], interval=8, depth=12)
    radii = ui.splines[0].props.loc[H.radius]
    assert all(np.diff(radii) > 0)

    spl = ui.tomogram.splines[0]
    ui.set_radius([0], 11.2)
    assert spl.radius == pytest.approx(11.2, abs=1e-6)
    ui.set_radius([0], "10.4")
    assert spl.radius == pytest.approx(10.4, abs=1e-6)
    ui.set_radius([0], "pl.col('npf') * 0.9")
    assert spl.radius == pytest.approx(spl.props.get_glob("npf") * 0.9, abs=1e-6)

    # Expr returns a non-numeric value
    with pytest.raises(ValueError):
        ui.set_radius([0], "pl.col('npf').replace({14: None})")
    # Expr returns a negative value
    with pytest.raises(ValueError):
        ui.set_radius([0], "pl.col('npf').cast(pl.Float32) * -1")


def test_simulator(ui: CylindraMainWidget):
    ui.ImageMenu.open_simulator()
    ui.simulator.create_empty_image(size=(50.0, 100.0, 50.0), scale=0.5)
    ui.register_path(coords=[[25.375, 83.644, 18.063], [25.375, 23.154, 28.607]])
    tester = mcls_testing.FunctionGuiTester(ui.simulator.generate_molecules)
    tester.click_preview()
    tester.click_preview()  # has to remove the preview layer
    ui.simulator.generate_molecules(
        spline=0,
        spacing=4.1,
        twist=-0.30,
        start=3,
        npf=14,
        radius=11,
        offsets=(0.0, 0.18),
    )
    layer = ui.mole_layers.last()
    ui.simulator.generate_molecules(
        spline=0,
        spacing=4,
        twist=20,
        start=1,
        npf=2,
        radius=6,
        offsets=(0.0, 0.18),
    )
    ui.simulator._get_components()
    ui.simulator.expand(layer, by=0.1, yrange=(11, 15), arange=(0, 14), allev=True)
    ui.simulator.twist(layer, by=0.3, yrange=(11, 15), arange=(0, 14), allev=True)
    ui.simulator.dilate(layer, by=-0.5, yrange=(11, 15), arange=(0, 14), allev=True)
    ui.simulator.displace(
        layer,
        expand="pl.col('nth') * 0.01",
        twist=pl.col("nth") * 0.01,
        dilate=1.1,
    )
    fgui = get_function_gui(ui.simulator.expand)
    fgui.reset_choices()
    fgui.layer.value = ui.mole_layers.nth(1)
    assert len(ui.simulator._get_proper_molecules_layers()) == 2

    # preview
    for method in [
        ui.simulator.expand,
        ui.simulator.twist,
        ui.simulator.dilate,
        ui.simulator.displace,
    ]:
        tester = mcls_testing.FunctionGuiTester(method)
        tester.click_preview()
        tester.click_preview()

    ui.simulator.add_component(layer, TEST_DIR / "beta-tubulin.mrc")
    comp = list(ui.simulator.component_list._iter_components())[0]
    comp.remove_me()
    ui.simulator.close()


@pytest_group("simulate", maxfail=1)
def test_simulate_tomogram(ui: CylindraMainWidget):
    ui.simulator.create_image_with_straight_line(25, (40, 42, 42), scale=0.5)
    ui.simulator.generate_molecules(
        spline=0,
        spacing=4.06,
        twist=-0.31,
        start=3,
        npf=14,
        radius=11.8,
        offsets=(0.0, 0.0),
    )

    kwargs = {
        "components": [(ui.mole_layers.last().name, TEST_DIR / "beta-tubulin.mrc")],
        "tilt_range": (-60.0, 60.0),
        "n_tilt": 11,
        "interpolation": 1,
        "seed": 0,
    }
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        assert len(list(dirpath.glob("*"))) == 0
        ui.simulator.simulate_tomogram(**kwargs, nsr=[0.5, 2.0], save_dir=dirpath)
        assert len(list(dirpath.glob("*.mrc"))) == 2
        ui.load_project(dirpath / "simulation-project.tar", filter=None)
    ui.simulator.simulate_tomogram_and_open(**kwargs, nsr=1.2)
    ui.simulator.close()


@pytest_group("simulate", maxfail=1)
def test_simulate_tilt_series(ui: CylindraMainWidget):
    ui.simulator.create_image_with_straight_line(
        25, (40, 42, 42), scale=0.5, yxrotation=10
    )
    ui.simulator.generate_molecules(0, 4.06, -0.05, 3, 13, 11.8, (0.0, 0.0))
    with tempfile.TemporaryDirectory() as dirpath:
        ui.simulator.simulate_tilt_series(
            components=[(ui.mole_layers.last().name, TEST_DIR / "beta-tubulin.mrc")],
            save_dir=dirpath,
            tilt_range=(-60.0, 60.0),
            n_tilt=11,
            interpolation=1,
        )
        ui.simulator.simulate_projection(
            components=[(ui.mole_layers.last().name, TEST_DIR / "beta-tubulin.mrc")],
            save_dir=Path(dirpath) / "projection_test",
            nsr=[0.1, 1.5],
        )
        ui.simulator.simulate_tomogram_from_tilt_series(Path(dirpath) / "image.mrc")
    ui.simulator.close()


def test_project_viewer():
    pviewer = view_project(PROJECT_DIR_14PF)
    # TODO: fails due to delayed returned callback
    # pviewer.load_this_project(path=pviewer._get_project_path())
    pviewer.preview_image().close()
    pviewer.close()


def test_molecules_methods(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    layer0 = ui.mole_layers["Mole-0"]
    layer1 = ui.mole_layers["Mole-1"]
    ui.MoleculesMenu.View.show_orientation(layer0)
    ui.concatenate_molecules([layer0, layer1])
    last_layer = ui.mole_layers.last()
    assert last_layer.data.shape[0] == layer0.data.shape[0] + layer1.data.shape[0]
    ui.mole_layers.delete(include="concat")
    ui.split_molecules("Mole-0", by=Mole.pf)
    with pytest.raises(TypeError):
        ui.split_molecules(ui.parent_viewer.layers[0].name, by=Mole.pf)
    ui.interpolate_spline_properties("Mole-0", interpolation=1)
    ui.MoleculesMenu.View.render_molecules(
        layer0,
        template_path=TEST_DIR / "beta-tubulin.mrc",
    )

    color0 = layer0.face_color
    ui.paint_molecules(layer0, color_by="pf-id", limits=(0, 12))
    color1 = layer0.face_color
    assert not np.allclose(color0, color1)
    ui.macro.undo()
    assert_allclose(layer0.face_color, color0)
    ui.macro.redo()
    assert_allclose(layer0.face_color, color1)

    ui.MoleculesMenu.View.render_molecules(
        layer0,
        template_path=TEST_DIR / "beta-tubulin.mrc",
    )
    ui.rename_molecules("Mole", "XYZ", exclude="_")
    names_split = [f"Mole-0_{i}" for i in range(14)]
    assert ui.mole_layers.names() == ["XYZ-0", "XYZ-1"] + names_split
    ui.macro.undo()
    assert ui.mole_layers.names() == ["Mole-0", "Mole-1"] + names_split
    ui.macro.redo()
    assert ui.mole_layers.names() == ["XYZ-0", "XYZ-1"] + names_split
    ui.macro.undo()
    ui.delete_molecules(include="_")
    assert ui.mole_layers.names() == ["Mole-0", "Mole-1"]
    ui.macro.undo()
    assert ui.mole_layers.names() == ["Mole-0", "Mole-1"] + names_split
    ui.macro.redo()
    assert ui.mole_layers.names() == ["Mole-0", "Mole-1"]

    ui.paint_molecules("Mole-0", "nth", {0: "blue", 1: "yellow"}, (0, 10))
    ui.macro.undo()
    ui.macro.redo()
    ui.ImageMenu.show_colorbar(ui.mole_layers["Mole-0"], orientation="horizontal")
    ui.ImageMenu.show_colorbar(ui.mole_layers["Mole-0"], orientation="vertical")

    tester = mcls_testing.FunctionGuiTester(ui.rename_molecules)
    tester.update_parameters(old="Mole", new="XYZ")
    tester = mcls_testing.FunctionGuiTester(ui.delete_molecules)
    tester.update_parameters(include="-0")

    ui.register_molecules([[10, 10, 10], [10, 20, 20]])


def test_transform_molecules(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    layer = ui.mole_layers["Mole-0"]
    ui.translate_molecules("Mole-0", [3, -5, 2.2], internal=False)
    new_layer = ui.mole_layers.last()
    assert new_layer.data.shape == layer.data.shape
    assert_allclose(
        new_layer.data - layer.data,
        np.tile([3, -5, 2.2], (layer.data.shape[0], 1)),
        rtol=1e-5,
        atol=1e-6,
    )
    ui.macro.undo()
    ui.macro.redo()

    ui.translate_molecules("Mole-0", [1, 2, 3], internal=True)
    new_layer = ui.mole_layers.last()
    assert_allclose(
        np.linalg.norm(new_layer.data - layer.data, axis=1),
        np.sqrt(14),
        rtol=1e-5,
        atol=1e-6,
    )
    ui.macro.undo()
    ui.macro.redo()

    ui.rotate_molecules("Mole-0", degrees=[("z", 10), ("x", 3)])
    ui.macro.undo()
    ui.macro.redo()


def test_merge_molecules(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None)
    ui.merge_molecule_info(pos="Mole-0", rotation="Mole-1", features="Mole-0")
    assert_allclose(ui.mole_layers.last().data, ui.mole_layers["Mole-0"].data)
    ui.copy_molecules_features(
        source="Mole-0",
        destinations=["Mole-1"],
        column="position-nm",
        alias="new",
    )
    assert_allclose(
        ui.mole_layers["Mole-1"].molecules.features["new"].to_numpy(),
        ui.mole_layers["Mole-0"].molecules.features["position-nm"].to_numpy(),
    )


def test_molecule_features(ui: CylindraMainWidget):
    import polars as pl

    ui.load_project(PROJECT_DIR_14PF, filter=None)
    layer = ui.mole_layers["Mole-0"]
    ui.filter_molecules("Mole-0", predicate='pl.col("position-nm") < 9.2')
    assert ui.mole_layers.last().features["position-nm"].max() < 9.2
    # make sure predicate can also be a polars.Expr
    ui.filter_molecules("Mole-0", predicate=pl.col("position-nm") < 8)
    assert ui.mole_layers.last().features["position-nm"].max() < 8
    ui.calculate_molecule_features(
        "Mole-0",
        column_name="new",
        expression='pl.col("pf-id") < 4',
    )
    assert "new" == layer.features.columns[-1]
    ui.macro.undo()
    ui.macro.redo()
    ui.mole_layers.last().point_size = 3.5


def test_auto_align(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])

    ui._runner.run(interval=32.0)
    ui.infer_polarity()
    ui.align_to_polarity(orientation="MinusToPlus")
    assert ui.tomogram.splines[0].orientation == "MinusToPlus"
    assert ui.tomogram.splines[1].orientation == "MinusToPlus"


def test_molecules_to_spline(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    assert len(ui.tomogram.splines) == 2
    old_ori = ui.tomogram.splines[0].orientation
    layer_src = ui.mole_layers["Mole-0"]
    ui.molecules_to_spline(layers=[layer_src])
    assert len(ui.tomogram.splines) == 2
    assert ui.tomogram.splines[0].orientation == old_ori
    assert layer_src.source_component is ui.tomogram.splines[0]
    ui.translate_molecules(layer_src, [1, 1, 1], internal=True, inherit_source=True)
    layer_trans = ui.mole_layers.last()
    assert layer_trans.source_component is ui.tomogram.splines[0]
    old_spl = ui.tomogram.splines[0]
    ui.molecules_to_spline(layers=[layer_trans], update_sources=True)
    new_spl = ui.tomogram.splines[0]
    assert old_spl is not new_spl
    assert layer_trans.source_component is new_spl
    assert layer_src.source_component is new_spl
    ui.molecules_to_spline(layers=[layer_trans], update_sources=False)
    new_spl0 = ui.tomogram.splines[0]
    assert new_spl is not new_spl0
    assert layer_trans.source_component is new_spl0
    assert layer_src.source_component is new_spl
    ui.molecules_to_spline(layers=[layer_trans], inherits=["npf", "orientation"])
    new_spl1 = ui.tomogram.splines[0]
    assert ["npf", "orientation"] == sorted(new_spl1.props.glob.columns)

    ui.protofilaments_to_spline(layer=layer_trans, ids=[1, 4])


def test_calc_lattice_structures(ui: CylindraMainWidget):
    exc_group = ExceptionGroup()
    for orientation, path, invert in product(
        ["PlusToMinus", "MinusToPlus"],
        [PROJECT_DIR_13PF, PROJECT_DIR_14PF],
        [True, False],
    ):
        ui.load_project(path, filter=None, read_image=False)
        spacing = ui.tomogram.splines[0].props.get_glob(H.spacing)
        skew = ui.tomogram.splines[0].props.get_glob(H.skew)
        twist = ui.tomogram.splines[0].props.get_glob(H.twist)
        rise_angle = ui.tomogram.splines[0].props.get_glob(H.rise)
        npf = ui.tomogram.splines[0].props.get_glob(H.npf)
        if invert:
            ui.invert_spline(spline=0)
        ui.mole_layers.clear()
        ui.map_monomers(splines=[0], orientation=orientation)
        layer = ui.mole_layers.last()
        ui.calculate_lattice_structure(
            layer, ["spacing", "skew_angle", "twist", "rise_angle"]
        )
        ay_ratio: float = np.sin(np.pi / npf) * npf / np.pi
        with exc_group.merging(f"ori={orientation}, path={path.name}, inv={invert}"):
            # individial parameters must be almost equal to the global ones
            feat = layer.molecules.features
            spacing_sm = feat[Mole.spacing][npf:-npf]
            skew_sm = feat[Mole.skew][npf:-npf] / ay_ratio
            twist_sm = feat[Mole.twist][npf:-npf]
            r = Mole.rise
            feat_rise = feat.filter(pl.col(r).is_finite() & pl.col(r).is_not_nan())[r]
            rise_sm = feat_rise * ay_ratio

            assert spacing_sm.mean() == pytest.approx(spacing, abs=1e-3)
            assert skew_sm.mean() == pytest.approx(skew, abs=1e-2)
            assert twist_sm.mean() == pytest.approx(twist, abs=1e-2)
            assert rise_sm.mean() == pytest.approx(rise_angle, abs=2e-2)

    exc_group.raise_exceptions()


PDB_TEXT = """
HEADER    SOME HEADER
TITLE     SOME PROTEIN
ATOM      1  N   ASP A   3      -1.748   0.194   0.277  1.00100.00           N
ATOM      2  CA  ASP A   3      -0.438  -0.361   0.725  1.00100.00           C
ATOM      3  C   ASP A   3      -0.186  -0.101   2.209  1.00100.00           C
ATOM      4  O   ASP A   3       0.628  -0.783   2.826  1.00100.00           O
ATOM      5  CB  ASP A   3       0.713   0.233  -0.097  1.00100.00           C
ATOM      6  CG  ASP A   3       0.421   0.261  -1.589  1.00100.00           C
ATOM      7  OD1 ASP A   3      -0.036  -0.766  -2.138  1.00100.00           O
ATOM      8  OD2 ASP A   3       0.645   1.322  -2.211  1.00100.00           O
"""


def test_calc_misc(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    ui.mole_layers.clear()
    ui.map_monomers(splines=[0])
    layer = ui.mole_layers.last()
    all_props = cylmeasure.LatticeParameters.choices()
    ui.calculate_lattice_structure(layer=layer, props=all_props)
    assert layer.features[Mole.radius].std() < 0.1
    ui.paint_molecules(layer, color_by=Mole.nth, limits=(0, 10))
    colors = ui.mole_layers.last().face_color
    ui.MoleculesMenu.View.plot_molecule_feature(layer, backend="qt")
    with tempfile.TemporaryDirectory() as dirpath:
        fp = Path(dirpath) / "test-project.tar"
        ui.save_project(fp)
        ui.load_project(fp, filter=None)
        assert_allclose(ui.mole_layers.last().face_color, colors)

        fp_img = Path(dirpath) / "test.mrc"
        fp_pdb = Path(dirpath) / "test.pdb"
        fp_pdb.write_text(PDB_TEXT)
        ui.sta.AlignmentMenu.TemplateImage.convert_pdb_to_image(
            fp_pdb, fp_img, degrees=[("z", 90)]
        )

        fp_csv = Path(dirpath) / "test.csv"
        df = pl.DataFrame({"z": [1, 2, 3], "y": [4, 5, 6], "x": [2, 1, 1]})
        df.write_csv(fp_csv)
        ui.sta.AlignmentMenu.TemplateImage.convert_csv_to_image(fp_csv, fp_img)
        df.with_columns(pl.Series("weight", [0.4, 0.5, 0.9])).write_csv(fp_csv)
        ui.sta.AlignmentMenu.TemplateImage.convert_csv_to_image(fp_csv, fp_img)


def test_lattice_structure_of_curved_microtubule(ui: CylindraMainWidget):
    ui.simulator.create_empty_image(size=(60.0, 200.0, 90.0), scale=1.0)
    ui.register_path(coords=[[35.0, 27.8, 22.0], [35.0, 106, 25.3], [35.0, 179, 55.5]])

    def _get_mean(p: str):
        s = ui.mole_layers.last().molecules.features[p]
        s0 = s.filter(s.is_finite())
        return s0.mean()

    ui.simulator.generate_molecules(
        spline=0,
        spacing=4.1,
        twist=-0.25,
        start=3,
        npf=14,
        radius=11.0,
    )
    ui.calculate_lattice_structure(
        layer="Mole(Sim)-0", props=["spacing", "twist", "skew_angle", "rise_angle"]
    )
    assert _get_mean("spacing") == pytest.approx(4.1, abs=3e-3)
    assert _get_mean("twist") == pytest.approx(-0.25, abs=1e-4)


def test_spline_fitter(ui: CylindraMainWidget):
    ui.open_image(
        TEST_DIR / "14pf_MT.tif",
        scale=1.052,
        tilt_range=(-60.0, 60.0),
        bin_size=[1],
        filter=None,
    )
    ui.register_path(coords=[[22, 117, 35], [22, 36, 58]])
    ui.register_path(coords=[[22, 117, 35], [22, 36, 58]])
    ui.SplinesMenu.Fitting.fit_splines_manually()
    ui.spline_fitter.auto_contrast.value = False
    ui.spline_fitter._next_pos()
    ui.spline_fitter._prev_pos()
    ui.spline_fitter._next_num()
    ui.spline_fitter._prev_num()
    ui.spline_fitter.pos.value = 1
    ui.spline_fitter.auto_contrast.value = True
    ui.spline_fitter._next_pos()
    ui.spline_fitter._prev_pos()
    ui.spline_fitter.resample_volumes(30)
    ui.spline_fitter.fit(shifts=[[1.094, 0.797], [1.094, 0.797], [1.094, 0.698]], i=0)
    ui.macro.undo()
    ui.macro.redo()


def test_cli(make_napari_viewer):
    import sys

    from cylindra.__main__ import main
    from cylindra.cli import set_testing
    from cylindra.core import ACTIVE_WIDGETS

    viewer: napari.Viewer = make_napari_viewer()
    set_testing(True)

    def run_cli(*args):
        sys.argv = [str(a) for a in args]
        main(viewer, ignore_sys_exit=True)

    # test help
    for cmd in [
        "average",
        "config",
        "find",
        "new",
        "open",
        "preview",
        "run",
        "workflow",
    ]:
        run_cli("cylindra", cmd, "--help")
    with thread_worker.blocking_mode():
        run_cli("cylindra")
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "project.json")
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "project.json", "--gui")
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_tar.tar")
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_zip.zip")
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_tar.tar::project.json")
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_tar.tar::Mole-0.csv")

        with tempfile.TemporaryDirectory() as dirpath:
            run_cli(
                "cylindra", "new",
                Path(dirpath) / "test-project",
                "--image", TEST_DIR / "14pf_MT.tif",
                "--multiscales", "1", "2",
                "--missing_wedge", "-60", "50",
                "--molecules", PROJECT_DIR_14PF / "Mole-*",
            )  # fmt: skip
            run_cli("cylindra", "open", Path(dirpath) / "test-project")
        run_cli("cylindra", "config", "--list")
        run_cli("cylindra", "config", PROJECT_DIR_14PF, "--remove")
        run_cli("cylindra", "workflow", "--list")
        with tempfile.TemporaryDirectory() as dirpath:
            run_cli(
                "cylindra", "average",
                TEST_DIR / "test_project_*",
                "--molecules", "Mole-*",
                "--size", "10.0",
                "--output", Path(dirpath) / "test.tif",
                "--filter", "col('nth') % 2 == 0",
                "--split", "--seed", "123",
            )  # fmt: skip
        run_cli("cylindra", "find", "**/*.zip")
        run_cli("cylindra", "find", "**/*.zip", "--called", "register_path")

    for widget in ACTIVE_WIDGETS:
        widget.close()
    ACTIVE_WIDGETS.clear()
    use_app().process_events()


def test_function_menu(make_napari_viewer):
    from cylindra.widgets.subwidgets import Volume

    viewer: napari.Viewer = make_napari_viewer()
    vol = Volume(viewer)
    viewer.window.add_dock_widget(vol)
    img = ip.asarray(
        np.arange(1000, dtype=np.float32).reshape(10, 10, 10), axes="zyx"
    ).set_scale(zyx=0.3, unit="nm")
    im = viewer.add_image(img, name="test image")
    vol.binning(im, 2)
    vol.gaussian_filter(im)
    vol.threshold(im)
    vol.binary_operation(im, "add", viewer.layers[-1])
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        vol.save_volume(viewer.layers[-1], dirpath / "test_image.tif")
        vol.save_volume(viewer.layers[-1], dirpath / "test_image.mrc")
        lbl = viewer.add_labels((img < 320).astype(np.int32), name="test labels")
        vol.save_label_as_mask(lbl, dirpath / "test_label.tif")
        vol.save_label_as_mask(lbl, dirpath / "test_label.mrc")
    vol.plane_clip()
    tester = mcls_testing.FunctionGuiTester(vol.gaussian_filter)
    tester.click_preview()
    tester.click_preview()
    tester = mcls_testing.FunctionGuiTester(vol.threshold)
    tester.click_preview()
    tester.click_preview()


def test_viterbi_alignment(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    layer = ui.parent_viewer.layers["Mole-0"]
    ui.filter_molecules(
        layer,
        "pl.col('nth').lt(4) & pl.col('pf-id').eq(1) | pl.col('nth').lt(5) & pl.col('pf-id').eq(4)",
    )
    layer_filt = ui.mole_layers.last()
    ui.sta.align_all_viterbi(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(2.3, 2.3, 2.3),
        range_long=(4, 4.5),
    )

    ui.sta.align_all_viterbi(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        range_long=(4, 4.5),
    )
    # multi-template
    ui.sta.align_all_viterbi(
        layer_filt,
        template_path=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(2.3, 2.3, 2.3),
        range_long=(4, 4.5),
    )

    ui.sta.align_all_viterbi(
        layer_filt,
        template_path=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        range_long=(4, 4.5),
    )

    mole = ui.mole_layers.last().molecules
    for _, sub in mole.groupby("pf-id"):
        dist = np.sqrt(np.sum((np.diff(sub.pos, axis=0)) ** 2, axis=1))
        assert np.all((4 <= dist) & (dist <= 4.5))
        assert np.all(sub.features["align-dx"].to_numpy() <= 2.3)
        assert np.all(sub.features["align-dy"].to_numpy() <= 2.3)
        assert np.all(sub.features["align-dz"].to_numpy() <= 2.3)


def test_mesh_annealing(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    layer = ui.parent_viewer.layers["Mole-0"]
    ui.filter_molecules(
        layer, "pl.col('nth').lt(3) | (pl.col('nth').eq(3) & pl.col('pf-id').lt(2))"
    )
    layer_filt = ui.mole_layers.last()
    mole = layer_filt.molecules
    dist_lon = np.sqrt(np.sum((mole.pos[0] - mole.pos[13]) ** 2))
    dist_lat = np.sqrt(np.sum((mole.pos[0] - mole.pos[1]) ** 2))
    assert dist_lon == pytest.approx(4.09, abs=0.2)

    # click preview
    tester = mcls_testing.FunctionGuiTester(ui.sta.align_all_annealing)
    tester.click_preview()

    # test return same results with same random seeds
    trajectories = []
    for _ in range(2):
        ui.sta.align_all_annealing(
            layer_filt,
            template_path=TEST_DIR / "beta-tubulin.mrc",
            mask_params=(0.3, 0.8),
            max_shifts=(1.2, 1.2, 1.2),
            rotations=((0, 0), (5, 5), (0, 0)),
            range_long=(dist_lon - 0.1, dist_lon + 0.1),
            range_lat=(dist_lat - 0.1, dist_lat + 0.1),
            angle_max=20,
            random_seeds=[0, 1],
        )
        trajectories.append(ui.mole_layers.last().metadata["annealing-result"].energies)
    assert trajectories[0] is not trajectories[1]
    assert_allclose(trajectories[0], trajectories[1])

    ui.sta.align_all_annealing(
        layer_filt,
        template_path=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        range_long=(dist_lon - 0.1, dist_lon + 0.1),
        range_lat=(dist_lat - 0.1, dist_lat + 0.1),
        angle_max=20,
        random_seeds=[0, 1],
    )
    ui.sta.save_annealing_scores(
        ui.mole_layers.last(), PROJECT_DIR_13PF / "annealing.csv"
    )
    ui.macro.undo()
    ui.macro.redo()


def test_landscape(ui: CylindraMainWidget):
    from cylindra._napari import LandscapeSurface

    ui.load_project(PROJECT_DIR_13PF, filter=None)
    layer = ui.parent_viewer.layers["Mole-0"]
    ui.filter_molecules(layer, "pl.col('nth') < 3")
    layer_filt = ui.mole_layers.last()
    mole = layer_filt.molecules
    dist_lon = np.sqrt(np.sum((mole.pos[0] - mole.pos[13]) ** 2))
    assert dist_lon == pytest.approx(4.09, abs=0.2)

    ui.sta.construct_landscape(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        upsample_factor=2,
    )
    layer_land = ui.parent_viewer.layers[-1]
    assert isinstance(layer_land, LandscapeSurface)
    repr(layer_land.landscape)
    layer_land.level = layer_land.level * 1.01  # test setter
    layer_land.show_min = False
    layer_land.show_min = True
    layer_land.resolution = 2.1

    ui.sta.run_align_on_landscape(layer_land)
    ui.sta.run_viterbi_on_landscape(
        layer_land,
        range_long=("-0.1", "+0.1"),
        angle_max=10,
    )
    # click preview
    tester = mcls_testing.FunctionGuiTester(ui.sta.run_annealing_on_landscape)
    tester.gui  # noqa: B018
    tester.click_preview()
    ui.sta.run_annealing_on_landscape(
        layer_land.name,
        range_long=("-0.1", "+0.1"),
        range_lat=("-0.1", "+0.1"),
        angle_max=20,
        random_seeds=[0, 1],
    )
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        ui.save_project(dirpath / "test-project.tar", save_landscape=True)
        ui.load_project(dirpath / "test-project.tar", filter=None)


def test_regionprops(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    for meth in ["mean", "median", "min", "max"]:
        ui.convolve_feature(
            layer=ui.parent_viewer.layers["Mole-0"],
            target="nth",
            method=meth,
            footprint=[[0, 1, 0], [1, 1, 1], [1, 1, 1]],
        )
    ui.binarize_feature(
        layer=ui.parent_viewer.layers["Mole-0"],
        target="pf-id",
        threshold=6.0,
        larger_true=True,
    )
    ui.label_feature_clusters(
        layer=ui.parent_viewer.layers["Mole-0"], target="pf-id_binarize"
    )
    ui.regionprops_features(
        layer=ui.parent_viewer.layers["Mole-0"],
        target="nth",
        label="pf-id_binarize_label",
        properties=[
            "area", "length", "width", "sum", "mean", "median",
            "max", "min", "std",
        ],
    )  # fmt: skip
    ui.count_neighbors("Mole-0")
    ui.distance_from_spline("Mole-0", spline=1)
    dist = ui.mole_layers[0].features["distance"][13:-13]
    assert np.all(np.abs(dist - ui.splines[0].radius) < 0.2)


def test_showing_widgets(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    ui.OthersMenu.Macro.show_macro()
    ui.OthersMenu.Macro.show_full_macro()
    ui.OthersMenu.Macro.show_native_macro()
    ui.OthersMenu.open_logger()
    loader = ui.FileMenu.open_image_loader()
    loader.path = TEST_DIR / "13pf_MT.tif"
    loader.scan_header()
    loader.preview_image().close()
    ui.FileMenu.view_project(PROJECT_DIR_13PF / "project.json")

    loader.tilt_model.value = None
    assert loader.tilt_model.value is None
    loader.tilt_model.value = {"kind": "x", "range": (-50, 60)}
    assert loader.tilt_model.value == {"kind": "x", "range": (-50, 60)}
    loader.tilt_model.value = {"kind": "y", "range": (-50, 60)}
    assert loader.tilt_model.value == {"kind": "y", "range": (-50, 60)}
    loader.tilt_model.value = {"kind": "dual", "xrange": (-50, 60), "yrange": (-50, 60)}
    assert loader.tilt_model.value == {
        "kind": "dual",
        "xrange": (-50, 60),
        "yrange": (-50, 60),
    }
    loader.tilt_model.value = (-60, 60)
    assert loader.tilt_model.value == {"kind": "y", "range": (-60, 60)}


def test_image_processor(ui: CylindraMainWidget):
    input_path = TEST_DIR / "13pf_MT.tif"
    ui.FileMenu.open_image_processor()
    ui.image_processor.input_image = input_path
    with tempfile.TemporaryDirectory() as dirpath:
        output_path = Path(dirpath) / "output.tif"
        ui.image_processor.convert_dtype(input_path, output_path, dtype="float32")
        ui.image_processor.invert(input_path, output_path)
        ui.image_processor.lowpass_filter(input_path, output_path)
        ui.image_processor.binning(input_path, output_path, bin_size=2)
        ui.image_processor.flip(input_path, output_path, axes="z")
        ui.image_processor.preview(input_path)


def test_workflows_custom(ui: CylindraMainWidget):
    name = "Test"
    code = "import numpy as np\ndef main(ui):\n    print(ui.default_config)\n"
    with tempfile.TemporaryDirectory() as dirpath, _config.patch_workflow_path(dirpath):
        ui.OthersMenu.Workflows.define_workflow(name, code)
        ui.OthersMenu.Workflows.edit_workflow(name, code)
        ui.OthersMenu.Workflows.edit_workflow(name, code)  # test overwriting
        ui.run_workflow(name)
        ui.OthersMenu.Workflows.run_workflow(name)
        ui.OthersMenu.Workflows.import_workflow(
            Path(dirpath) / f"{name}.py", name="imported"
        )
        ui.OthersMenu.Workflows.delete_workflow([name])
        ui.OthersMenu.Workflows.copy_workflow_directory()

        # test invalid code
        with pytest.raises(Exception):  # noqa: B017
            # attribute error
            ui.OthersMenu.Workflows.define_workflow(
                "Test-2", "def main(ui):\n    ui.bad_method_name()\n"
            )
        with pytest.raises(Exception):  # noqa: B017
            # not enough arguments
            ui.OthersMenu.Workflows.define_workflow(
                "Test-2", "def main(ui):\n    ui.open_image()\n"
            )


def test_stash(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None)
    with tempfile.TemporaryDirectory() as dirpath, _config.patch_stash_dir(dirpath):
        ui.FileMenu.Stash.stash_project()
        name0 = _config.get_stash_list()[0]
        ui.FileMenu.Stash.load_stash_project(name0, filter=None)
        ui.FileMenu.Stash.pop_stash_project(name0, filter=None)
        ui.FileMenu.Stash.stash_project()
        name1 = _config.get_stash_list()[0]
        ui.FileMenu.Stash.delete_stash_project(name1)
        ui.FileMenu.Stash.clear_stash_projects()
    ui.OthersMenu.configure_dask(num_workers=2)
    ui.OthersMenu.configure_dask(num_workers=None)
