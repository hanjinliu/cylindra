from pathlib import Path
import tempfile
import napari
from itertools import product
import numpy as np
from numpy.testing import assert_allclose
import impy as ip
from acryo import Molecules
import polars as pl
from magicclass import testing as mcls_testing, get_function_gui
from magicclass.utils import thread_worker

from cylindra import view_project, _config, cylstructure
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.sta import MASK_CHOICES
from cylindra.const import PropertyNames as H, MoleculesHeader as Mole
import pytest
from .utils import pytest_group, ExceptionGroup
from ._const import TEST_DIR, PROJECT_DIR_13PF, PROJECT_DIR_14PF

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5]]
coords = {13: coords_13pf, 14: coords_14pf}


def assert_canvas(ui: CylindraMainWidget, isnone):
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
    assert ui.GlobalProperties.params.params2.polarity.txt == ori

    spec = ui._reserved_layers.prof.features["spline-id"] == ui.SplineControl.num
    arr = ui._reserved_layers.prof.text.string.array[spec]
    if ori == "MinusToPlus":
        assert (arr[0], arr[-1]) == ("-", "+")
    elif ori == "PlusToMinus":
        assert (arr[0], arr[-1]) == ("+", "-")


def test_click_buttons(ui: CylindraMainWidget):
    mcls_testing.check_function_gui_buildable(ui)


def test_tooltip(ui: CylindraMainWidget):
    mcls_testing.check_tooltip(ui)


def test_misc_actions(ui: CylindraMainWidget):
    ui.OthersMenu.cylindra_info()
    ui.OthersMenu.open_command_palette()


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
    ui._runner.run(interval=24.0, paint=True)
    ui.auto_align_to_polarity(align_to="MinusToPlus")
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
    for mol0, mol1 in zip(old_molecules, new_molecules):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt_range == (-60, 60)

    # try .tar file
    ui.save_project(save_path / "test_tar.tar")
    ui.overwrite_project()
    ui.load_project(save_path / "test_tar.tar", filter="DoG")
    assert len(ui.macro.undo_stack["undo"]) == 0
    new_splines = ui.tomogram.splines
    new_molecules = list(ui.mole_layers.iter_molecules())
    assert old_splines[0].close_to(new_splines[0])
    assert old_splines[1].close_to(new_splines[1])
    for mol0, mol1 in zip(old_molecules, new_molecules):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt_range == (-60, 60)

    # try .zip file
    ui.save_project(save_path / "test_zip.zip")
    ui.overwrite_project()
    ui.load_project(save_path / "test_zip.zip", filter="DoG")
    assert len(ui.macro.undo_stack["undo"]) == 0
    new_splines = ui.tomogram.splines
    new_molecules = list(ui.mole_layers.iter_molecules())
    assert old_splines[0].close_to(new_splines[0])
    assert old_splines[1].close_to(new_splines[1])
    for mol0, mol1 in zip(old_molecules, new_molecules):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt_range == (-60, 60)

    ui.show_splines()
    ui.show_splines_as_meshes()
    ui.load_splines(save_path / "spline-0.json")
    ui.set_source_spline(ui.mole_layers["Mole-0"], 0)


def test_io_with_different_data(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    params = [
        dict(local_props=False, global_props=False),
        dict(local_props=False, global_props=True),
        dict(local_props=True, global_props=False),
        dict(local_props=True, global_props=True),
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

    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False, read_image=False)


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
        ui._runner.run([0], local_props=False, paint=False)  # no localprops
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
    assert len(ui.macro.undo_stack["undo"]) == 0
    assert ui.splines[0].radius is None
    assert len(ui.macro) > 1
    assert str(ui.macro[-1]).startswith("ui.register_path(")


def test_map_molecules(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
    assert ui.get_loader("Mole-0").molecules is ui.mole_layers["Mole-0"].molecules
    ui.map_monomers_with_extensions(0, {0: (1, 1), 1: (-1, -1)})
    ui.map_along_pf(0)
    ui.map_centers([0])
    ui.macro.undo()
    ui.macro.undo()
    ui.macro.undo()
    ui.macro.redo()
    ui.macro.redo()
    ui.macro.redo()


def test_napari_operations(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
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
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = Path(tmpdir) / "test_macro.py"
        fp.write_text("print(0)")
        ui.OthersMenu.Macro.load_macro_file(fp)


def test_spline_switch(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.filter_reference_image()
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])

    # check canvas is updated correctly
    ui.add_anchors(interval=15.0)
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

        ui._runner.run(interval=32.6)

        # check results
        for spl in ui.tomogram.splines:
            spacing_mean = spl.localprops[H.spacing].mean()
            spacing_glob = spl.props.get_glob(H.spacing)
            # GDP-bound microtubule has lattice spacing in this range
            assert 4.08 < spacing_glob < 4.11
            assert spacing_glob == pytest.approx(spacing_mean, abs=0.02)
            assert all(spl.localprops[H.npf] == 13)
            assert all(spl.localprops[H.rise] > 8.3)

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

        ui.SplinesMenu.show_localprops()

        # Check align polarity.
        # Only spline 0 will get updated.
        ui.align_to_polarity(orientation="MinusToPlus")
        ui.SplineControl.num = 0
        ui.SplineControl.pos = 1
        assert_orientation(ui, "MinusToPlus")
        assert (
            ui.LocalProperties.params.spacing.txt
            == f" {ui.splines[ui.SplineControl.num].localprops[H.spacing][1]:.2f} nm"
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
            == f" {ui.splines[ui.SplineControl.num].localprops[H.spacing][1]:.2f} nm"
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


def test_preview(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
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
    ui.binarize_feature(ui.mole_layers["Mole-0"], "nth", threshold=3)
    tester = mcls_testing.FunctionGuiTester(ui.label_feature_clusters)
    tester.click_preview()


def test_sub_widgets(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    ui.ImageMenu.open_slicer()
    ui.spline_slicer.refresh_widget_state()
    ui.spline_slicer.show_what = "CFT"
    ui.spline_slicer._update_canvas()
    ui.spline_slicer.show_what = "R-projection"
    ui.spline_slicer._update_canvas()
    ui.spline_slicer.show_what = "Y-projection"
    ui.spline_slicer._update_canvas()
    ui.spline_slicer.show_what = "Filtered-R-projection"
    ui.spline_slicer._update_canvas()
    ui.spline_slicer.measure_cft_here()

    # file iterator
    ui._file_iterator.set_pattern(f"{TEST_DIR.as_posix()}/*.tif")
    ui._file_iterator.last_file()
    ui._file_iterator.first_file()
    ui._file_iterator.next_file()
    ui._file_iterator.prev_file()
    ui._file_iterator.open_image(ui._file_iterator.path)
    ui._file_iterator.preview_all().close()

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
    ui.AnalysisMenu.open_spectra_inspector()
    ui.spectra_inspector.log_scale = True
    ui.spectra_inspector.log_scale = False


@pytest.mark.parametrize("bin_size", [1, 2])
def test_sta(ui: CylindraMainWidget, bin_size: int):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    ui.sta.average_all("Mole-0", size=12.0, bin_size=bin_size)
    for method in ["steps", "first", "last", "random"]:
        ui.sta.average_subset(
            "Mole-0",
            size=12.0,
            method=method,
            bin_size=bin_size,
        )
    ui.sta.average_groups("Mole-0", size=12.0, bin_size=bin_size, by=pl.col("pf-id"))
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
    ui.sta.params.mask_choice = MASK_CHOICES[2]
    ui.sta.params.mask_choice = MASK_CHOICES[1]
    ui.sta.show_template()
    ui.sta.show_mask()
    ui.sta.align_averaged(
        layers=["Mole-0"],
        template_path=template_path,
        mask_params=(1, 1),
        bin_size=bin_size,
    )
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
    ui.sta.align_all_template_free(
        layers=["Mole-0"],
        mask_params=(1, 1),
        size=12.0,
        bin_size=bin_size,
    )
    ui.sta.align_all_multi_template(
        layers=["Mole-0"],
        template_paths=[template_path, template_path],
        mask_params=(1, 1),
        bin_size=bin_size,
    )


def test_seam_search(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    ui.filter_molecules(
        ui.parent_viewer.layers["Mole-0"], predicate="pl.col('nth') < 5"
    )
    ui.sta.params.template_path.value = TEST_DIR / "beta-tubulin.mrc"
    ui.sta.params.mask_choice = MASK_CHOICES[1]
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


def test_classify_pca(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
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
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
    mole = ui.mole_layers["Mole-0"].molecules
    shifts = np.zeros((mole.count(), 3), dtype=np.float32)
    shifts[:, 0] = np.linspace(-1, 1, mole.count())
    mole_tr = mole.translate_internal(shifts)

    ui.add_molecules(mole_tr, "Corn", source=ui.splines[0])
    layer = ui.parent_viewer.layers["Corn"]
    ui.measure_radius_by_molecules([layer], interval=8, depth=12)
    radii = ui.splines[0].localprops[H.radius]
    assert all(np.diff(radii) > 0)

    spl = ui.tomogram.splines[0]
    ui.set_radius([0], 11.2)
    assert spl.radius == pytest.approx(11.2, abs=1e-6)
    ui.set_radius([0], "10.4")
    assert spl.radius == pytest.approx(10.4, abs=1e-6)
    ui.set_radius([0], "pl.col('npf') * 0.9")
    assert spl.radius == pytest.approx(spl.props.get_glob("npf") * 0.9, abs=1e-6)


def test_simulator(ui: CylindraMainWidget):
    ui.cylinder_simulator.create_empty_image(size=(50.0, 100.0, 50.0), scale=0.5)
    ui.register_path(coords=[[25.375, 83.644, 18.063], [25.375, 23.154, 28.607]])
    ui.cylinder_simulator.set_current_spline(idx=0)
    ui.cylinder_simulator.update_model(
        spacing=4.1,
        dimer_twist=-0.30,
        start=3,
        npf=14,
        radius=9.14,
        offsets=(0.0, 0.18),
    )
    ui.cylinder_simulator.expand(exp=0.1, yrange=(11, 15), arange=(0, 14), allev=True)
    ui.cylinder_simulator.twist(
        dimer_twist=0.3, yrange=(11, 15), arange=(0, 14), allev=True
    )
    ui.cylinder_simulator.dilate(
        radius=-0.5, yrange=(11, 15), arange=(0, 14), allev=True
    )
    ui.cylinder_simulator.send_moleclues_to_viewer()
    ui.cylinder_simulator.close()


@pytest_group("simulate", maxfail=1)
def test_simulate_tomogram(ui: CylindraMainWidget):
    ui.cylinder_simulator.create_straight_line(25, (40, 42, 42), scale=0.5)
    ui.cylinder_simulator.update_model(
        spacing=4.06,
        dimer_twist=-0.31,
        start=3,
        npf=14,
        radius=9.14,
        offsets=(0.0, 0.0),
    )

    kwargs = dict(
        template_path=TEST_DIR / "beta-tubulin.mrc",
        tilt_range=(-60.0, 60.0),
        n_tilt=11,
        interpolation=1,
        seed=0,
    )
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        assert len(list(dirpath.glob("*"))) == 0
        ui.cylinder_simulator.simulate_tomogram(
            **kwargs, nsr=[0.5, 2.0], save_dir=dirpath
        )
        assert len(list(dirpath.glob("*.mrc"))) == 2
    ui.cylinder_simulator.simulate_tomogram_and_open(**kwargs, nsr=1.2)
    ui.cylinder_simulator.close()


@pytest_group("simulate", maxfail=1)
def test_simulate_tilt_series(ui: CylindraMainWidget):
    ui.cylinder_simulator.create_straight_line(
        25, (40, 42, 42), scale=0.5, yxrotation=10
    )
    with tempfile.TemporaryDirectory() as dirpath:
        fp = Path(dirpath) / "test.mrc"
        assert not fp.exists()
        ui.cylinder_simulator.simulate_tilt_series(
            template_path=TEST_DIR / "beta-tubulin.mrc",
            save_path=fp,
            tilt_range=(-60.0, 60.0),
            n_tilt=11,
            interpolation=1,
        )
        assert fp.exists()
    ui.cylinder_simulator.close()


def test_project_viewer():
    view_project(PROJECT_DIR_14PF).close()


def test_molecules_methods(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
    layer0 = ui.mole_layers["Mole-0"]
    layer1 = ui.mole_layers["Mole-1"]
    ui.MoleculesMenu.View.show_orientation(layer0)
    ui.concatenate_molecules([layer0, layer1])
    last_layer = ui.mole_layers.last()
    assert last_layer.data.shape[0] == layer0.data.shape[0] + layer1.data.shape[0]
    ui.mole_layers.delete(include="concat")
    ui.split_molecules("Mole-0", by=Mole.pf)
    ui.interpolate_spline_properties("Mole-0", interpolation=1)
    ui.MoleculesMenu.View.render_molecules(
        layer0,
        template_path=TEST_DIR / "beta-tubulin.mrc",
    )
    ui.paint_molecules(layer0, color_by="pf-id", limits=(0, 12))
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
    ui.ImageMenu.show_colorbar(ui.mole_layers["Mole-0"])


def test_transform_molecules(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
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
    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
    ui.merge_molecule_info(pos="Mole-0", rotation="Mole-1", features="Mole-0")
    assert_allclose(ui.mole_layers.last().data, ui.mole_layers["Mole-0"].data)


def test_molecule_features(ui: CylindraMainWidget):
    import polars as pl

    ui.load_project(PROJECT_DIR_14PF, filter=None, paint=False)
    ui.MoleculesMenu.View.show_molecule_features()
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


def test_auto_align(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])

    ui._runner.run(interval=32.0)
    ui.auto_align_to_polarity(align_to="MinusToPlus")
    assert ui.tomogram.splines[0].orientation == "MinusToPlus"
    assert ui.tomogram.splines[1].orientation == "MinusToPlus"


def test_molecules_to_spline(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
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


def test_calc_lattice_structures(ui: CylindraMainWidget):
    exc_group = ExceptionGroup()
    for orientation, path, invert in product(
        ["PlusToMinus", "MinusToPlus"],
        [PROJECT_DIR_13PF, PROJECT_DIR_14PF],
        [True, False],
    ):
        ui.load_project(path, filter=None, paint=False, read_image=False)
        spacing = ui.tomogram.splines[0].props.get_glob(H.spacing)
        skew = ui.tomogram.splines[0].props.get_glob(H.skew)
        dimer_twist = ui.tomogram.splines[0].props.get_glob(H.dimer_twist)
        rise_angle = ui.tomogram.splines[0].props.get_glob(H.rise)
        npf = ui.tomogram.splines[0].props.get_glob(H.npf)
        if invert:
            ui.invert_spline(spline=0)
        ui.mole_layers.clear()
        ui.map_monomers(splines=[0], orientation=orientation)
        layer = ui.mole_layers.last()
        ui.calculate_lattice_structure(layer, ["interv", "skew", "dimer_twist", "rise"])
        ay_ratio: float = np.sin(np.pi / npf) * npf / np.pi
        with exc_group.merging(f"ori={orientation}, path={path.name}, inv={invert}"):
            # individial parameters must be almost equal to the global ones
            feat = layer.molecules.features
            spacing_sm = feat["interval-nm"][npf:-npf]
            skew_sm = feat["skew-deg"][npf:-npf] / ay_ratio
            twist_sm = feat["dimer-twist-deg"][npf:-npf]
            r = "rise-angle-deg"
            feat_rise = feat.filter(pl.col(r).is_finite() & pl.col(r).is_not_nan())[r]
            rise_sm = feat_rise * ay_ratio

            assert spacing_sm.median() == pytest.approx(spacing, abs=1e-3)
            assert skew_sm.median() == pytest.approx(skew, abs=1e-2)
            assert twist_sm.median() == pytest.approx(dimer_twist, abs=1e-2)
            assert rise_sm.median() == pytest.approx(rise_angle, abs=1e-2)

    exc_group.raise_exceptions()


def test_calc_misc(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    ui.map_monomers(splines=[0])
    layer = ui.mole_layers.last()
    all_props = cylstructure.LatticeParameters.choices()
    ui.calculate_lattice_structure(layer=layer, props=all_props)
    assert layer.features["radius-nm"].std() < 0.1
    ui.paint_molecules(layer, color_by="radius-nm", limits=(8, 10))
    ui.MoleculesMenu.View.plot_molecule_feature(layer, backend="qt")


def test_spline_fitter(ui: CylindraMainWidget):
    ui.open_image(
        TEST_DIR / f"14pf_MT.tif",
        scale=1.052,
        tilt_range=(-60.0, 60.0),
        bin_size=[1],
        filter=None,
    )
    ui.register_path(coords=[[21.974, 117.148, 34.873], [21.974, 36.449, 58.084]])
    ui.SplinesMenu.fit_splines_manually(30)
    ui.spline_fitter.controller.pos.value = 1
    ui.spline_fitter.fit(
        shifts=[[1.094, 0.797], [1.094, 0.797], [1.094, 0.698]], i=0, max_interval=50.0
    )
    ui.macro.undo()
    ui.macro.redo()


def test_cli(make_napari_viewer):
    import sys
    from cylindra.__main__ import main

    viewer: napari.Viewer = make_napari_viewer()
    sys.argv = ["cylindra"]
    main(viewer)
    sys.argv = ["cylindra", str(PROJECT_DIR_14PF / "project.json")]
    main(viewer)


def test_function_menu(make_napari_viewer):
    from cylindra.widgets.subwidgets import Volume

    viewer: napari.Viewer = make_napari_viewer()
    vol = Volume()
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


def test_viterbi_alignment(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    layer = ui.parent_viewer.layers["Mole-0"]
    ui.filter_molecules(layer, "(pl.col('nth') < 4) & (pl.col('pf-id') < 2)")
    layer_filt = ui.mole_layers.last()
    ui.sta.align_all_viterbi(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(2.3, 2.3, 2.3),
        distance_range=(4, 4.5),
    )

    ui.sta.align_all_viterbi(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        distance_range=(4, 4.5),
    )

    ui.sta.align_all_viterbi_multi_template(
        layer_filt,
        template_paths=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(2.3, 2.3, 2.3),
        distance_range=(4, 4.5),
    )

    ui.sta.align_all_viterbi_multi_template(
        layer_filt,
        template_paths=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        distance_range=(4, 4.5),
    )

    mole = ui.mole_layers.last().molecules
    for _, sub in mole.groupby("pf-id"):
        dist = np.sqrt(np.sum((np.diff(sub.pos, axis=0)) ** 2, axis=1))
        assert np.all((4 <= dist) & (dist <= 4.5))
        assert np.all(sub.features["align-dx"].to_numpy() <= 2.3)
        assert np.all(sub.features["align-dy"].to_numpy() <= 2.3)
        assert np.all(sub.features["align-dz"].to_numpy() <= 2.3)


def test_mesh_annealing(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    layer = ui.parent_viewer.layers["Mole-0"]
    ui.filter_molecules(layer, "pl.col('nth') < 3")
    layer_filt = ui.mole_layers.last()
    mole = layer_filt.molecules
    dist_lon = np.sqrt(np.sum((mole.pos[0] - mole.pos[13]) ** 2))
    dist_lat = np.sqrt(np.sum((mole.pos[0] - mole.pos[1]) ** 2))
    assert dist_lon == pytest.approx(4.09, abs=0.2)

    # click preview
    fgui = get_function_gui(ui.sta.align_all_annealing)
    assert fgui[-2].widget_type == "PushButton" and fgui[-2] is not fgui.call_button
    fgui[-2].clicked()
    fgui = get_function_gui(ui.sta.align_all_annealing_multi_template)
    assert fgui[-2].widget_type == "PushButton" and fgui[-2] is not fgui.call_button
    fgui[-2].clicked()

    ui.sta.align_all_annealing(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        distance_range_long=(dist_lon - 0.1, dist_lon + 0.1),
        distance_range_lat=(dist_lat - 0.1, dist_lat + 0.1),
        angle_max=20,
        random_seeds=[0],
    )

    ui.sta.align_all_annealing(
        layer_filt,
        template_path=TEST_DIR / "beta-tubulin.mrc",
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        distance_range_long=(dist_lon - 0.1, dist_lon + 0.1),
        distance_range_lat=(dist_lat - 0.1, dist_lat + 0.1),
        angle_max=20,
        random_seeds=[0, 1],
        return_all=True,
    )

    ui.sta.align_all_annealing_multi_template(
        layer_filt,
        template_paths=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        distance_range_long=(dist_lon - 0.1, dist_lon + 0.1),
        distance_range_lat=(dist_lat - 0.1, dist_lat + 0.1),
        angle_max=20,
        random_seeds=[0],
    )

    ui.sta.align_all_annealing_multi_template(
        layer_filt,
        template_paths=[TEST_DIR / "beta-tubulin.mrc", TEST_DIR / "beta-tubulin.mrc"],
        mask_params=(0.3, 0.8),
        max_shifts=(1.2, 1.2, 1.2),
        rotations=((0, 0), (5, 5), (0, 0)),
        distance_range_long=(dist_lon - 0.1, dist_lon + 0.1),
        distance_range_lat=(dist_lat - 0.1, dist_lat + 0.1),
        angle_max=20,
        random_seeds=[0, 1],
        return_all=True,
    )
    ui.macro.undo()
    ui.macro.redo()


def test_regionprops(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
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
            "area",
            "length",
            "width",
            "sum",
            "mean",
            "median",
            "max",
            "min",
            "std",
        ],
    )
    ui.calculate_curve_index("Mole-0")


def test_showing_widgets(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    ui.OthersMenu.Macro.show_macro()
    ui.OthersMenu.Macro.show_full_macro()
    ui.OthersMenu.Macro.show_native_macro()
    ui.OthersMenu.open_logger()
    ui.FileMenu.open_image_loader()
    ui.FileMenu.view_project(PROJECT_DIR_13PF / "project.json")


def test_image_processor(ui: CylindraMainWidget):
    input_path = TEST_DIR / "13pf_MT.tif"
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
    code = (
        "import numpy as np\n"
        "def main(ui):\n"
        "    ui.load_project('path/to/project.json')\n"
    )
    with tempfile.TemporaryDirectory() as dirpath, _config.patch_workflow_path(dirpath):
        ui.OthersMenu.Workflows.define_workflow(name, code)
        ui.OthersMenu.Workflows.edit_workflow(name, code)
        ui.OthersMenu.Workflows.import_workflow(
            Path(dirpath) / f"{name}.py", name="imported"
        )
        ui.OthersMenu.Workflows.delete_workflow([name])
        ui.OthersMenu.Workflows.copy_workflow_directory()


def test_stash(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=None, paint=False)
    with tempfile.TemporaryDirectory() as dirpath, _config.patch_stash_dir(dirpath):
        ui.FileMenu.Stash.stash_project()
        name0 = _config.get_stash_list()[0]
        ui.FileMenu.Stash.load_stash_project(name0, filter=None)
        ui.FileMenu.Stash.pop_stash_project(name0, filter=None)
        ui.FileMenu.Stash.stash_project()
        name1 = _config.get_stash_list()[0]
        ui.FileMenu.Stash.delete_stash_project(name1)
        ui.FileMenu.Stash.clear_stash_projects()
