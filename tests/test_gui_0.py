from pathlib import Path
import tempfile

from numpy.testing import assert_allclose
from acryo import Molecules
from magicclass import testing as mcls_testing

from cylindra import view_project
from cylindra.widgets import CylindraMainWidget
from cylindra.const import PropertyNames as H, MoleculesHeader as Mole
import pytest
from .utils import pytest_group
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
    assert ui.get_spline().orientation == ori
    assert ui.GlobalProperties.params.params2.polarity.txt == ori

    spec = ui.layer_prof.features["spline-id"] == ui.SplineControl.num
    arr = ui.layer_prof.text.string.array[spec]
    if ori == "MinusToPlus":
        assert (arr[0], arr[-1]) == ("-", "+")
    elif ori == "PlusToMinus":
        assert (arr[0], arr[-1]) == ("+", "-")


def test_click_buttons(ui: CylindraMainWidget):
    mcls_testing.check_function_gui_buildable(ui)


def test_tooltip(ui: CylindraMainWidget):
    mcls_testing.check_tooltip(ui)


@pytest.mark.parametrize(
    "save_path,npf", [(PROJECT_DIR_13PF, 13), (PROJECT_DIR_14PF, 14)]
)
def test_io(ui: CylindraMainWidget, save_path: Path, npf: int):
    path = TEST_DIR / f"{npf}pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=[1, 2])
    ui.set_multiscale(1)
    ui.register_path(coords=coords[npf])
    ui.register_path(coords=coords[npf][::-1])
    ui.run_workflow(interval=24.0)
    ui.auto_align_to_polarity(align_to="MinusToPlus")
    ui.map_monomers(splines=[0, 1])

    # Save project
    old_splines = ui.tomogram.splines.copy()
    old_molecules = [ui.get_molecules("Mono-0"), ui.get_molecules("Mono-1")]
    ui.save_project(save_path)
    ui.load_project(save_path, filter=True)
    new_splines = ui.tomogram.splines
    new_molecules = [ui.get_molecules("Mono-0"), ui.get_molecules("Mono-1")]
    assert old_splines[0] == new_splines[0]
    assert old_splines[1] == new_splines[1]
    for mol0, mol1 in zip(old_molecules, new_molecules):
        assert_molecule_equal(mol0, mol1)
    assert ui.tomogram.tilt_range == (-60, 60)


def test_spline_deletion(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    assert ui.layer_prof.features["spline-id"].values[0] == 0.0
    assert ui.layer_prof.features["spline-id"].values[-1] == 1.0
    ui.clear_current()
    assert ui.layer_prof.features["spline-id"].values[0] == 0.0
    assert ui.layer_prof.features["spline-id"].values[-1] == 0.0
    ui.register_path(coords=coords_13pf[::-1])
    assert ui.layer_prof.features["spline-id"].values[0] == 0.0
    assert ui.layer_prof.features["spline-id"].values[-1] == 1.0
    ui.delete_spline(0)
    assert ui.layer_prof.features["spline-id"].values[0] == 0.0
    assert ui.layer_prof.features["spline-id"].values[-1] == 0.0


def test_spline_switch(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.filter_reference_image()
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])

    # check canvas is updated correctly
    ui.add_anchors(interval=15.0)
    assert_canvas(ui, [True, True, True])
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

    ui.run_workflow(interval=16.0)

    # check results
    spl = ui.tomogram.splines[0]
    ypitch_mean = spl.localprops[H.spacing].mean()
    ypitch_glob = spl.get_globalprops(H.spacing)
    assert (
        4.075 < ypitch_glob < 4.105
    )  # GDP-bound microtubule has lattice spacing in this range
    assert abs(ypitch_glob - ypitch_mean) < 0.011
    assert all(spl.localprops[H.nPF] == 13)
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
    ui.SplineControl.num = 1
    ui.set_spline_props(spline=1, orientation="MinusToPlus")
    assert_orientation(ui, "MinusToPlus")
    ui.SplineControl.num = 0
    assert_orientation(ui, "PlusToMinus")
    ui.SplineControl.num = 1
    assert_orientation(ui, "MinusToPlus")
    assert_canvas(ui, [False, False, False])

    # Check align polarity.
    # Only spline 0 will get updated.
    ui.align_to_polarity(orientation="MinusToPlus")
    ui.SplineControl.num = 0
    ui.SplineControl.pos = 1
    assert_orientation(ui, "MinusToPlus")
    assert (
        ui.LocalProperties.params.spacing.txt
        == f" {ui.get_spline().localprops[H.spacing][1]:.2f} nm"
    )
    assert (
        ui.GlobalProperties.params.params1.spacing.txt
        == f" {ui.get_spline().get_globalprops(H.spacing):.2f} nm"
    )

    ui.SplineControl.num = 1
    assert ui.SplineControl.pos == 1
    assert_orientation(ui, "MinusToPlus")
    assert (
        ui.LocalProperties.params.spacing.txt
        == f" {ui.get_spline().localprops[H.spacing][1]:.2f} nm"
    )
    assert (
        ui.GlobalProperties.params.params1.spacing.txt
        == f" {ui.get_spline().get_globalprops(H.spacing):.2f} nm"
    )

    assert_canvas(ui, [False, False, False])

    ui.clear_all()

    assert ui.LocalProperties.params.spacing.txt == " -- nm"
    assert ui.GlobalProperties.params.params1.spacing.txt == " -- nm"


def test_set_label_colormaps(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.set_colormap(color_by="skewAngle", cmap="viridis", limits=(-1, 1))


def test_set_molecule_colormap(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.paint_molecules(
        ui.parent_viewer.layers["Mono-0"],
        "nth",
        {0: "blue", 1: "yellow"},
        (0, 10),
    )


def test_preview(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    tester = mcls_testing.FunctionGuiTester(ui.translate_molecules)
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer + 1
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.extend_molecules)
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

    tester = mcls_testing.FunctionGuiTester(ui.paint_molecules)
    nlayer = len(ui.parent_viewer.layers)
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer
    tester.click_preview()
    assert len(ui.parent_viewer.layers) == nlayer

    tester = mcls_testing.FunctionGuiTester(ui.load_project_for_reanalysis)
    tester.update_parameters(path=PROJECT_DIR_13PF)
    tester.click_preview()

    tester = mcls_testing.FunctionGuiTester(ui.load_molecules)
    tester.update_parameters(
        paths=[PROJECT_DIR_13PF / "Mono-0.csv", PROJECT_DIR_13PF / "Mono-1.csv"]
    )
    tester.click_preview()


@pytest.mark.parametrize("bin_size", [1, 2])
def test_sta(ui: CylindraMainWidget, bin_size: int):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.sta.average_all(ui.parent_viewer.layers["Mono-0"], size=12.0, bin_size=bin_size)
    for method in ["steps", "first", "last", "random"]:
        ui.sta.average_subset(
            ui.parent_viewer.layers["Mono-0"],
            size=12.0,
            method=method,
            bin_size=bin_size,
        )
    ui.sta.calculate_fsc(
        ui.parent_viewer.layers["Mono-0"],
        mask_params=None,
        size=8.0,
        seed=0,
        interpolation=1,
    )
    template_path = TEST_DIR / "beta-tubulin.mrc"
    ui.sta.align_averaged(
        layers=[ui.parent_viewer.layers["Mono-0"]],
        template_path=template_path,
        mask_params=(1, 1),
        bin_size=bin_size,
    )
    ui.sta.align_all(
        layers=[ui.parent_viewer.layers["Mono-0"]],
        template_path=template_path,
        mask_params=(1, 1),
        max_shifts=(1.0, 1.1, 1.0),
        y_rotation=(1.0, 1.0),
        interpolation=1,
        bin_size=bin_size,
    )
    ui.sta.seam_search(
        layer=ui.parent_viewer.layers["Mono-0"],
        template_path=template_path,
        mask_params=(1, 1),
    )

    with tempfile.TemporaryDirectory() as dirpath:
        molepath = Path(dirpath) / "monomers.txt"
        ui.save_molecules(layer=ui.parent_viewer.layers["Mono-0"], save_path=molepath)
        mole = ui.get_molecules("Mono-0")
        ui.load_molecules(molepath)
        mole_read = ui.get_molecules("monomers")
        assert_molecule_equal(mole, mole_read)

        ui.sta.save_last_average(dirpath)


@pytest_group("classify", maxfail=1)
@pytest.mark.parametrize("binsize", [1, 2])
def test_classify_pca(ui: CylindraMainWidget, binsize: int):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.sta.classify_pca(
        ui.parent_viewer.layers["Mono-0"],
        mask_params=None,
        size=6.0,
        interpolation=1,
        bin_size=binsize,
    )


def test_clip_spline(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    spl = ui.tomogram.splines[0]
    length_old = spl.length()
    ui.clip_spline(0, (10, 5))
    length_new = spl.length()
    assert (length_old - length_new) - 15 < 1e-2

    length_old = spl.length()
    ui.clip_spline(0, (3, 1))
    length_new = spl.length()
    assert (length_old - length_new) - 4 < 1e-2


def test_simulator(ui: CylindraMainWidget):
    ui.cylinder_simulator.create_empty_image(size=(50.0, 100.0, 50.0), scale=0.5)
    ui.register_path(coords=[[25.375, 83.644, 18.063], [25.375, 23.154, 28.607]])
    ui.cylinder_simulator.set_current_spline(idx=0)
    ui.cylinder_simulator.update_model(
        spacing=4.1,
        skew=-0.30,
        rise=11.0,
        npf=14,
        radius=9.14,
        offsets=(0.0, 0.18),
    )
    ui.cylinder_simulator.expand(exp=0.1, yrange=(11, 15), arange=(0, 14), allev=True)
    ui.cylinder_simulator.screw(skew=0.3, yrange=(11, 15), arange=(0, 14), allev=True)
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
        skew=-0.31,
        rise=10.5,
        npf=14,
        radius=9.14,
        offsets=(0.0, 0.0),
    )

    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        assert len(list(dirpath.glob("*"))) == 0
        ui.cylinder_simulator.simulate_tomogram(
            template_path=TEST_DIR / "beta-tubulin.mrc",
            save_dir=dirpath,
            nsr=[0.5, 2.0],
            tilt_range=(-60.0, 60.0),
            n_tilt=11,
            interpolation=1,
            seed=0,
        )
        assert len(list(dirpath.glob("*.mrc"))) == 2
    ui.cylinder_simulator.close()


@pytest_group("simulate", maxfail=1)
def test_simulate_tilt_series(ui: CylindraMainWidget):
    ui.cylinder_simulator.create_straight_line(
        25, (40, 42, 42), scale=0.5, yxrotation=10
    )
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        assert len(list(dirpath.glob("*"))) == 0
        ui.cylinder_simulator.simulate_tilt_series(
            template_path=TEST_DIR / "beta-tubulin.mrc",
            save_dir=dirpath,
            nsr=[0.5, 2.0],
            tilt_range=(-60.0, 60.0),
            n_tilt=11,
            interpolation=1,
            seed=0,
        )
        assert len(list(dirpath.glob("*.mrc"))) == 2
    ui.cylinder_simulator.close()


def test_project_viewer():
    view_project(PROJECT_DIR_13PF).close()


def test_show_orientation(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.show_orientation(ui.parent_viewer.layers["Mono-0"])


def test_merge_molecules(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.merge_molecule_info(
        pos=ui.parent_viewer.layers["Mono-0"],
        rotation=ui.parent_viewer.layers["Mono-1"],
        features=ui.parent_viewer.layers["Mono-0"],
    )
    last_layer = ui.parent_viewer.layers[-1]
    assert_allclose(last_layer.data, ui.parent_viewer.layers["Mono-0"].data)


def test_molecule_features(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    ui.show_molecule_features()
    ui.filter_molecules(
        layer=ui.parent_viewer.layers["Mono-0"], predicate='pl.col("position-nm") < 9.2'
    )
    ui.calculate_molecule_features(
        layer=ui.parent_viewer.layers["Mono-0"],
        column_name="new",
        expression='pl.col("pf-id") < 4',
    )
    ui.calculate_intervals(layer=ui.parent_viewer.layers["Mono-0"])


def test_auto_align(ui: CylindraMainWidget):
    path = TEST_DIR / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, tilt_range=(-60, 60), bin_size=2)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])

    ui.run_workflow(interval=32.0)
    ui.auto_align_to_polarity(align_to="MinusToPlus")
    assert ui.tomogram.splines[0].orientation == "MinusToPlus"
    assert ui.tomogram.splines[1].orientation == "MinusToPlus"


def test_molecules_to_spline(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_13PF, filter=False)
    assert len(ui.tomogram.splines) == 2
    ui.molecules_to_spline(layers=[ui.parent_viewer.layers["Mono-0"]], interval=20)
    assert len(ui.tomogram.splines) == 2


# NOTE: calc_intervals and calc_skews are very likely to contain bugs with different
# orientation, monomer mapping cases. Check all of the possible inputs just in case.


@pytest_group("calc_intervals", maxfail=4)
@pytest.mark.parametrize("orientation", ["PlusToMinus", "MinusToPlus"])
@pytest.mark.parametrize("path", [PROJECT_DIR_13PF, PROJECT_DIR_14PF])
@pytest.mark.parametrize("invert", [True, False])
def test_calc_intervals(
    ui: CylindraMainWidget,
    path: Path,
    invert: bool,
    orientation: str,
):
    ui.load_project(path, filter=False)
    spacing = ui.tomogram.splines[0].get_globalprops(H.spacing)
    npf = ui.tomogram.splines[0].get_globalprops(H.nPF)
    if invert:
        ui.invert_spline(spline=0)
    ui.map_monomers(splines=[0], orientation=orientation)
    layer = ui.parent_viewer.layers[-1]
    ui.calculate_intervals(layer=layer)
    interval = layer.features["interval-nm"][:-npf]
    # individial intervals must be almost equal to the global spacing
    assert abs(interval.mean() - spacing) < 1e-3


@pytest_group("calc_skews", maxfail=4)
@pytest.mark.parametrize("orientation", ["PlusToMinus", "MinusToPlus"])
@pytest.mark.parametrize("path", [PROJECT_DIR_13PF, PROJECT_DIR_14PF])
@pytest.mark.parametrize("invert", [False, True])
def test_calc_skews(
    ui: CylindraMainWidget,
    path: Path,
    invert: bool,
    orientation: str,
):
    ui.load_project(path, filter=False)
    skew_angle = ui.tomogram.splines[0].get_globalprops(H.skew)
    npf = ui.tomogram.splines[0].get_globalprops(H.nPF)
    if invert:
        ui.invert_spline(spline=0)
    ui.map_monomers(splines=[0], orientation=orientation)
    layer = ui.parent_viewer.layers[-1]
    ui.calculate_skews(layer=layer)
    each_skew = layer.features["skew-deg"][:-npf]
    # individial skews must be almost equal to the global skew angle
    assert abs(each_skew.mean() - skew_angle) < 1e-2


def test_spline_fitter(ui: CylindraMainWidget):
    ui.open_image(
        TEST_DIR / f"14pf_MT.tif",
        scale=1.052,
        tilt_range=(-60.0, 60.0),
        bin_size=[1],
        filter=False,
    )
    ui.register_path(coords=[[21.974, 117.148, 34.873], [21.974, 36.449, 58.084]])
    ui.spline_fitter.fit(
        shifts=[[1.094, 0.797], [1.094, 0.797], [1.094, 0.698]], i=0, max_interval=50.0
    )
