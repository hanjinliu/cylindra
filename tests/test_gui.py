from pathlib import Path
import tempfile

from numpy.testing import assert_allclose
from acryo import Molecules
from cylindra import start, view_project, CylindraMainWidget
from cylindra.const import PropertyNames as H

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]]
TEST_PATH = Path(__file__).parent

def assert_canvas(ui: CylindraMainWidget, isnone):
    for i in range(3):
        if isnone[i]:
            assert ui.SplineControl.canvas[i].image is None, f"{i}-th canvas"
        else:
            assert ui.SplineControl.canvas[i].image is not None, f"{i}-th canvas"

def assert_molecule_equal(mole0: Molecules, mole1: Molecules):
    assert_allclose(mole0.pos, mole1.pos, atol=1e-8, rtol=1e-8)
    assert_allclose(mole0.x, mole1.x, atol=1e-8, rtol=1e-8)
    assert_allclose(mole0.y, mole1.y, atol=1e-8, rtol=1e-8)
    assert_allclose(mole0.z, mole1.z, atol=1e-8, rtol=1e-8)

def assert_orientation(ui: CylindraMainWidget, ori: str):
    assert ui.get_spline().orientation == ori
    assert ui.GlobalProperties.params.params2.polarity.txt == ori
    
    spec = ui.layer_prof.features["spline-id"] == ui.SplineControl.num
    arr = ui.layer_prof.text.string.array[spec]
    if ori == "MinusToPlus":
        assert (arr[0], arr[-1]) == ("-", "+")
    elif ori == "PlusToMinus":
        assert (arr[0], arr[-1]) == ("+", "-")

def test_spline_deletion():
    ui = start()
    path = TEST_PATH / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, bin_size=2)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    assert ui.layer_prof.features["spline-id"].values[0] == 0.
    assert ui.layer_prof.features["spline-id"].values[-1] == 1.
    ui.clear_current()
    assert ui.layer_prof.features["spline-id"].values[0] == 0.
    assert ui.layer_prof.features["spline-id"].values[-1] == 0.
    ui.register_path(coords=coords_13pf[::-1])
    assert ui.layer_prof.features["spline-id"].values[0] == 0.
    assert ui.layer_prof.features["spline-id"].values[-1] == 1.
    ui.delete_spline(0)
    assert ui.layer_prof.features["spline-id"].values[0] == 0.
    assert ui.layer_prof.features["spline-id"].values[-1] == 0.
    
    ui.parent_viewer.layers.events.removing.disconnect()
    ui.parent_viewer.layers.events.removed.disconnect()

def test_spline_switch():
    ui = start()
    path = TEST_PATH / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, bin_size=2)
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
    
    ui.cylindrical_fit(interval=16.0)
    
    # check results
    spl = ui.tomogram.splines[0]
    ypitch_mean = spl.localprops[H.yPitch].mean()
    ypitch_glob = spl.globalprops[H.yPitch]
    assert 4.075 < ypitch_glob < 4.105  # GDP-bound microtubule has lattice spacing in this range
    assert abs(ypitch_glob - ypitch_mean) < 0.011
    assert all(spl.localprops[H.nPF] == 13)
    assert all(spl.localprops[H.riseAngle] > 8.3)
    
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
    ui.SplineControl.set_orientation(i=0, orientation="PlusToMinus")
    assert_orientation(ui, "PlusToMinus")
    ui.SplineControl.num = 1
    ui.SplineControl.set_orientation(i=1, orientation="MinusToPlus")
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
    assert ui.LocalProperties.params.pitch.txt == f" {ui.get_spline().localprops[H.yPitch].values[1]:.2f} nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {ui.get_spline().globalprops[H.yPitch]:.2f} nm"
    
    ui.SplineControl.num = 1
    assert ui.SplineControl.pos == 1
    assert_orientation(ui, "MinusToPlus")
    assert ui.LocalProperties.params.pitch.txt == f" {ui.get_spline().localprops[H.yPitch].values[1]:.2f} nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {ui.get_spline().globalprops[H.yPitch]:.2f} nm"
    
    assert_canvas(ui, [False, False, False])

    # map monomer coordinates and save them.
    ui.map_monomers(splines=[0])
    ui.average_subset(ui.parent_viewer.layers['Mono-0'], size=18.)
    ui.calculate_fsc(ui.parent_viewer.layers['Mono-0'], mask_params=None, size=18.,
                     seed=0, interpolation=1)
    template_path = TEST_PATH / "template.mrc"
    ui.align_averaged(layer=ui.parent_viewer.layers['Mono-0'], template_path=template_path, 
                      mask_params=(1, 1))
    ui.align_all(layer=ui.parent_viewer.layers['Mono-0'], template_path=template_path, mask_params=(1, 1), 
                 tilt_range=(-60., 60.), max_shifts=(1.0, 1.1, 1.0), y_rotation=(1.0, 1.0))
    ui.seam_search(layer=ui.parent_viewer.layers['Mono-0'], template_path=template_path, mask_params=(1, 1))
    ui.save_molecules(layer=ui.parent_viewer.layers['Mono-0'],
                      save_path=TEST_PATH/"monomers.txt"
                      )
    mole = ui.get_molecules('Mono-0')
    ui.clear_all()
    ui.load_molecules(TEST_PATH/"monomers.txt")
    mole_read = ui.get_molecules('monomers')
    assert_molecule_equal(mole, mole_read)
    assert_canvas(ui, [True, True, True])
    assert ui.LocalProperties.params.pitch.txt == " -- nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == " -- nm"
    
    # cleanup
    ui.parent_viewer.layers.events.removing.disconnect()
    ui.parent_viewer.layers.events.removed.disconnect()
    ui.sub_viewer.close()
 

def test_io():
    ui = start()
    path = TEST_PATH / "13pf_MT.tif"
    ui.open_image(path=path, scale=1.052, bin_size=1)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    ui.cylindrical_fit(interval=24.0)
    ui.map_monomers(splines=[0, 1])
    
    # Save project
    old_splines = ui.tomogram.splines.copy()
    old_molecules = [ui.get_molecules('Mono-0'), ui.get_molecules('Mono-1')]
    ui.save_project(TEST_PATH / "test-project.json")
    ui.load_project(TEST_PATH / "test-project.json")
    new_splines = ui.tomogram.splines
    new_molecules = [ui.get_molecules('Mono-0'), ui.get_molecules('Mono-1')]
    assert old_splines[0] == new_splines[0]
    assert old_splines[1] == new_splines[1]
    for mol0, mol1 in zip(old_molecules, new_molecules):
        assert_molecule_equal(mol0, mol1)

    # cleanup
    ui.parent_viewer.layers.events.removing.disconnect()
    ui.parent_viewer.layers.events.removed.disconnect()

def test_simulator():
    ui = start()
    ui._Simulator.create_empty_image(size=(100.0, 200.0, 100.0), scale=0.5, bin_size=[2])
    ui.register_path(coords=[[50.25, 179.886, 33.022], [50.25, 23.331, 72.339]])
    ui._Simulator.set_current_spline(idx=0)
    ui._Simulator.update_model(idx=0, interval=4.1, skew=-0.30, rise=11.0, npf=14, radius=9.14, offsets=(0.0, 0.18))
    ui._Simulator.expand(shift=0.1, yrange=(11, 31), arange=(0, 14), n_allev=1)
    ui._Simulator.screw(skew=0.3, yrange=(11, 31), arange=(0, 14), n_allev=1)
    ui._Simulator.dilate(radius=-0.5, yrange=(11, 31), arange=(0, 14), n_allev=1)
    ui._Simulator.send_moleclues_to_viewer()

def test_single_simulation():
    ui = start()
    ui._Simulator.create_empty_image(size=(100.0, 200.0, 100.0), scale=0.5, bin_size=[2])
    ui.register_path(coords=[[50.25, 179.886, 33.022], [50.25, 23.331, 72.339]])
    ui._Simulator.set_current_spline(idx=0)
    ui._Simulator.update_model(idx=0, interval=4.1, skew=-0.30, rise=11.0, npf=14, radius=9.14, offsets=(0.0, 0.18))
    ui._Simulator.simulate_tomogram(path=TEST_PATH / "template.mrc")

def test_batch_simulation():
    ui = start()
    ui._Simulator.create_empty_image(size=(50.0, 100.0, 50.0), scale=0.5, bin_size=[4])
    ui.register_path(coords=[[25.375, 83.644, 18.063], [25.375, 23.154, 28.607]])
    ui._Simulator.set_current_spline(idx=0)
    ui._Simulator.update_model(idx=0, interval=4.1, skew=-0.31, rise=10.5, npf=14, radius=9.14, offsets=(0.0, 0.0))
    
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = Path(dirpath)
        assert len(list(dirpath.glob("*"))) == 0
        ui._Simulator.simulate_tomogram_batch(
            path=TEST_PATH / "template.mrc",
            save_path=dirpath,
            nsr=[0.5, 1.0, 2.0],
            tilt_range=(-60.0, 60.0),
            n_tilt=31,
            interpolation=3,
            save_mode='mrc',
            seed=0,
        )
        assert len(list(dirpath.glob("*.mrc"))) == 3

def test_project_viewer():
    view_project(TEST_PATH / "test-project.json")
