import pytest
from mtprops import start, MTPropsWidget
from mtprops.components import Molecules
from mtprops.const import H
from pathlib import Path
from numpy.testing import assert_allclose


coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]]


def assert_canvas(ui: MTPropsWidget, isnone):
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
    

def test_spline_switch():
    ui = start()
    path = Path(__file__).parent / "13pf_MT.tif"
    ui.Open_image(path=path, scale=1.052, bin_size=2)
    ui.Filter_reference_image()
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    
    # check canvas is updated correctly
    ui.Add_anchors(interval=15.0)
    assert_canvas(ui, [True, True, True])
    ui.Sample_subtomograms()
    assert_canvas(ui, [False, False, True])
    ui.SplineControl.num = 1
    assert_canvas(ui, [False, False, True])
    ui.SplineControl.num = 0
    assert_canvas(ui, [False, False, True])
    ui.SplineControl.pos = 1
    assert_canvas(ui, [False, False, True])
    ui.SplineControl.pos = 0
    assert_canvas(ui, [False, False, True])
    
    ui.run_mtprops(interval=16.0)
    
    # check results
    spl = ui.tomogram.splines[0]
    ypitch_mean = spl.localprops[H.yPitch].mean()
    ypitch_glob = spl.globalprops[H.yPitch]
    assert 4.075 < ypitch_glob < 4.105  # GDP-bound microtubule has pitch length in this range
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
    assert ui.get_current_spline().orientation == "PlusToMinus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "PlusToMinus"
    ui.SplineControl.num = 1
    ui.SplineControl.set_orientation(i=1, orientation="MinusToPlus")
    assert ui.get_current_spline().orientation == "MinusToPlus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    
    ui.SplineControl.num = 0
    assert ui.get_current_spline().orientation == "PlusToMinus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "PlusToMinus"
    ui.SplineControl.num = 1
    assert ui.get_current_spline().orientation == "MinusToPlus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    assert_canvas(ui, [False, False, False])
    
    # Check align polarity.
    # Only spline 0 will get updated.
    ui.Align_to_polarity(orientation="MinusToPlus")
    ui.SplineControl.num = 0
    ui.SplineControl.pos = 1
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    assert ui.LocalProperties.params.pitch.txt == f" {ui.get_current_spline().localprops[H.yPitch].values[1]:.2f} nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {ui.get_current_spline().globalprops[H.yPitch]:.2f} nm"
    
    ui.SplineControl.num = 1
    assert ui.SplineControl.pos == 1
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    assert ui.LocalProperties.params.pitch.txt == f" {ui.get_current_spline().localprops[H.yPitch].values[1]:.2f} nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {ui.get_current_spline().globalprops[H.yPitch]:.2f} nm"
    
    assert_canvas(ui, [False, False, False])

    # map monomer coordinates and save them.
    ui.Map_monomers(splines=[0])
    ui.Average_subset(ui.parent_viewer.layers['Mono-0'], size=18.)
    ui.Calculate_FSC(ui.parent_viewer.layers['Mono-0'], mask_params=None, shape=(18., 18., 18.),
                     seed=0, interpolation=1)
    template_path = Path(__file__).parent / "template.mrc"
    ui.Align_averaged(layer=ui.parent_viewer.layers['Mono-0'], template_path=template_path, 
                      mask_params=(1, 1), chunk_size=78)
    ui.Align_all(layer=ui.parent_viewer.layers['Mono-0'], template_path=template_path, mask_params=(1, 1), 
                 max_shifts=(1.0, 1.1, 1.0), y_rotation=(1.0, 1.0), chunk_size=78,)
    ui.Seam_search(layer=ui.parent_viewer.layers['Mono-0'], template_path=template_path, mask_params=(1, 1),
                   chunk_size=78)
    ui.Save_molecules(layer=ui.parent_viewer.layers['Mono-0'],
                      save_path=Path(__file__).parent/"monomers.txt"
                      )
    mole = ui.get_molecules('Mono-0')
    ui.clear_all()
    ui.Load_molecules(Path(__file__).parent/"monomers.txt")
    mole_read = ui.get_molecules('monomers')
    assert_molecule_equal(mole, mole_read)
    assert_canvas(ui, [True, True, True])
    assert ui.LocalProperties.params.pitch.txt == " -- nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == " -- nm"


def test_io():
    ui = start()
    path = Path(__file__).parent / "13pf_MT.tif"
    ui.Open_image(path=path, scale=1.052, bin_size=1)
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    ui.run_mtprops(interval=24.0)
    ui.Map_monomers(splines=[0, 1])
    
    # Save project
    old_splines = ui.tomogram.splines.copy()
    old_molecules = [ui.get_molecules('Mono-0'), ui.get_molecules('Mono-1')]
    ui.Save_project(Path(__file__).parent / "test-project.json")
    ui.Load_project(Path(__file__).parent / "test-project.json")
    new_splines = ui.tomogram.splines
    new_molecules = [ui.get_molecules('Mono-0'), ui.get_molecules('Mono-1')]
    assert old_splines[0] == new_splines[0]
    assert old_splines[1] == new_splines[1]
    for mol0, mol1 in zip(old_molecules, new_molecules):
        assert_molecule_equal(mol0, mol1)
    