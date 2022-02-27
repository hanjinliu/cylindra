import pytest
from mtprops import start, MTPropsWidget
from mtprops.const import H
from pathlib import Path
import napari
import time

viewer = napari.Viewer()
ui = start(viewer)

coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]
coords_14pf = [[21.97, 123.1, 32.98], [21.97, 83.3, 40.5], [21.97, 17.6, 64.96]]


def assert_canvas(ui: MTPropsWidget, isnone):
    for i in range(3):
        if isnone[i]:
            assert ui.SplineControl.canvas[i].image is None, f"{i}-th canvas"
        else:
            assert ui.SplineControl.canvas[i].image is not None, f"{i}-th canvas"

def test_spline_switch():    
    path = Path(__file__).parent / "13pf_MT.tif"
    ui.load_tomogram(path=path, scale='1.052', bin_size=2, light_background=False,
                     cutoff=0.0, subtomo_length=48.0, subtomo_width=44.0)
    ui.Apply_lowpass_to_reference_image()
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
    
    ui.run_mtprops(interval=16.0, ft_size=32.0, n_refine=1, dense_mode=True, dense_mode_sigma=0.2, 
                   local_props=True, global_props=True, paint=True)
    
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
    assert ui.LocalProperties.params.pitch.txt == f" -- nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {ui.get_current_spline().globalprops[H.yPitch]:.2f} nm"
    
    ui.SplineControl.num = 1
    assert ui.SplineControl.pos == 1
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    assert ui.LocalProperties.params.pitch.txt == f" {ui.get_current_spline().localprops[H.yPitch][1]:.2f} nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {ui.get_current_spline().globalprops[H.yPitch]:.2f} nm"
    
    assert_canvas(ui, [False, False, False])

    # map monomer coordinates and save them.
    ui.Map_monomers(splines=[0])
    ui.Average_subset(ui.parent_viewer.layers['Monomers-0'], size=18.)
    ui.Calculate_FSC(ui.parent_viewer.layers['Monomers-0'], mask_params=None, shape=(18., 18., 18.),
                     seed=0, interpolation=1)
    ui.Save_monomer_coordinates(save_path=Path(__file__).parent/"monomer_coords.txt", 
                                layer=viewer.layers['Monomers-0'], separator=",", unit="pixel")
    ui.Save_monomer_angles(save_path=Path(__file__).parent/"monomer_angles.txt",
                           layer=viewer.layers['Monomers-0'], rotation_axes="ZXZ", in_degree=True, separator=",")
    ui.clear_all()
    assert_canvas(ui, [True, True, True])
    assert ui.LocalProperties.params.pitch.txt == " -- nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == " -- nm"


def test_many_tomograms():
    path = Path(__file__).parent / "13pf_MT.tif"
    ui.load_tomogram(path=path, scale='1.052', bin_size=2, light_background=False,
                     cutoff=0.0, subtomo_length=48.0, subtomo_width=44.0)
    ui.register_path(coords=coords_13pf)
    ui.run_mtprops()
    assert_canvas(ui, [False, False, False])
    spl0 = ui.get_current_spline()
    
    path = Path(__file__).parent / "14pf_MT.tif"
    ui.load_tomogram(path=path, scale='1.052', bin_size=2, light_background=False,
                     cutoff=0.0, subtomo_length=48.0, subtomo_width=44.0)
    assert_canvas(ui, [True, True, True])
    ui.register_path(coords=coords_14pf)
    ui.run_mtprops()
    assert_canvas(ui, [False, False, False])
    spl1 = ui.get_current_spline()
    
    ui._TomogramList.Load(0)
    time.sleep(0.2)  # canvas update needs some time
    assert_canvas(ui, [False, False, False])
    assert ui.LocalProperties.params.pitch.txt == f" {spl0.localprops[H.yPitch][0]:.2f} nm"
    ui._TomogramList.Load(1)
    time.sleep(0.2)  # canvas update needs some time
    assert_canvas(ui, [False, False, False])
    assert ui.LocalProperties.params.pitch.txt == f" {spl1.localprops[H.yPitch][0]:.2f} nm"
    
    with pytest.raises(Exception):
        ui._TomogramList.Delete(1)
    ui._TomogramList.Delete(0)