from mtprops import start, MtTomogram
from mtprops.utils import no_verbose
from mtprops.const import H
from pathlib import Path
import napari
import numpy as np
from numpy.testing import assert_allclose

viewer = napari.Viewer()
ui = start(viewer)
coords_13pf = [[18.97, 190.0, 28.99], [18.97, 107.8, 51.48], [18.97, 35.2, 79.90]]

def test_run_all():    
    path = Path(__file__).parent / "13pf_MT.tif"
    ui.load_tomogram(path=path, scale='1.052', bin_size=2, light_background=False,
                     cutoff=0.0, subtomo_length=48.0, subtomo_width=44.0)
    ui.Apply_lowpass_to_reference_image()
    ui.register_path(coords=coords_13pf)
    ui.register_path(coords=coords_13pf[::-1])
    ui.run_mtprops(interval=16.0, ft_size=32.0, n_refine=1, dense_mode=True, dense_mode_sigma=0.2, 
                   local_props=True, global_props=True, paint=True)
    spl = ui.tomogram.splines[0]
    ypitch_mean = spl.localprops[H.yPitch].mean()
    ypitch_glob = spl.globalprops[H.yPitch]
    assert 4.075 < ypitch_glob < 4.105  # GDP-bound microtubule has pitch length in this range
    assert abs(ypitch_glob - ypitch_mean) < 0.013
    assert all(spl.localprops[H.nPF] == 13)
    assert all(spl.localprops[H.riseAngle] > 8.3)
    ui.SplineControl.set_orientation(i=0, orientation="PlusToMinus")
    assert ui.get_current_spline().orientation == "PlusToMinus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "PlusToMinus"
    ui.SplineControl.num = 1
    ui.SplineControl.set_orientation(i=0, orientation="MinusToPlus")
    assert ui.get_current_spline().orientation == "MinusToPlus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    
    ui.SplineControl.num = 1
    assert ui.get_current_spline().orientation == "PlusToMinus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "PlusToMinus"
    
    ui.SplineControl.num = 0
    assert ui.get_current_spline().orientation == "MinusToPlus"
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    
    ui.Align_to_polarity(orientation="MinusToPlus")
    assert ui.GlobalProperties.params.params2.polarity.txt == "MinusToPlus"
    assert ui.LocalProperties.params.pitch.txt == " -- nm"
    
    assert ui.SplineControl.canvas[0].image is not None
    assert ui.LocalProperties.params.pitch.txt == f" {spl.localprops[H.yPitch][1]:.2f} nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == f" {spl.globalprops[H.yPitch]:.2f} nm"

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
    assert ui.SplineControl.canvas[0].image is None
    assert ui.LocalProperties.params.pitch.txt == " -- nm"
    assert ui.GlobalProperties.params.params1.pitch.txt == " -- nm"
