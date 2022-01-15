from mtprops import start, MtTomogram
from mtprops.const import H
from pathlib import Path
import napari
import numpy as np
from numpy.testing import assert_allclose

viewer = napari.Viewer()
ui = start(viewer)

def test_run_all():    
    path = Path(__file__).parent / "13pf_MT.tif"
    ui.call(path=path,
            scale='1.052', 
            bin_size=2,
            light_background=False,
            cutoff=0.0,
            subtomo_length=48.0,
            subtomo_width=44.0
            )
    ui.Apply_lowpass_to_reference_image()
    ui.register_path(coords=[[18.97, 190.0, 28.99], 
                             [18.97, 107.8, 51.48],
                             [18.97,  35.2, 79.90]])
    ui.run_for_all_path(interval=16.0, ft_size=32.0, n_refine=1, dense_mode=True)
    ui.Global_FT_analysis()
    spl = ui.active_tomogram.splines[0]
    ypitch_mean = spl.localprops[H.yPitch].mean()
    ypitch_glob = spl.globalprops[H.yPitch]
    assert 4.075 < ypitch_glob < 4.105  # GDP-bound microtubule has pitch length in this range
    assert abs(ypitch_glob - ypitch_mean) < 0.013
    assert all(spl.localprops[H.nPF] == 13)
    assert all(spl.localprops[H.riseAngle] > 8.3)
    
    path = Path(__file__).parent / "14pf_MT.tif"
    ui.call(path=path, 
            scale='1.052', 
            bin_size=1, 
            light_background=False,
            cutoff=0.0,
            subtomo_length=48.0,
            subtomo_width=44.0
            )
    ui.register_path(coords=[[21.97, 123.1, 32.98],
                             [21.97, 83.3, 40.5],
                             [21.97, 17.6, 64.96]])
    ui.run_for_all_path(interval=16.0, ft_size=32.0, n_refine=1, dense_mode=True)
    ui.Global_FT_analysis()
    spl = ui.active_tomogram.splines[0]
    ypitch_mean = spl.localprops[H.yPitch].mean()
    ypitch_glob = spl.globalprops[H.yPitch]
    assert 4.075 < ypitch_glob < 4.105 # GDP-bound microtubule has pitch length in this range
    assert abs(ypitch_glob - ypitch_mean) < 0.013
    assert all(spl.localprops[H.nPF] == 14)
    assert all(spl.localprops[H.riseAngle] > 7.5)
    assert spl.globalprops[H.skewAngle] < -0.25 # 14-pf MT has negative skew (Atherton et al., 2019)
    

def test_result_io():
    path = Path(__file__).parent / "14pf_MT.tif"
    save_path = Path(__file__).parent / "result.json"
    
    tomo = MtTomogram.imread(path, light_background=False)
    tomo.add_spline(np.array([[21.97, 123.1, 32.98],
                              [21.97, 83.3, 40.5],
                              [21.97, 17.6, 64.96]]))
    tomo.fit()
    tomo.refine()
    tomo.save_json(save_path)
    tomo2 = MtTomogram()
    tomo2.load_json(save_path)
    assert tomo2.splines[0] == tomo.splines[0]
    assert tomo2.splines[0].localprops is None
    assert tomo2.splines[0].globalprops is None
    
    tomo.measure_radius()
    tomo.local_ft_params()
    tomo.save_json(save_path)
    tomo2 = MtTomogram()
    tomo2.load_json(save_path)
    assert_allclose(tomo.collect_anchor_coords(0), tomo2.collect_anchor_coords(0))
    assert_allclose(tomo.collect_localprops(0), tomo2.collect_localprops(0))
    assert tomo2.splines[0].globalprops is None
    
    tomo.global_ft_params()
    tomo.save_json(save_path)
    tomo2 = MtTomogram()
    tomo2.load_json(save_path)
    assert_allclose(tomo.collect_anchor_coords(0), tomo2.collect_anchor_coords(0))
    assert_allclose(tomo.collect_localprops(0), tomo2.collect_localprops(0))
    assert_allclose(tomo.collect_radii(0), tomo2.collect_radii(0))
    assert_allclose(tomo.global_ft_params(0), tomo2.global_ft_params(0))
    
    
def test_intensity_scalability():
    ...

def test_image_size_scalability():
    ...
    
def test_image_scale_scalability():
    ...
    
def test_spline_robustness():
    ...