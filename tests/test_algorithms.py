from mtprops import start
from mtprops.const import H
from pathlib import Path
ui = start()

def test_run_all():    
    path = Path(__file__).parent / "13pf_MT.tif"
    ui._loader.call(path=path,
                    scale='1.052', 
                    bin_size=2,
                    light_background=False,
                    cutoff_freq=0.2,
                    subtomo_length=48.0,
                    subtomo_width=44.0,
                    use_lowpass=False
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
    assert 4.075 < ypitch_glob < 4.105 # GDP-bound microtubule has pitch length in this range
    assert abs(ypitch_glob - ypitch_mean) < 0.013
    assert all(spl.localprops[H.nPF] == 13)
    assert all(spl.localprops[H.riseAngle] > 8.3)
    
    path = Path(__file__).parent / "14pf_MT.tif"
    ui._loader.call(path=path, 
                    scale='1.052', 
                    bin_size=1, 
                    light_background=False,
                    cutoff_freq=0.2,
                    subtomo_length=48.0,
                    subtomo_width=44.0,
                    use_lowpass=False
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
    

def test_viewing():
    ...

def test_result_io():
    ...
    
def test_picking():
    ...
    
def test_intensity_scalability():
    ...

def test_image_size_scalability():
    ...
    
def test_image_scale_scalability():
    ...
    
def test_spline_robustness():
    ...