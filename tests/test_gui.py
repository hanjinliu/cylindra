from mtprops import start
from pathlib import Path
ui = start()

def test_run_all():    
    path = Path(__file__).parent / "test_image.tif"
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
    ypitch_mean = ui.active_tomogram.splines[0].localprops["yPitch"].mean()
    ypitch_glob = ui.active_tomogram.splines[0].globalprops["yPitch"]
    assert 4.09 < ypitch_glob < 4.11
    assert abs(ypitch_glob - ypitch_mean) < 0.02
    

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