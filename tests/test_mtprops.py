from mtprops import start

ui = start()

def test_all():    
    ui._loader.call(path=r'C:\Users\liuha\Desktop\Analysis\python_codes\MTProps\tests\test_image.tif', scale='1.052', bin_size=2, light_background=False, cutoff_freq=0.2, subtomo_length=48.0, subtomo_width=44.0, use_lowpass=False)
    ui.Apply_lowpass_to_reference_image()
    ui.register_path(coords=[[18.974, 189.9208736484154, 28.987283758928427], [18.974, 107.83000259799333, 51.48333976793832], [18.974, 35.21115513031228, 79.89941051616134]])
    ui.run_for_all_path(interval=32.0, ft_size=32.0, n_refine=1, dense_mode=True)
    