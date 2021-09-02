from __future__ import annotations
import pandas as pd
import napari
from .widget import MTProfiler
from ._impy import impy as ip

def start(viewer:"napari.Viewer", 
          interval_nm:float=24,
          light_background:bool=True
          ):
    mtprof = MTProfiler(viewer=viewer,
                        interval_nm=interval_nm,
                        light_background=light_background
                        )
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    return mtprof

def load(viewer:"napari.Viewer",
         df:str|pd.DataFrame, 
         img:str|"ip.arrays.LazyImgArray",
         binsize:int=4,
         interval_nm:float=33.4,
         light_background:bool=True
         ):
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(img, str):
        img = ip.lazy_imread(img, chunks=(64, 1024, 1024))

    mtprof = MTProfiler(viewer=viewer,
                        interval_nm=interval_nm,
                        light_background=light_background
                        )
    mtprof.load_image(img, binsize=binsize)
    mtprof.from_dataframe(df)
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    return mtprof