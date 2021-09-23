from __future__ import annotations
import pandas as pd
import napari
from .widget import MTProfiler
from ._dependencies import impy as ip

def start(viewer:"napari.Viewer"=None):
    if viewer is None:
        viewer = napari.Viewer()
    mtprof = MTProfiler()
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    return mtprof

def load(df:str|pd.DataFrame, 
         img:str|"ip.arrays.LazyImgArray",
         viewer:"napari.Viewer"=None,
         binsize:int=4):
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(img, str):
        img = ip.lazy_imread(img, chunks=(64, 1024, 1024))
    if viewer is None:
        viewer = napari.Viewer()
    mtprof = MTProfiler()
    mtprof._load_image(img, binsize=binsize)
    mtprof._from_dataframe(df)
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    return mtprof