from __future__ import annotations
import pandas as pd
import napari
from .widget import MTProfiler
import impy as ip

def start(viewer:"napari.Viewer"=None):
    if viewer is None:
        ip.gui.start()
        viewer = ip.gui.viewer
    mtprof = MTProfiler()
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    return mtprof

def load(df:str|pd.DataFrame, 
         img:str,
         viewer:"napari.Viewer"=None
         ):
    if isinstance(df, str):
        df = pd.read_csv(df)
    if viewer is None:
        ip.gui.start()
        viewer = ip.gui.viewer
    mtprof = MTProfiler()
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    mtprof._loader._imread(img)
    mtprof.load_image()
    mtprof._from_dataframe(df)
    return mtprof