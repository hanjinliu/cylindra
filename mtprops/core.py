from __future__ import annotations
import pandas as pd
import napari
from qtpy.QtWidgets import QFileDialog

from .widget import MTProfiler
from ._impy import impy as ip

def start(viewer:"napari.Viewer", path:str|None=None, binsize=4):
    # open file dialog if path is not specified.
    if path is None:
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window.qt_viewer,
            caption="Select image ...",
            directory=hist[0],
        )
        if filenames != [] and filenames is not None:
            path = filenames[0]
            napari.utils.history.update_open_history(filenames[0])
        else:
            return None
            
    img = ip.lazy_imread(path, chunks=(64, 1024, 1024))
    
    mtprof = MTProfiler(img, viewer=viewer, binsize=binsize)
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    # TODO: viewer.window._status_bar._toggle_activity_dock(True)
    return mtprof

def load(viewer, df:str|pd.DataFrame, img:str|"ip.arrays.LazyImgArray", binsize:int=4):
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(img, str):
        img = ip.lazy_imread(img, chunks=(64, 1024, 1024))
    
    mtprof = MTProfiler(img, viewer=viewer, binsize=binsize)
    mtprof.from_dataframe(df)
    return mtprof