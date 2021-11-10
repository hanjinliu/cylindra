from __future__ import annotations
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
