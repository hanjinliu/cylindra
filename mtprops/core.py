from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from .widgets import MTPropsWidget
    

def start(viewer: "napari.Viewer" = None) -> "MTPropsWidget":
    """
    Start napari viewer and dock MTProfiler widget as a dock widget.
    By default impy's viewer is used.
    """
    from .widgets import MTPropsWidget
    import impy as ip
    
    if viewer is None:
        ip.gui.start()
        viewer = ip.gui.viewer
    mtprof = MTPropsWidget()
    dock = viewer.window.add_dock_widget(mtprof, area="right", allowed_areas=["right"],
                                         name="MT Profiler")
    dock.setMinimumHeight(300)
    return mtprof
