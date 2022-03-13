from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from .widgets import MTPropsWidget
    

def start(
    project_file: str | None = None,
    globals_file: str | None = None,
    viewer: "napari.Viewer" = None,
) -> "MTPropsWidget":
    """
    Start napari viewer and dock MTProfiler widget as a dock widget.
    
    Parameters
    ----------
    viewer : napari.Viewer
        Give a viewer object and this viewer will be used as the parent.
    """
    from .widgets import MTPropsWidget
    import logging
    
    if viewer is None:
        import napari
        viewer = napari.Viewer()
    widget = MTPropsWidget()
    
    logger = logging.getLogger(__name__.split(".")[0])
    logger.addHandler(widget.Panels.log)
    formatter = logging.Formatter(fmt="%(levelname)s || %(message)s")
    widget.Panels.log.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    
    dock = viewer.window.add_dock_widget(
        widget,
        area="right",
        allowed_areas=["right"],
        name="MT Profiler"
    )
    dock.setMinimumHeight(300)
    
    if project_file is not None:
        widget.Load_state(project_file)
    if globals_file is not None:
        widget.Others.Global_variables.Load_variables(globals_file)
    return widget
