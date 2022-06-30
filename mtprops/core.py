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
    
    ui = MTPropsWidget()
    
    if viewer is None:
        import napari
        viewer = napari.Viewer()
    
    logger = logging.getLogger(__name__.split(".")[0])
    logger.addHandler(ui.log)
    formatter = logging.Formatter(fmt="%(levelname)s || %(message)s")
    ui.log.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    
    dock = viewer.window.add_dock_widget(
        ui,
        area="right",
        allowed_areas=["right"],
        name="MTProps"
    )
    dock.setMinimumHeight(300)
    viewer.window.add_dock_widget(ui._LoggerWindow)
    
    if project_file is not None:
        ui.load_project(project_file)
    if globals_file is not None:
        ui.Others.Global_variables.load_variables(globals_file)
    return ui
