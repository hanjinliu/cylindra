from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from cylindra.widgets import CylindraMainWidget

_CURRENT_INSTANCE: CylindraMainWidget | None = None

def start(
    project_file: str | None = None,
    globals_file: str | None = None,
    viewer: "napari.Viewer" = None,
) -> "CylindraMainWidget":
    """
    Start napari viewer and dock cylindra widget as a dock widget.
    
    Parameters
    ----------
    project_file : path-like, optional
        If given, load the project file.
    globals_file : path-like, optional
        If given, load the global variable file.
    viewer : napari.Viewer
        Give a viewer object and this viewer will be used as the parent.
    """
    from cylindra.widgets import CylindraMainWidget
    import logging
    
    global _CURRENT_INSTANCE
    
    ui = CylindraMainWidget()
    
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
        name="cylindra"
    )
    dock.setMinimumHeight(300)
    viewer.window.add_dock_widget(ui._LoggerWindow)
    
    if project_file is not None:
        ui.load_project(project_file)
    if globals_file is not None:
        ui.Others.Global_variables.load_variables(globals_file)
    _CURRENT_INSTANCE = ui
    return ui

def instance() -> CylindraMainWidget | None:
    """Get the current CylindraMainWidget instance."""
    return _CURRENT_INSTANCE

def view_project(project_file, run=False) -> None:
    """View the Cylindra project file."""
    from cylindra.project import CylindraProject
    
    return CylindraProject.from_json(project_file).make_project_viewer().show(run=run)
