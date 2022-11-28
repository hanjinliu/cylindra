from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

if TYPE_CHECKING:
    import napari
    from acryo import Molecules
    from cylindra.widgets import CylindraMainWidget
    from cylindra.components import CylSpline
    from cylindra.project import CylindraProject

PathLike = Union[str, Path]
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

def view_project(project_file: PathLike, run: bool = False) -> None:
    """View the Cylindra project file."""
    from cylindra.project import CylindraProject
    
    return CylindraProject.from_json(project_file).make_project_viewer().show(run=run)

def read_project(file: PathLike) -> CylindraProject:
    """Read the Cylindra project file."""
    from cylindra.project import CylindraProject
    
    return CylindraProject.from_json(file)

def read_molecules(
    file: PathLike,
    pos_cols: Sequence[str] = ("z", "y", "x"),
    rot_cols: Sequence[str] = ("zvec", "yvec", "xvec"),
    **kwargs,
) -> Molecules:
    """
    Read a molecules CSV file.
    
    Parameters
    ----------
    file : PathLike
        File path.
    pos_cols : sequence of str, default is ("z", "y", "x")
        Column names for the molecule positions.
    rot_cols : sequence of str, default is ("zvec", "yvec", "xvec")
        Column names for the molecule rotation vectors.
    **kwargs
        Keyword arguments to be passed to `pd.read_csv`.
    
    Returns
    -------
    Molecules
        Molecules object.
    """
    from acryo import Molecules
    
    return Molecules.from_csv(
        file, pos_cols=list(pos_cols), rot_cols=list(rot_cols), **kwargs
    )

def read_spline(file: PathLike) -> CylSpline:
    """
    Read the spline file.

    Parameters
    ----------
    file : PathLike
        File path.

    Returns
    -------
    CylSpline
        CylSpline object.
    """
    from cylindra.components import CylSpline
    
    return CylSpline.from_json(file)


def read_localprops(file: PathLike):
    """
    Read a local-property file as a `DataFrameDict`.

    Parameters
    ----------
    file : PathLike
        File path.

    Returns
    -------
    DataFrameDict
        Dictionary of data frames.
    """
    from cylindra._list import DataFrameList
    
    if Path(file).is_dir():
        return DataFrameList.glob_csv(file)
    return DataFrameList.from_csv(file)
