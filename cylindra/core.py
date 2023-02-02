from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence, Union
import numpy as np
import polars as pl

if TYPE_CHECKING:
    import napari
    from acryo import Molecules, TomogramCollection
    from cylindra.widgets import CylindraMainWidget
    from cylindra.components import CylSpline
    from cylindra.project import CylindraProject

PathLike = Union[str, Path]
_CURRENT_INSTANCE: CylindraMainWidget | None = None

def start(
    project_file: str | None = None,
    globals_file: str | None = None,
    viewer: "napari.Viewer" = None,
    *,
    log_level: int | str = "INFO",
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
    
    logger = logging.getLogger("cylindra")
    logger.addHandler(ui.log)
    formatter = logging.Formatter(fmt="%(levelname)s || %(message)s")
    ui.log.setFormatter(formatter)
    
    # set log level
    if isinstance(log_level, str):
        log_level = log_level.upper()
        if log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            log_level = getattr(logging, log_level)
        else:
            raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(log_level)
    
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
    
    try:
        viewer.window._qt_viewer.console.push({"ui": ui})
    except Exception as e:
        print(e)
        pass
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
    
    path = Path(file)
    return Molecules.from_csv(
        path, pos_cols=list(pos_cols), rot_cols=list(rot_cols), **kwargs
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
    Read local-property file(s) as a `DataFrameList`.

    Parameters
    ----------
    file : PathLike
        File path.

    Returns
    -------
    DataFrameList
        Dictionary of data frames.
    """
    from cylindra._list import DataFrameList
    
    if Path(file).is_dir():
        return DataFrameList.glob_csv(file)
    return DataFrameList.from_csv(file)

def read_globalprops(file: PathLike):
    """
    Read a local-property file as a `DataFrameList`.

    Parameters
    ----------
    file : PathLike
        File path.

    Returns
    -------
    DataFrameList
        Dictionary of data frames.
    """
    import pandas as pd
    
    path = Path(file)
    if path.is_dir():
        dfs: list[pd.DataFrame] = []
        for p in path.glob("**/globalprops.csv"):
            df = pd.read_csv(p, index_col=0)
            dfs.append(df)
        if len(dfs) == 0:
            raise FileNotFoundError(f"No globalprops.csv file found under {path}.")
        return pd.concat(dfs, axis=0, ignore_index=True)
    else:
        return pd.read_csv(path, index_col=0)

def collect_tomograms(files: Iterable[PathLike], order: int = 3) -> TomogramCollection:
    import impy as ip
    from acryo import TomogramCollection
    from cylindra.const import GlobalVariables as GVar
    
    col = TomogramCollection(order=order)
    need_initialize = True

    for file in files:
        project = read_project(file)
        tomogram = ip.lazy_imread(project.image, chunks=GVar.daskChunk)
        if need_initialize:
            template = ip.imread(project.template_image)
            col = col.replace(output_shape=template.shape, scale=tomogram.scale.x)
            need_initialize = False

        for fp in project.molecules:
            col.add_tomogram(tomogram, molecules=read_molecules(fp))

    return col
