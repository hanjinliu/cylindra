from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence, Union
from contextlib import suppress
from cylindra.const import MoleculesHeader as Mole

if TYPE_CHECKING:
    import napari
    from acryo import Molecules
    from cylindra._molecules_layer import MoleculesLayer
    from cylindra.project.sequence import ProjectSequence
    from cylindra.widgets import CylindraMainWidget
    from cylindra.components import CylSpline
    from cylindra.project import CylindraProject

PathLike = Union[str, Path]
_CURRENT_INSTANCE: CylindraMainWidget | None = None


def start(
    project_file: str | None = None,
    globals_file: str | None = None,
    viewer: napari.Viewer = None,
    *,
    log_level: int | str = "INFO",
) -> CylindraMainWidget:
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
    from cylindra._ipython import install_ipython_magic
    import numpy as np
    import impy as ip
    import polars as pl
    import matplotlib.pyplot as plt
    from magicclass import defaults, logging

    global _CURRENT_INSTANCE

    ui = CylindraMainWidget()

    defaults["macro-highlight"] = True
    defaults["undo-max-history"] = 16
    del defaults

    if viewer is None:
        import napari

        viewer = napari.Viewer()

    # set logger
    logger = logging.getLogger("cylindra")
    formatter = logging.Formatter(fmt="%(levelname)s || %(message)s")
    logger.widget.setFormatter(formatter)
    logger.widget.min_height = 200

    # set log level
    if isinstance(log_level, str):
        log_level = log_level.upper()
        if log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            log_level = getattr(logging, log_level)
        else:
            raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(log_level)

    dock = viewer.window.add_dock_widget(
        ui, area="right", allowed_areas=["right"], name="cylindra"
    )
    dock.setMinimumHeight(300)
    viewer.window.add_dock_widget(logger.widget, name="Log")

    if project_file is not None:
        ui.load_project(project_file)
    if globals_file is not None:
        ui.Others.GlobalVariables.load_variables(globals_file)
    _CURRENT_INSTANCE = ui

    with suppress(Exception):
        # update console namespace
        viewer.window._qt_viewer.console.push(
            {
                ".ui": ui,  # only available from namespace dict
                "ui": ui,
                "np": np,
                "ip": ip,
                "pl": pl,
                "plt": plt,
                "Path": Path,
            }
        )

    install_ipython_magic()
    return ui


def instance() -> CylindraMainWidget | None:
    """Get the current CylindraMainWidget instance."""
    return _CURRENT_INSTANCE


def view_project(project_file: PathLike, run: bool = False):
    """View the Cylindra project file."""
    from cylindra.project import CylindraProject, get_project_json

    widget = CylindraProject.from_json(
        get_project_json(project_file)
    ).make_project_viewer()
    widget.show(run=run)
    return widget


def read_project(file: PathLike) -> CylindraProject:
    """Read the Cylindra project file."""
    from cylindra.project import CylindraProject, get_project_json

    return CylindraProject.from_json(get_project_json(file))


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


def collect_projects(
    files: PathLike | Iterable[PathLike], *, skip_exc: bool = False
) -> ProjectSequence:
    """
    Collect project files into a ProjectSequence object.

    >>> collect_projects("path/to/dir/*.json")

    Parameters
    ----------
    files : path-like or iterable of path-like
        Project file paths or a glob pattern.
    """
    from cylindra.project.sequence import ProjectSequence

    if isinstance(files, (str, Path)) and "*" in str(files):
        import glob

        files = glob.glob(str(files))
    seq = ProjectSequence.from_paths(files, skip_exc=skip_exc)
    return seq


def layer_to_coordinates(layer: MoleculesLayer, npf: int | None = None):
    """Convert point coordinates of a Points layer into a structured array."""
    import impy as ip

    if npf is None:
        npf = layer.molecules.features[Mole.pf].max() + 1
    data = layer.data.reshape(-1, npf, 3)
    data = ip.asarray(data, name=layer.name, axes=["L", "PF", "dim"])
    data.axes["dim"].labels = ("z", "y", "x")
    return data
