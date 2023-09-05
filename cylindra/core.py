from __future__ import annotations

import glob
from pathlib import Path
from weakref import WeakSet
from typing import TYPE_CHECKING, Iterable, Sequence, Union
from contextlib import suppress

if TYPE_CHECKING:
    import napari
    from magicgui.widgets import Widget
    from acryo import Molecules
    from cylindra.project import ProjectSequence
    from cylindra.widgets import CylindraMainWidget
    from cylindra.components import CylSpline
    from cylindra.project import CylindraProject

PathLike = Union[str, Path]
_CURRENT_INSTANCE: CylindraMainWidget | None = None
_ACTIVE_WIDGETS: WeakSet[Widget] = WeakSet()


def start(
    project_file: str | None = None,
    globals_file: str | None = None,
    viewer: napari.Viewer | None = None,
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
    import numpy as np
    import impy as ip
    import polars as pl
    import matplotlib.pyplot as plt
    from magicclass import logging
    import napari
    from IPython import get_ipython

    global _CURRENT_INSTANCE

    if viewer is None:
        viewer = napari.Viewer()
    elif not isinstance(viewer, napari.Viewer):
        raise TypeError(f"viewer must be a napari.Viewer object, got {type(viewer)}")

    ui = CylindraMainWidget()
    ui.macro.options.max_undo = 16
    _ACTIVE_WIDGETS.add(ui)

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
    ui.macro.options.syntax_highlight = True

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

    try:
        # napari viewer does not disconnect layer events when the viewer is closed,
        # so we need to do it manually
        @viewer.window._qt_window.destroyed.connect
        def _on_destroy():
            viewer.layers.events.removing.disconnect()
            viewer.layers.events.removed.disconnect()

        # napari-console disables calltips by default. It's better to enable it.
        viewer.window._qt_viewer.console.enable_calltips = True

    except Exception:
        # since it uses private API, it may break in the future
        pass

    # Programmatically run `%matplotlib inline` magic
    ipy = get_ipython()
    ipy.run_line_magic("matplotlib", "inline")

    ui.show()
    return ui


def instance(create: bool = False) -> CylindraMainWidget | None:
    """Get the current CylindraMainWidget instance."""
    ins = _CURRENT_INSTANCE
    if ins is None and create:
        ins = start()
    return ins


def view_project(project_file: PathLike, run: bool = False):
    """View the Cylindra project file."""
    from cylindra.project import CylindraProject, get_project_file

    widget = CylindraProject.from_json(
        get_project_file(project_file)
    ).make_project_viewer()
    widget.show(run=run)
    _ACTIVE_WIDGETS.add(widget)
    return widget


def read_project(file: PathLike) -> CylindraProject:
    """Read the Cylindra project file."""
    from cylindra.project import CylindraProject, get_project_file

    return CylindraProject.from_json(get_project_file(file))


def read_molecules(
    file: PathLike,
    pos_cols: Sequence[str] = ("z", "y", "x"),
    rot_cols: Sequence[str] = ("zvec", "yvec", "xvec"),
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
    return Molecules.from_file(path, pos_cols=list(pos_cols), rot_cols=list(rot_cols))


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
    from cylindra.project import ProjectSequence, get_project_file

    if isinstance(files, (str, Path)):
        if "*" in str(files):
            files = glob.glob(str(files))
        else:
            files = [get_project_file(files)]
    if hasattr(files, "__iter__"):
        files = [get_project_file(f) for f in files]
    else:
        raise TypeError(f"files must be path or iterable of paths, got {type(files)}")
    if len(files) == 0:
        raise ValueError(f"No project files found. Please check the input paths.")
    seq = ProjectSequence.from_paths(files, skip_exc=skip_exc)
    return seq


def collect_molecules(
    files: PathLike | Iterable[PathLike],
    file_id_column: str = "file-id",
    **kwargs,
) -> Molecules:
    """
    Collect molecules from multiple files into a single Molecules object.

    To distinguish molecules from different files, a new column named `file-id` will be added to the
    feature data frame of the output Molecules object.

    Parameters
    ----------
    files : path-like or iterable of path-like
        Input path(s). Can be a glob pattern.
    file_id_column : str, default is "file-id"
        Column name used to specify file identifier.

    Returns
    -------
    Molecules
        Concatenated molecules.
    """
    import polars as pl
    from acryo import Molecules

    if isinstance(files, (str, Path)):
        if "*" in str(files):
            files = glob.glob(str(files))
    molecules = list[Molecules]()
    for i, f in enumerate(files):
        mole = Molecules.from_file(f, **kwargs)
        molecules.append(
            mole.with_features(pl.repeat(i, len(mole)).alias(file_id_column))
        )
    if len(molecules) == 0:
        raise ValueError(f"No molecules found. Please check the input paths.")
    return Molecules.concat(molecules)
