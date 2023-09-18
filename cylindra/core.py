from __future__ import annotations

import glob
from pathlib import Path
from weakref import WeakSet
from typing import TYPE_CHECKING, Iterable, Literal, Sequence, Union, overload
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
    ui.macro.options.syntax_highlight = False

    if project_file is not None:
        ui.load_project(project_file)
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


@overload
def instance(create: Literal[False] = False) -> CylindraMainWidget | None:
    ...


@overload
def instance(create: Literal[True]) -> CylindraMainWidget:
    ...


def instance(create: bool = False) -> CylindraMainWidget | None:
    """Get the current CylindraMainWidget instance."""
    ins = _CURRENT_INSTANCE
    if ins is None and create:
        ins = start()
    return ins


def view_project(project_file: PathLike, run: bool = False):
    """View the Cylindra project file."""
    from cylindra.project import CylindraProject

    widget = CylindraProject.from_file(project_file).make_project_viewer()
    widget.show(run=run)
    _ACTIVE_WIDGETS.add(widget)
    return widget


def read_project(file: PathLike) -> CylindraProject:
    """Read the Cylindra project file."""
    from cylindra.project import CylindraProject

    return CylindraProject.from_file(file)


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
        Project file paths or glob pattern(s).
    """
    from cylindra.project import ProjectSequence

    if isinstance(files, (str, Path)):
        if "*" in str(files):
            _files = glob.glob(str(files))
        else:
            if not Path(files).exists():
                raise FileNotFoundError(f"File not found: {files}")
            _files = [files]
    elif hasattr(files, "__iter__"):
        _files = []
        for f in files:
            f = str(f)
            if "*" not in f:
                _files.append(f)
            else:
                _files.extend([_f for _f in glob.glob(f)])
    else:
        raise TypeError(f"files must be path or iterable of paths, got {type(files)}")
    if len(_files) == 0:
        raise FileNotFoundError(f"No project files found from the input {files!r}.")
    seq = ProjectSequence.from_paths(_files, skip_exc=skip_exc)
    return seq
