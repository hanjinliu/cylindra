from __future__ import annotations

import glob
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Sequence, overload
from weakref import WeakSet

if TYPE_CHECKING:
    import napari
    from acryo import Molecules
    from magicgui.widgets import Widget

    from cylindra.components import CylSpline
    from cylindra.project import CylindraProject, ProjectSequence
    from cylindra.widgets import CylindraMainWidget

    PathLike = str | Path

_CURRENT_INSTANCE: CylindraMainWidget | None = None
ACTIVE_WIDGETS: WeakSet[Widget] = WeakSet()


def start(
    project_file: str | None = None,
    viewer: napari.Viewer | None = None,
    *,
    log_level: int | str = "INFO",
    headless: bool = False,
    add_main_widget: bool = True,
    run: bool = True,
) -> CylindraMainWidget:
    """
    Start napari viewer and dock cylindra widget as a dock widget.

    Parameters
    ----------
    project_file : path-like, optional
        If given, load the project file.
    viewer : napari.Viewer
        Give a viewer object and this viewer will be used as the parent.
    log_level : int or str, default "INFO"
        Log level. See `logging` module for details.
    headless : bool, default False
        If True, do not show the viewer.
    """
    from cylindra.widgets import CylindraMainWidget  # noqa: I001
    import impy as ip
    import matplotlib.pyplot as plt
    import napari
    import numpy as np
    import polars as pl
    from IPython import get_ipython
    from magicclass import logging

    from cylindra._config import init_config

    global _CURRENT_INSTANCE

    if viewer is None:
        viewer = napari.Viewer(show=not headless)
    elif not isinstance(viewer, napari.Viewer):
        raise TypeError(f"viewer must be a napari.Viewer object, got {type(viewer)}")

    init_config()
    ui = CylindraMainWidget()
    ui.macro.options.max_undo = 16
    ACTIVE_WIDGETS.add(ui)

    # set logger
    logger = logging.getLogger("cylindra")
    formatter = logging.Formatter(fmt="%(levelname)s || %(message)s")
    logger.widget.setFormatter(formatter)
    logger.widget.min_height = 150

    # set log level
    if isinstance(log_level, str):
        log_level = log_level.upper()
        if log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            log_level = getattr(logging, log_level)
        else:
            raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(log_level)

    # set polars display options
    pl.Config().set_tbl_width_chars(120)

    if add_main_widget:
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

    with suppress(Exception):  # This block uses private API.
        # napari viewer does not disconnect layer events when the viewer is closed,
        # so we need to do it manually
        @viewer.window._qt_window.destroyed.connect
        def _on_destroy():
            viewer.layers.events.removing.disconnect()
            viewer.layers.events.removed.disconnect()

        # napari-console disables calltips by default. It's better to enable it.
        viewer.window._qt_viewer.console.enable_calltips = True

    ui.show(run=run)
    if add_main_widget:
        try:  # Just in case
            # avoid accidentally closing/hiding the dock widget
            dock.title.close_button.disconnect()
            dock.title.hide_button.disconnect()
        except Exception:  # pragma: no cover
            print("Failed to disconnect the close/hide button of the dock widget.")

    # Programmatically run `%matplotlib inline` magic
    if ipy := get_ipython():
        ipy.run_line_magic("matplotlib", "inline")

    return ui


def start_as_plugin(run: bool = True):
    """Start Cylindra as a napari plugin"""
    import napari
    from magicclass import logging

    ui = start(
        viewer=napari.current_viewer(),
        add_main_widget=False,
        run=run,
    )
    # float logger widget
    logger = logging.getLogger("cylindra")
    logger.widget.native.parentWidget().setFloating(True)
    logger.widget.height = 160
    return ui


# fmt: off
@overload
def instance(create: Literal[False] = False) -> CylindraMainWidget | None: ...
@overload
def instance(create: Literal[True]) -> CylindraMainWidget: ...
# fmt: on


def instance(create=False):
    """Get the current CylindraMainWidget instance."""
    ins = _CURRENT_INSTANCE
    if ins is None and create:
        ins = start()
    return ins


def view_project(project_file: PathLike, show: bool = True):
    """View the Cylindra project file."""
    from cylindra.project import CylindraProject

    widget = CylindraProject.from_file(project_file).make_project_viewer()
    if show:
        widget.show(run=False)
    ACTIVE_WIDGETS.add(widget)
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
    pos_cols : sequence of str, default ("z", "y", "x")
        Column names for the molecule positions.
    rot_cols : sequence of str, default ("zvec", "yvec", "xvec")
        Column names for the molecule rotation vectors.

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
    files: PathLike | Iterable[PathLike],
    *,
    skip_exc: bool = False,
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
                _files.extend(list(glob.glob(f)))
    else:
        raise TypeError(f"files must be path or iterable of paths, got {type(files)}")
    if len(_files) == 0:
        raise FileNotFoundError(f"No project files found from the input {files!r}.")
    seq = ProjectSequence.from_paths(_files, check_scale=False, skip_exc=skip_exc)
    return seq
