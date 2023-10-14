import sys


def dont_use_native_menu_bar():
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
    return None


def init_opengl_and_dpi():
    from qtpy import QtCore, QtWidgets as QtW

    # Docking vispy widget in napari viewer requires this.
    QtW.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    # High DPI support for High-DPI devices such as Surface Pro.
    QtW.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    return None


if sys.platform == "darwin":
    dont_use_native_menu_bar()

init_opengl_and_dpi()

from cylindra.core import (
    start,
    instance,
    view_project,
    read_project,
    read_molecules,
    read_spline,
    collect_projects,
)


def import_metadata():
    from importlib.metadata import PackageNotFoundError, version, metadata

    try:
        _version = version("cylindra")
    except PackageNotFoundError:
        _version = "uninstalled"
    _author = metadata("cylindra")["Author"]
    _email = metadata("cylindra")["Author-email"]
    return _version, _author, _email


__version__, __author__, __email__ = import_metadata()

from cylindra._config import init_config

init_config()

del import_metadata, init_config, dont_use_native_menu_bar, init_opengl_and_dpi

__all__ = [
    "start",
    "instance",
    "view_project",
    "read_project",
    "read_molecules",
    "read_spline",
    "collect_projects",
]
