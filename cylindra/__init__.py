import sys

if sys.platform == "darwin":
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)

    del QCoreApplication, Qt

from cylindra.core import (
    start,
    instance,
    view_project,
    read_project,
    read_molecules,
    read_spline,
    collect_projects,
    collect_molecules,
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

del import_metadata, init_config

__all__ = [
    "start",
    "instance",
    "view_project",
    "read_project",
    "read_molecules",
    "read_spline",
    "collect_projects",
    "collect_molecules",
]
