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
)

from importlib.metadata import PackageNotFoundError, version, metadata

try:
    __version__ = version("cylindra")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = metadata("cylindra")["Author"]
__email__ = metadata("cylindra")["Author-email"]

del version, PackageNotFoundError, metadata

__all__ = [
    "start",
    "instance",
    "view_project",
    "read_project",
    "read_molecules",
    "read_spline",
    "collect_projects",
]
