import sys

if sys.platform == "darwin":
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)
    
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
from cylindra.__about__ import __version__, __author__, __email__

from magicclass import defaults

defaults["macro-highlight"] = True

del defaults

__all__ = [
    "start",
    "instance",
    "view_project",
    "read_project",
    "read_molecules",
    "read_spline",
    "collect_projects",
]
