import sys

if sys.platform == "darwin":
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)

from cylindra.core import (
    start,
    instance,
    view_project,
    read_project,
    read_molecules,
    read_spline,
    read_localprops,
    read_globalprops,
)
from cylindra.components import CylTomogram, CylinderModel
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import add_molecules, layer_to_coordinates
from cylindra.__about__ import __version__, __author__, __email__

from magicclass import defaults

defaults["macro-highlight"] = True


__all__ = [
    "start",
    "instance",
    "view_project",
    "read_project",
    "read_molecules",
    "read_spline",
    "read_localprops",
    "read_globalprops",
    "CylTomogram",
    "CylinderModel",
    "CylindraMainWidget",
    "add_molecules",
    "layer_to_coordinates",
]
