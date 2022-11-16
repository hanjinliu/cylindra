import sys

if sys.platform == "darwin":
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)

from cylindra.core import start, instance
from cylindra.components import CylTomogram, CylinderModel
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets.widget_utils import add_molecules, layer_to_coordinates
from cylindra._info import __version__, __author__, __email__

__all__ = [
    "start",
    "instance",
    "CylTomogram",
    "CylinderModel",
    "CylindraMainWidget",
    "add_molecules",
    "layer_to_coordinates",
]
