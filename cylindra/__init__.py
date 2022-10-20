import sys

if sys.platform == "darwin":
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)

from .core import start
from .components import CylTomogram
from .widgets import CylindraMainWidget
from .widgets.widget_utils import add_molecules, layer_to_coordinates
from ._info import __version__, __author__, __email__

__all__ = [
    "start",
    "CylTomogram",
    "CylindraMainWidget",
    "add_molecules",
    "layer_to_coordinates",
]
