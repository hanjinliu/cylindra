__version__ = "0.9.1.dev0"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

import sys

if sys.platform == "darwin":
    from qtpy.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_DontUseNativeMenuBar)

from .core import start
from .components import MtTomogram
from .widgets import MTPropsWidget
from .widgets.widget_utils import add_molecules

__all__ = ["start", "MtTomogram", "MTPropsWidget", "add_molecules"]