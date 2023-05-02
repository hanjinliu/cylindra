from .main import CylindraMainWidget
from . import _ui_init  # initialize main widget  # noqa: F401

del _ui_init

__all__ = ["CylindraMainWidget"]
