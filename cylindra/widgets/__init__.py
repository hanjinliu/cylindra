import sys
from .main import CylindraMainWidget
from . import _ui_init  # initialize main widget  # noqa: F401

del _ui_init


def dont_use_native_menu_bar():  # pragma: no cover
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

del dont_use_native_menu_bar, init_opengl_and_dpi

__all__ = ["CylindraMainWidget"]
