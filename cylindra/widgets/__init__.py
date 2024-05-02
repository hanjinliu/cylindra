import sys


def dont_use_native_menu_bar():  # pragma: no cover
    # cylindra has menu bars in sub widgets.
    from qtpy.QtCore import QCoreApplication, Qt

    if QCoreApplication.instance() is not None:
        return None
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
    return None


def init_opengl_and_dpi():
    from qtpy import QT6
    from qtpy.QtCore import QCoreApplication, Qt

    if QCoreApplication.instance() is not None:
        return None
    # Docking vispy widget in napari viewer requires this.
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    # High DPI support for High-DPI devices such as Surface Pro. Only for Qt<6.
    if not QT6:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    return None


if sys.platform == "darwin":
    dont_use_native_menu_bar()

init_opengl_and_dpi()

del dont_use_native_menu_bar, init_opengl_and_dpi

from cylindra.widgets.main import CylindraMainWidget  # noqa: E402, I001
from cylindra.widgets import _ui_init  # initialize main widget  # noqa: E402

del _ui_init

__all__ = ["CylindraMainWidget"]
