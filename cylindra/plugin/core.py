from __future__ import annotations

import warnings
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Literal, ParamSpec, TypeVar, overload

from cylindra.plugin.function import CylindraPluginFunction

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

_P = ParamSpec("_P")
_R = TypeVar("_R")


@overload
def register_function(
    func: Callable[_P, _R],
    *,
    record: bool = True,
    name: str | None = None,
    import_from: str | ModuleType | None = None,
) -> CylindraPluginFunction[_P, _R]:
    ...


@overload
def register_function(
    func: Literal[None],
    *,
    record: bool = True,
    name: str | None = None,
    import_from: str | ModuleType | None = None,
) -> Callable[..., CylindraPluginFunction[_P, _R]]:
    ...


def register_function(
    func=None,
    *,
    record=True,
    name=None,
    import_from=None,
):
    """
    Register a function as a plugin function.

    The registered function will be added to the plugin menu when the module is
    installed as a plugin.

    Parameters
    ----------
    func : callable, optional
        The plugin function. If the function is to be called in the GUI, its signature
        must be interpretable for `magicgui`. The first argument of the function must be
        the `CylindraMainWidget` instance.
    record : bool, default True
        If False, the function will not be recorded in the macro.
    name : str, optional
        Name to display in the menu. If None, the capitalized function name will be
        used.
    import_from : str or ModuleType, optional
        Module to import the function from when macro containing the function calls is
        created. If None, the function will be imported from the place defined in the
        source code.
    """
    import_from = _norm_import_from(import_from)

    def _inner(func: Callable[_P, _R]) -> CylindraPluginFunction[_P, _R]:
        f = CylindraPluginFunction(func, name=name, module=import_from)
        if not record:
            f._is_recordable = record
        return f

    return _inner if func is None else _inner(func)


def load_plugin(
    ui: CylindraMainWidget,
    plugin_name: str,
    raises: bool = False,
) -> bool:
    import importlib

    from magicclass import magicmenu

    try:
        mod = importlib.import_module(plugin_name)
    except ImportError as e:
        if raises:
            raise
        else:
            warnings.warn(
                f"Could not load plugin {plugin_name!r}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return False

    @magicmenu(name=plugin_name, record=False)
    class newmenu:
        pass

    _newmenu = newmenu()
    ui.OthersMenu.Plugins.append(_newmenu)
    _newmenu.native.setParent(
        ui.OthersMenu.Plugins.native, _newmenu.native.windowFlags()
    )
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, CylindraPluginFunction):
            _newmenu.append(obj.as_method(ui))
    return True


def _norm_import_from(mod: str | ModuleType | None) -> str | None:
    if mod is None:
        return None
    if isinstance(mod, str):
        return mod
    elif isinstance(mod, ModuleType):
        return mod.__name__
    raise TypeError(
        f"import_from must be a string or a module, not {type(mod).__name__}"
    )
