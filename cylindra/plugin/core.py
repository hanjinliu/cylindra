from __future__ import annotations

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
) -> CylindraPluginFunction[_P, _R]:
    ...


@overload
def register_function(
    func: Literal[None],
    *,
    record: bool = True,
    name: str | None = None,
) -> Callable[..., CylindraPluginFunction[_P, _R]]:
    ...


def register_function(
    func=None,
    *,
    record=True,
    name=None,
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
    """

    def _inner(func: Callable[_P, _R]) -> CylindraPluginFunction[_P, _R]:
        f = CylindraPluginFunction(func, name=name)
        if not record:
            f._is_recordable = record
        return f

    return _inner if func is None else _inner(func)


def load_plugin(ui: CylindraMainWidget) -> None:
    from cylindra.plugin._find import iter_plugin_info

    for plugin_info in iter_plugin_info():
        plugin_info.load(ui)
    return None
