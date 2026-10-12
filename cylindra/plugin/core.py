from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Concatenate, overload

from cylindra.plugin.function import CylindraPluginFunction

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

type _PluginFunc[**P, R] = Callable[Concatenate[CylindraMainWidget, P], R]


@overload
def register_function[**P, R](
    func: _PluginFunc[P, R],
    *,
    record: bool = True,
    name: str | None = None,
) -> CylindraPluginFunction[P, R]: ...


@overload
def register_function[**P, R](
    func: None = None,
    *,
    record: bool = True,
    name: str | None = None,
) -> Callable[[_PluginFunc[P, R]], CylindraPluginFunction[P, R]]: ...


def register_function(func=None, *, record=True, name=None):
    """Register a function as a plugin function.

    The registered function will be added to the plugin menu when the module is
    installed as a plugin.

    `magicclass` decorators, such as `setup_function_gui` and `impl_preview`,
    are compatible with this function.

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

    def _inner[**P, R](fn: _PluginFunc[P, R]) -> CylindraPluginFunction[P, R]:
        return CylindraPluginFunction(fn, name=name, record=record)

    return _inner if func is None else _inner(func)


def load_plugin(ui: CylindraMainWidget) -> None:
    """Load all the available plugins into the given instance."""
    from cylindra.plugin._find import iter_plugin_info

    for plugin_info in iter_plugin_info():
        plugin_info.load(ui)
