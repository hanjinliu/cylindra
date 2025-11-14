from __future__ import annotations

import importlib
import warnings
from importlib.metadata import distributions
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple

from magicgui.types import Separator

from cylindra.plugin.function import CylindraPluginFunction

if TYPE_CHECKING:
    from magicclass._gui import MenuGui

    from cylindra.widgets import CylindraMainWidget

ENTRY_POINT_GROUP_NAME = "cylindra.plugin"


class PluginInfo(NamedTuple):
    """Tuple for plugin specification."""

    name: str
    value: str
    version: str

    def load(self, ui: CylindraMainWidget) -> bool:
        """Load this plugin to the cylindra widget."""
        return load_plugin(ui, module_name=self.value, display_name=self.name)

    def reload(self, ui: CylindraMainWidget) -> None:
        """Reload this plugin to the cylindra widget."""
        reload_plugin(ui, module_name=self.value, display_name=self.name)


def iter_plugin_info() -> Iterator[PluginInfo]:
    dist_observed = set()
    for dist in distributions():
        if dist.name in dist_observed:
            continue
        for ep in dist.entry_points:
            if ep.group == ENTRY_POINT_GROUP_NAME:
                yield PluginInfo(ep.name, ep.value, dist.version)
        dist_observed.add(dist.name)


def load_plugin(
    ui: CylindraMainWidget,
    module_name: str,
    display_name: str,
    raises: bool = False,
) -> bool:
    from magicclass import magicmenu

    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        if raises:
            raise
        else:
            warnings.warn(
                f"Could not load plugin {module_name!r}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return False

    if display_name in ui.PluginsMenu:
        _newmenu = ui.PluginsMenu[display_name]
    else:

        @magicmenu(name=display_name, record=False)
        class newmenu:
            pass

        _newmenu = newmenu()
        ui.PluginsMenu.append(_newmenu)
        _newmenu.native.setParent(ui.PluginsMenu.native, _newmenu.native.windowFlags())
    _update_menu_gui(mod, ui, _newmenu)
    return True


def reload_plugin(
    ui: CylindraMainWidget,
    module_name: str,
    display_name: str,
) -> None:
    """Reload the plugin module and update the menu."""

    mod = importlib.import_module(module_name)
    mod = _reload(mod, module_name)

    _newmenu = ui.PluginsMenu[display_name]
    _newmenu.clear()
    _update_menu_gui(mod, ui, _newmenu)


def _reload(mod: ModuleType, root_mod: str) -> ModuleType:
    """Recursively reload a module and its submodules."""
    for obj in mod.__dict__.values():
        if isinstance(obj, ModuleType) and obj.__package__ == root_mod:
            _reload(obj, root_mod)
    return importlib.reload(mod)


def _dir_or_all(mod: ModuleType) -> Iterator[Any]:
    if hasattr(mod, "__cylindra_methods__"):
        _list = mod.__cylindra_methods__
    elif hasattr(mod, "__all__"):
        _list = mod.__all__
    else:
        _list = dir(mod)
    for attr in _list:
        if isinstance(attr, str):
            obj = getattr(mod, attr)
        else:
            obj = attr
        yield obj


def _update_menu_gui(mod: ModuleType, ui: CylindraMainWidget, menu: MenuGui):
    for obj in _dir_or_all(mod):
        if isinstance(obj, CylindraPluginFunction):
            menu.append(obj.update_module(mod).as_method(ui))
        elif obj is Separator:
            menu.native.addSeparator()
