from __future__ import annotations

import importlib
import warnings
from importlib.metadata import distributions
from types import ModuleType
from typing import TYPE_CHECKING, Iterator, NamedTuple

from cylindra.plugin.function import CylindraPluginFunction

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget

ENTRY_POINT_GROUP_NAME = "cylindra.plugin"


class PluginInfo(NamedTuple):
    name: str
    value: str
    version: str

    def load(self, ui: CylindraMainWidget) -> bool:
        return load_plugin(ui, module_name=self.value, display_name=self.name)

    def reload(self, ui: CylindraMainWidget) -> None:
        reload_plugin(ui, module_name=self.value, display_name=self.name)


def iter_plugin_info() -> Iterator[PluginInfo]:
    for dist in distributions():
        for ep in dist.entry_points:
            if ep.group == ENTRY_POINT_GROUP_NAME:
                yield PluginInfo(ep.name, ep.value, dist.version)


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
    for attr in _dir_or_all(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, CylindraPluginFunction):
            _newmenu.append(obj.update_module(mod).as_method(ui))
    return True


def reload_plugin(
    ui: CylindraMainWidget,
    module_name: str,
    display_name: str,
) -> None:
    import importlib

    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)

    _newmenu = ui.PluginsMenu[display_name]
    _newmenu.clear()
    for attr in _dir_or_all(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, CylindraPluginFunction):
            _newmenu.append(obj.update_module(mod).as_method(ui))


def _dir_or_all(mod: ModuleType) -> list[str]:
    if hasattr(mod, "__all__"):
        return mod.__all__
    else:
        return dir(mod)
