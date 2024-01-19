from __future__ import annotations

import weakref
from fnmatch import fnmatch
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    MutableSequence,
    TypeVar,
)

import napari
from macrokit import Expr, Head, symbol

from cylindra.types import MoleculesLayer

if TYPE_CHECKING:
    from acryo import Molecules
    from magicclass import MagicTemplate
    from typing_extensions import Self

    from cylindra.widgets.batch import CylindraBatchWidget  # noqa: F401
    from cylindra.widgets.batch._utils import LoaderInfo
    from cylindra.widgets.main import CylindraMainWidget  # noqa: F401

_V = TypeVar("_V")
_W = TypeVar("_W", bound="MagicTemplate")


class Accessor(MutableSequence[_V], Generic[_V, _W]):
    def __init__(self, widget: _W | None = None):
        if widget is not None:
            self._widget = weakref.ref(widget)
        else:
            self._widget = lambda: None
        self._instances = dict[int, "Self"]()

    def widget(self) -> _W:
        widget = self._widget()
        if widget is None:
            raise RuntimeError("Widget is already deleted.")
        return widget

    def viewer(self) -> napari.Viewer:
        viewer = self.widget().parent_viewer
        if viewer is None:
            raise RuntimeError("Viewer not found.")
        return viewer

    def __get__(self, instance: Any, owner: type) -> Self:
        if instance is None:
            return self
        _id = id(instance)
        if _id not in self._instances:
            self._instances[_id] = self.__class__(instance)
        return self._instances[_id]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"


_Condition = Callable[[MoleculesLayer], bool]


class MoleculesLayerAccessor(Accessor[MoleculesLayer, "CylindraMainWidget"]):
    """Accessor to the molecules layers of the viewer."""

    def __getitem__(self, name: int | str) -> MoleculesLayer:
        if hasattr(name, "__index__"):
            return self.nth(name)
        return _get_monomer_layer(self.viewer(), name)

    def __setitem__(self, name: str, layer: MoleculesLayer) -> None:
        if name in self:
            raise ValueError(f"Layer {name} already exists.")
        return self.append(layer)

    def __delitem__(self, name: str) -> None:
        viewer = self.widget().parent_viewer
        del viewer.layers[name]

    def clear(self) -> None:
        """Clear all molecules layers."""
        return self.delete()

    def first(self) -> MoleculesLayer:
        """The first molecules layer."""
        return next(self._filtered_layers())

    def last(self) -> MoleculesLayer:
        """The last molecules layer."""
        return self.list()[-1]

    def nth(self, n: int) -> MoleculesLayer:
        """The n-th molecules layer."""
        if not hasattr(n, "__index__"):
            raise TypeError(f"n must be an integer, not {type(n)}.")
        return self.list()[n]

    def names(self) -> list[str]:
        """All molecules layer names."""
        return [layer.name for layer in self]

    def count(self) -> int:
        """Number of molecules layers."""
        return len(self)

    def _ipython_key_completions_(self) -> list[str]:
        """Just for autocompletion."""  # BUG: not working
        return self.names()

    def __iter__(self) -> Iterator[MoleculesLayer]:
        # adding layers during iteration causes RecursionError
        _layers = list(self.viewer().layers)
        for layer in _layers:
            if isinstance(layer, MoleculesLayer):
                yield layer

    _void = object()

    def get(self, name: str, default: _V = _void) -> MoleculesLayer | _V:
        """Get a layer by name, return default if not exist."""
        try:
            return self[name]
        except KeyError:
            if default is self._void:
                raise
            return default

    def iter(self, condition: _Condition | None = None) -> Iterator[MoleculesLayer]:
        """Iterate over molecules layers."""
        if condition is None:

            def condition(_):
                return True

        for layer in self:
            if condition(layer):
                yield layer

    def iter_molecules(
        self, condition: _Condition | None = None
    ) -> Iterator[Molecules]:
        """Iterate over molecules."""
        if condition is None:

            def condition(_):
                return True

        for layer in self:
            mole = layer.molecules
            if condition(mole):
                yield mole

    def delete(
        self,
        *,
        include: str = "",
        exclude: str = "",
        pattern: str = "",
    ) -> None:
        to_delete = self.list(include=include, exclude=exclude, pattern=pattern)
        ui = self.widget()
        viewer = self.viewer()

        def undo():
            for layer in to_delete:
                viewer.add_layer(layer)

        def redo():
            for layer in to_delete:
                del viewer.layers[layer.name]

        with ui.macro.blocked():
            redo()
        fn = Expr(Head.getattr, [self._my_symbol(), "delete"])
        kwargs = {"include": include, "exclude": exclude, "pattern": pattern}
        expr = Expr.parse_call(fn, kwargs=kwargs)
        ui.macro.append_with_undo(expr, undo, redo)
        return None

    def list(
        self,
        *,
        include: str = "",
        exclude: str = "",
        pattern: str = "",
    ) -> list[MoleculesLayer]:
        return list(self._filtered_layers(include, exclude, pattern))

    def rename(
        self,
        old: str,
        new: str,
        *,
        include: str = "",
        exclude: str = "",
        pattern: str = "",
    ) -> None:
        old_names = [l.name for l in self._filtered_layers(include, exclude, pattern)]
        new_names = [n.replace(old, new) for n in old_names]
        ui = self.widget()
        viewer = self.viewer()

        def undo():
            for old, new in zip(old_names, new_names, strict=True):
                viewer.layers[new].name = old

        def redo():
            for old, new in zip(old_names, new_names, strict=True):
                viewer.layers[old].name = new

        with ui.macro.blocked():
            redo()
        fn = Expr(Head.getattr, [self._my_symbol(), "rename"])
        kwargs = {
            "old": old,
            "new": new,
            "include": include,
            "exclude": exclude,
            "pattern": pattern,
        }
        expr = Expr.parse_call(fn, kwargs=kwargs)
        ui.macro.append_with_undo(expr, undo, redo)
        return None

    def _filtered_layers(
        self,
        include: str = "",
        exclude: str = "",
        pattern: str = "",
    ) -> Iterator[MoleculesLayer]:
        for layer in self:
            if include and include not in layer.name:
                continue
            if exclude and exclude in layer.name:
                continue
            if pattern and not fnmatch(layer.name, pattern):
                continue
            yield layer

    def _my_symbol(self) -> Expr:
        return Expr(Head.getattr, [symbol(self.widget()), "mole_layers"])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, other: Any) -> bool:
        return other in self.viewer().layers

    def insert(self, index: int, layer: MoleculesLayer) -> None:
        if not isinstance(layer, MoleculesLayer):
            raise TypeError("Only MoleculesLayer can be inserted.")
        return self.viewer().layers.insert(index, layer)


def _get_monomer_layer(viewer: napari.Viewer, name: str) -> MoleculesLayer:
    if not isinstance(name, str):
        raise TypeError(f"Layer name must be a string, not {type(name)}.")
    layer = viewer.layers[name]
    if not isinstance(layer, MoleculesLayer):
        raise TypeError(f"Layer {name} is not a MoleculesLayer.")
    return layer


class BatchLoaderAccessor(Accessor["LoaderInfo", "CylindraBatchWidget"]):
    def __getitem__(self, name: int | str) -> LoaderInfo:
        return self.widget()._loaders[name]

    def __setitem__(self, name: str, layer: LoaderInfo) -> None:
        if name in self:
            raise ValueError(f"Layer {name} already exists.")
        self.widget()._loaders[name] = layer

    def __delitem__(self, name: int | str) -> None:
        if hasattr(name, "__index__"):
            del self.widget()._loaders[name]
            return
        for i, info in enumerate(self.widget()._loaders):
            if info.name == name:
                self.widget()._loaders.pop(i)
                return
        raise KeyError(f"Loader {name} not found.")

    def insert(self, index: int, info: LoaderInfo) -> None:
        return self.widget()._loaders.insert(index, info)

    def __iter__(self) -> Iterator[LoaderInfo]:
        return iter(self.widget()._loaders)

    def __len__(self) -> int:
        return len(self.widget()._loaders)

    def names(self) -> list[str]:
        """All molecules layer names."""
        return [layer.name for layer in self]

    def count(self) -> int:
        """Number of molecules layers."""
        return len(self)

    def _ipython_key_completions_(self) -> list[str]:
        """Just for autocompletion."""
        return self.names()
