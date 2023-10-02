from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    TypeVar,
    MutableSequence,
)
from typing_extensions import Self
import weakref
from fnmatch import fnmatch

from macrokit import Expr, Head, symbol

import napari
from cylindra.types import MoleculesLayer

if TYPE_CHECKING:
    from cylindra.widgets.main import CylindraMainWidget
    from acryo import Molecules

_T = TypeVar("_T", bound="Accessor")
_V = TypeVar("_V")


class AccessorField(Generic[_T]):
    def __init__(self, constructor: type[_T]):
        self._instances = dict[int, _T]()
        self._constructor = constructor

    def __get__(self, instance: Any, owner: type) -> _T:
        if instance is None:
            return self
        _id = id(instance)
        if _id not in self._instances:
            self._instances[_id] = self._constructor(instance)
        return self._instances[_id]


class Accessor:
    def __init__(self, widget: CylindraMainWidget):
        self._widget = weakref.ref(widget)

    @classmethod
    def field(cls) -> AccessorField[Self]:
        return AccessorField(cls)

    def widget(self) -> CylindraMainWidget:
        widget = self._widget()
        if widget is None:
            raise RuntimeError("Widget is already deleted.")
        return widget

    def viewer(self) -> napari.Viewer:
        viewer = self.widget().parent_viewer
        if viewer is None:
            raise RuntimeError("Viewer not found.")
        return viewer


_Condition = Callable[[MoleculesLayer], bool]


class MoleculesLayerAccessor(Accessor, MutableSequence[MoleculesLayer]):
    """Accessor to the molecules layers of the viewer."""

    def __getitem__(self, name: str) -> MoleculesLayer:
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
        return list(layer.name for layer in self)

    def count(self) -> int:
        """Number of molecules layers."""
        return len(self)

    def _ipython_key_completions_(self) -> list[str]:
        """Just for autocompletion."""
        return list(self.viewer().layers)

    def __iter__(self) -> Iterator[MoleculesLayer]:
        for layer in self.viewer().layers:
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
            condition = lambda _: True
        for layer in self:
            if condition(layer):
                yield layer

    def iter_molecules(
        self, condition: _Condition | None = None
    ) -> Iterator[Molecules]:
        """Iterate over molecules."""
        if condition is None:
            condition = lambda _: True
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
        old_names = list(
            l.name for l in self._filtered_layers(include, exclude, pattern)
        )
        new_names = list(n.replace(old, new) for n in old_names)
        ui = self.widget()
        viewer = self.viewer()

        def undo():
            for old, new in zip(old_names, new_names):
                viewer.layers[new].name = old

        def redo():
            for old, new in zip(old_names, new_names):
                viewer.layers[old].name = new

        with ui.macro.blocked():
            redo()
        fn = Expr(Head.getattr, [self._my_symbol(), "rename"])
        kwargs = dict(
            old=old, new=new, include=include, exclude=exclude, pattern=pattern
        )
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
