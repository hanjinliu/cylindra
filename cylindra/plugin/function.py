from __future__ import annotations

import ast
import inspect
import warnings
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generic, ParamSpec, TypeVar

from macrokit import Expr, Symbol

if TYPE_CHECKING:
    from magicclass._gui.mgui_ext import Action

    from cylindra.widgets import CylindraMainWidget

_P = ParamSpec("_P")
_R = TypeVar("_R")


class CylindraPluginFunction(Generic[_P, _R]):
    def __init__(
        self,
        func: Callable[_P, _R],
        name: str | None = None,
        module: str | None = None,
        record: bool = True,
    ):
        from magicclass.utils import thread_worker

        if not callable(func):
            raise TypeError("func must be a callable")
        if not hasattr(func, "__name__"):
            raise ValueError("func must have a __name__ attribute.")
        if name is None:
            name = func.__name__.replace("_", " ").capitalize()
        self._name = name
        if module is None:
            module = func.__module__
        self._is_recordable = record
        if module == "__main__" and record:
            warnings.warn(
                f"Plugin function {func!r} is in the top-level module '__main__', "
                "which means it is only defined during this session. Calls of this "
                "function will be recorded in the macro but the script will not work. "
                "Add 'record=False' to the `register_function` decorator, or define "
                "plugin function in a separate module.",
                UserWarning,
                stacklevel=2,
            )
        self._module = module
        wraps(func)(self)
        self._func = func
        self._action_ref: Callable[[], Action | None] = lambda: None
        self.__signature__ = inspect.signature(func)
        first_arg = next(iter(self.__signature__.parameters.values()))
        self._ui_arg_name = first_arg.name
        # check if the first argument is a CylindraMainWidget
        if first_arg.annotation is not inspect.Parameter.empty:
            from cylindra.widgets import CylindraMainWidget

            if first_arg.annotation not in [CylindraMainWidget, "CylindraMainWidget"]:
                warnings.warn(
                    f"The first argument of a plugin function {func!r} should be a "
                    f"CylindraMainWidget but was {first_arg.annotation!r}.",
                    UserWarning,
                    stacklevel=2,
                )

        if isinstance(self._func, thread_worker):
            if record:
                self._func._set_recorder(self._record_macro)
            else:
                self._func._set_silencer()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._name}>"

    def import_statement(self) -> str:
        """Make an import statement for the plugin"""
        expr = f"import {self._module}"
        try:
            ast.parse(expr)
        except SyntaxError:
            raise ValueError(f"Invalid import statement: {expr}") from None
        return expr

    def update_module(self, mod: ModuleType):
        """Update the module name of the plugin function"""
        self._module = mod.__name__
        return self

    def as_method(self, ui):
        """As a method bound to the given CylindraMainWidget instance."""
        from magicclass.signature import upgrade_signature

        def _method(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return self(ui, *args, **kwargs)

        params = list(self.__signature__.parameters.values())
        aopt = getattr(self.__signature__, "additional_options", None)
        _method.__signature__ = inspect.Signature(params[1:])
        _method.__name__ = self._name
        _method.__doc__ = getattr(self._func, "__doc__", "")
        if qualname := getattr(self._func, "__qualname__", None):
            _method.__qualname__ = qualname
        upgrade_signature(_method, additional_options=aopt)
        return _method

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        from magicclass.utils import thread_worker

        from cylindra.widgets import CylindraMainWidget

        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        ui = bound.arguments[self._ui_arg_name]
        if not isinstance(ui, CylindraMainWidget):
            raise TypeError(
                f"Expected a CylindraMainWidget instance as the first argument "
                f"{self._ui_arg_name!r} but got {ui!r}"
            )
        first_arg, *args = bound.args
        assert first_arg is ui
        if isinstance(self._func, thread_worker):
            if action := self._action_ref():
                self._func._force_async = action.running
            else:
                self._func._force_async = False
            out = self._func.__get__(ui)(*args, **bound.kwargs)
        else:
            with ui.macro.blocked():
                out = self._func(*bound.args, **bound.kwargs)

            # macro recording
            if self._is_recordable:
                out = self._record_macro(ui, out, *args, **bound.kwargs)

        return out

    def _record_macro(
        self,
        ui: CylindraMainWidget,
        out,
        *args,
        **kwargs,
    ):
        from magicclass.undo import UndoCallback

        fn_expr = Expr("getattr", [Symbol(self._module), self._func.__name__])
        expr = Expr.parse_call(fn_expr, (ui,) + args, kwargs)
        ui.macro.append(expr)
        ui.macro._last_setval = None
        if self not in ui._plugins_called:
            ui._plugins_called.append(self)
        if isinstance(out, UndoCallback):
            ui.macro._append_undo(out.with_name(str(expr)))
            out = out.return_value
        else:
            ui.macro.clear_undo_stack()
        return out
