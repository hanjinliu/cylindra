from __future__ import annotations

import ast
import inspect
import warnings
from functools import wraps
from types import ModuleType
from typing import Callable, Generic, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


class CylindraPluginFunction(Generic[_P, _R]):
    def __init__(
        self,
        func: Callable[_P, _R],
        name: str | None = None,
        module: str | None = None,
    ):
        if not callable(func):
            raise TypeError("func must be a callable")
        if not hasattr(func, "__name__"):
            raise ValueError("func must have a __name__ attribute.")
        self._func = func
        if name is None:
            name = func.__name__.replace("_", " ").capitalize()
        self._name = name
        if module is None:
            module = func.__module__
        self._is_recordable = _is_recordable(func)
        if module == "__main__" and self._is_recordable:
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
        self.__signature__ = inspect.signature(func)
        first_arg = next(iter(self.__signature__.parameters.values()))
        self._ui_arg_name = first_arg.name
        # check if the first argument is a CylindraMainWidget
        if first_arg.annotation is not inspect.Parameter.empty:
            from cylindra.widgets import CylindraMainWidget

            if first_arg.annotation is not CylindraMainWidget:
                warnings.warn(
                    f"The first argument of a plugin function {func!r} should be a "
                    f"CylindraMainWidget but was {first_arg.annotation!r}.",
                    UserWarning,
                    stacklevel=2,
                )

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
        from magicclass.signature import upgrade_signature

        def _method(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return self(ui, *args, **kwargs)

        params = list(self.__signature__.parameters.values())
        _method.__signature__ = inspect.Signature(params[1:])
        _method.__name__ = self._name
        _method.__doc__ = getattr(self._func, "__doc__", "")
        upgrade_signature(_method)
        return _method

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        from macrokit import Expr, Symbol
        from magicclass.undo import UndoCallback

        from cylindra.widgets import CylindraMainWidget

        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        ui = bound.arguments[self._ui_arg_name]
        if not isinstance(ui, CylindraMainWidget):
            raise TypeError(
                f"Expected a CylindraMainWidget instance as the first argument "
                f"{self._ui_arg_name!r} but got {ui!r}"
            )
        # TODO: how to use thread_worker?
        out = self._func(*bound.args, **bound.kwargs)

        # macro recording
        _args = []
        _kwargs = {}
        for name, param in bound.signature.parameters.items():
            if name == self._ui_arg_name:
                _args.append(ui._my_symbol)
            elif param.kind is inspect.Parameter.POSITIONAL_ONLY:
                _args.append(bound.arguments[name])
            else:
                _kwargs[name] = bound.arguments[name]
        fn_expr = Expr("getattr", [Symbol(self._module), self._func.__name__])
        expr = Expr.parse_call(fn_expr, tuple(_args), _kwargs)
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


def _is_recordable(func: Callable) -> bool:
    if hasattr(func, "__is_recordable__"):
        return func.__is_recordable__
    if hasattr(func, "__func__"):
        return _is_recordable(func.__func__)
    return False
