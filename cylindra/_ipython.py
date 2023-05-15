from __future__ import annotations

from typing import TYPE_CHECKING, Any
from IPython import get_ipython, InteractiveShell
from IPython.core.magic import register_line_magic, needs_local_scope

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget


def _get_ui(local_ns: dict[str, Any]) -> CylindraMainWidget:
    ui = local_ns.get(".ui")
    if ui is None:
        raise NameError("Cylindra UI not found.")
    return ui


def install_ipython_magic():
    if shell := get_ipython():
        shell: InteractiveShell
        shell.enable_matplotlib(gui="inline")  # use inline plot by default

        @register_line_magic
        @needs_local_scope
        def macro(line: str, local_ns: dict[str, Any]) -> str:
            ui = _get_ui(local_ns)
            if line == "full":
                print(ui.macro)
            elif line == "":
                print(ui.macro[ui._macro_offset :])
            else:
                raise ValueError(f"Invalid argument: {line}")

        del macro
