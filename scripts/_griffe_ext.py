"""Griffe extension used by mkdocstrings to render cylindra docstrings."""

from __future__ import annotations

from typing import Any

from griffe import Extension, Module, Object

from cylindra import _shared_doc


def _replace_codes(docstring: str) -> str:
    """Convert doctest-style examples into fenced code blocks."""
    lines = docstring.split("\n")
    is_code = False
    for i in range(len(lines)):
        if is_code:
            if lines[i].startswith((">>> ", "... ")):
                lines[i] = lines[i][4:]
            else:
                is_code = False
                lines[i] = f"```\n{lines[i]}"
        else:
            if lines[i].startswith(">>> "):
                lines[i] = f"``` python\n{lines[i][4:]}"
                is_code = True
    return "\n".join(lines)


def _update_docstring(obj: Object) -> None:
    if obj.docstring is None or not (value := obj.docstring.value):
        return
    try:
        value = _shared_doc.update_doc(value, indent=0)
    except (KeyError, IndexError, ValueError):
        # Braces that are not `_shared_doc` placeholders, which only happens in
        # docstrings of other packages.
        pass
    if ">>> " in value:
        value = _replace_codes(value)
    obj.docstring.value = value


class CylindraDocstrings(Extension):
    """Fill in shared parameter descriptions and convert doctest examples."""

    def on_package(self, *, pkg: Module, **kwargs: Any) -> None:
        _update_docstring(pkg)

    def on_object(self, *, obj: Object, **kwargs: Any) -> None:
        _update_docstring(obj)
