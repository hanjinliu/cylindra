from __future__ import annotations

import warnings
from pathlib import Path

from griffe import Docstring, Parser, parsers

from cylindra import _shared_doc, instance, start

warnings.simplefilter("ignore", DeprecationWarning)

parse_orig = parsers[Parser.numpy]

DOCS = Path(__file__).parent.parent
PATH_13_3 = DOCS.parent / "tests" / "13pf_MT.tif"


def _dynamic_parse(docstring: Docstring, **kwargs):
    if docstring.value:
        docstring.value = _shared_doc.update_doc(docstring.value, indent=0)
        if ">>> " in docstring.value:
            docstring.value = _replace_codes(docstring.value)
    return parse_orig(docstring, **kwargs)


def _replace_codes(docstring: str):
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


def on_startup(**kwargs):
    parsers[Parser.numpy] = _dynamic_parse
    ui = start()
    ui.open_image(PATH_13_3)


def on_shutdown(**kwargs):
    parsers[Parser.numpy] = parse_orig
    ui = instance()
    try:
        ui.parent_viewer.close()
    except (AttributeError, RuntimeError):
        pass
