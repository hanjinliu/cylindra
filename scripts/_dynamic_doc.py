from __future__ import annotations

from pathlib import Path
import warnings
from griffe.dataclasses import Docstring
from griffe.docstrings import numpy

from cylindra import _shared_doc, start, instance

warnings.simplefilter("ignore", DeprecationWarning)

parse_orig = numpy._read_parameters

DOCS = Path(__file__).parent.parent
PATH_13_3 = DOCS.parent / "tests" / "13pf_MT.tif"


def on_startup(**kwargs):
    def _read_parameters(docstring: Docstring, **kwargs):
        if docstring.value:
            docstring.value = _shared_doc.update_doc(docstring.value, indent=0)
        return parse_orig(docstring, **kwargs)

    numpy._read_parameters = _read_parameters
    ui = start()
    ui.open_image(PATH_13_3)


def on_shutdown(**kwargs):
    numpy._read_parameters = parse_orig
    ui = instance()
    try:
        ui.parent_viewer.close()
    except (AttributeError, RuntimeError):
        pass
