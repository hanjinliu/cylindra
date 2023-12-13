from __future__ import annotations

import warnings
from griffe.dataclasses import Docstring
from griffe.docstrings import numpy

from cylindra import _shared_doc

warnings.simplefilter("ignore", DeprecationWarning)

parse_orig = numpy._read_parameters


def on_startup(**kwargs):
    def _read_parameters(docstring: Docstring, **kwargs):
        if docstring.value:
            docstring.value = _shared_doc.update_doc(docstring.value, indent=0)
        return parse_orig(docstring, **kwargs)

    numpy._read_parameters = _read_parameters


def on_shutdown(**kwargs):
    numpy._read_parameters = parse_orig
