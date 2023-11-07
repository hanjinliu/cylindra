"""CLI parsers for cylindra."""

from .average import ParserAverage
from .config import ParserConfig
from .find import ParserFind
from .new import ParserNew
from .none import ParserNone
from .open import ParserOpen
from .preview import ParserPreview
from .run import ParserRun
from .workflow import ParserWorkflow
from ._base import _ParserBase

__all__ = [
    "ParserAverage",
    "ParserConfig",
    "ParserFind",
    "ParserNew",
    "ParserNone",
    "ParserOpen",
    "ParserPreview",
    "ParserWorkflow",
    "ParserRun",
    "set_current_viewer",
]


def set_current_viewer(viewer):
    _ParserBase.viewer = viewer
