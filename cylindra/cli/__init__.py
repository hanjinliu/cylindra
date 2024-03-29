"""CLI parsers for cylindra."""

from ._base import ParserBase
from .average import ParserAverage
from .config import ParserConfig
from .find import ParserFind
from .new import ParserNew
from .none import ParserNone
from .open import ParserOpen
from .preview import ParserPreview
from .run import ParserRun
from .workflow import ParserWorkflow

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
    ParserBase.viewer = viewer


def set_testing(testing: bool):
    ParserBase._IS_TESTING = testing
