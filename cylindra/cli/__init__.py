"""CLI parsers for cylindra."""

from cylindra.cli._base import ParserBase
from cylindra.cli.average import ParserAverage
from cylindra.cli.config import ParserConfig
from cylindra.cli.find import ParserFind
from cylindra.cli.new import ParserNew
from cylindra.cli.none import ParserNone
from cylindra.cli.open import ParserOpen
from cylindra.cli.preview import ParserPreview
from cylindra.cli.run import ParserRun
from cylindra.cli.workflow import ParserWorkflow

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
