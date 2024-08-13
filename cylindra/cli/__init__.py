"""CLI parsers for cylindra."""

from cylindra.cli._base import ParserBase
from cylindra.cli.average import ParserAverage
from cylindra.cli.config import ParserConfig
from cylindra.cli.find import ParserFind
from cylindra.cli.new import ParserNew
from cylindra.cli.none import ParserNone
from cylindra.cli.open import ParserOpen
from cylindra.cli.plugin import ParserPlugin
from cylindra.cli.preview import ParserPreview
from cylindra.cli.run import ParserRun
from cylindra.cli.workflow import ParserWorkflow

__all__ = ["set_current_viewer", "set_testing", "exec"]


def set_current_viewer(viewer):
    ParserBase.viewer = viewer


def set_testing(testing: bool):
    ParserBase._IS_TESTING = testing


def exec(argv: list[str]):
    match argv:
        case "open", *args:
            ParserOpen().parse(args)
        case "preview", *args:
            ParserPreview().parse(args)
        case "run", *args:
            ParserRun().parse(args)
        case "average", *args:
            ParserAverage().parse(args)
        case "new", *args:
            ParserNew().parse(args)
        case "config", *args:
            ParserConfig().parse(args)
        case "find", *args:
            ParserFind().parse(args)
        case "workflow", *args:
            ParserWorkflow().parse(args)
        case "plugin", *args:
            ParserPlugin().parse(args)
        case args:
            ParserNone().parse(args)
    return None
