from __future__ import annotations

import argparse
from cylindra.core import start
from cylindra.cli._base import _ParserBase


class HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        doc = parser.__doc__
        assert doc is not None
        lines = doc.splitlines()
        while lines[0].strip() == "":
            lines.pop(0)
        nindents = len(lines[0]) - len(lines[0].lstrip())
        print("\n".join(line[nindents:] for line in lines))
        parser.exit()


class ParserNone(_ParserBase):
    """
    Usage: cylindra [OPTIONS] COMMAND [ARGUMENTS]

    Options:
        -v, --version   Show version.
        -h, --help      Show this message and exit.

    Commands:
        open       Open a project or an image.
        preview    View a project, image or others.
        find       Find cylindra projects by patterns and contents.
        run        Run a script.
        config     Edit/view the configuration.
        average    Average images.
        new        Create a new project.
        add        Add splines by coordinates.
    """

    def __init__(self):
        from cylindra import __version__

        super().__init__(
            prog="cylindra",
            description="Command line interface of cylindra.",
            add_help=False,
        )
        self.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"cylindra version {__version__}",
            help="Show version.",
        )
        self.add_argument(
            "-h",
            "--help",
            nargs=0,
            action=HelpAction,
        )

    def run_action(self, **kwargs):
        ui = start(viewer=self.viewer)
        ui.parent_viewer.show(block=self.viewer is None)
        return None
