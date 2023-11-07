from __future__ import annotations

import argparse
from pathlib import Path
from cylindra.cli._base import _ParserBase


def list_all_workflows():
    from cylindra._config import get_config

    print("List of workflows\n-----------------")
    for path in get_config().list_workflow_paths():
        print(f"{path.stem}")


class ListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        list_all_workflows()
        return parser.exit()


class ImportAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        from cylindra._config import workflow_path

        path = Path(namespace.path)
        workflow_path(path.name)
        return parser.exit()


class ParserWorkflow(_ParserBase):
    def __init__(self):
        super().__init__(
            prog="cylindra workflow", description="View, run and import workflows."
        )
        self.add_argument("path", nargs="?")
        self.add_argument(
            "--list",
            "-l",
            action=ListAction,
            nargs=0,
            help="List up all the available workflows.",
        )
        self.add_argument(
            "--import",
            action=ImportAction,
            nargs=0,
            help="Import a workflow from a file.",
        )
        self.add_argument("--wrap", "-w", action="store_true", help="Enable word wrap.")

    def run_action(self, path: str | None, wrap: bool = False, **kwargs):
        from cylindra._config import workflow_path
        import rich
        from rich.syntax import Syntax

        if path is None:
            return list_all_workflows()
        wpath = workflow_path(path)
        if not wpath.exists():
            raise FileNotFoundError(wpath)
        code = Syntax(
            wpath.read_text(),
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=wrap,
        )
        rich.print(code)
