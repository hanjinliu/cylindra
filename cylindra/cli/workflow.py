from __future__ import annotations

import argparse
from pathlib import Path

from cylindra.cli._base import ParserBase


def list_all_workflows():
    import rich
    from rich.panel import Panel

    from cylindra._config import get_config

    _list = list[str]()
    for path in get_config().list_workflow_paths():
        _list.append(f" - {path.stem}")

    rich.print(Panel("\n".join(_list), title="List of workflows", title_align="center"))


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


class ParserWorkflow(ParserBase):
    """
    cylindra workflow [bold green]path[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]path[/u bold green]
        Path to the workflow file.
        e.g. Show the workflow file named my_workflow.py
            `cylindra workflow my_workflow`

    [u bold cyan]options[/u bold cyan]
        [bold]--list, -l[/bold]
            List up all the available workflows.

        [bold]--import[/bold]
            Import a workflow from a python file.
            e.g. `cylindra workflow 3rd-party-file.py --import`

        [bold]--wrap, -w[/bold]
            Enable word wrap.
    """

    def __init__(self):
        super().__init__(
            prog="cylindra workflow", description="View, run and import workflows."
        )
        self.add_argument("path", nargs="?")
        self.add_argument("--list", "-l", action=ListAction, nargs=0)
        self.add_argument("--import", action=ImportAction, nargs=0)
        self.add_argument("--wrap", "-w", action="store_true")

    def run_action(self, path: str | None, wrap: bool = False, **kwargs):
        import rich
        from rich.syntax import Syntax

        from cylindra._config import workflow_path

        if path is None:
            return list_all_workflows()
        wpath = workflow_path(path)
        if not wpath.exists():
            if path == "list":  # just support `cylindra workflow list`
                return list_all_workflows()
            raise FileNotFoundError(wpath)
        code = Syntax(
            wpath.read_text(),
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=wrap,
        )
        rich.print(code)
