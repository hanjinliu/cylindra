from __future__ import annotations

import argparse
import glob
from cylindra.cli._base import ParserBase
from pathlib import Path
from cylindra.core import read_project, view_project


class ShowScriptAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if _path := getattr(namespace, "path", None):
            import rich
            from rich.syntax import Syntax

            prj = read_project(_path)
            with prj.open_project() as dir:
                txt = prj.script_py_path(dir).read_text()
                syntax = Syntax(
                    txt,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=getattr(namespace, "wrap", False),
                )
                rich.print(syntax)
        else:
            raise ValueError("path is not given.")
        return parser.exit()


class ParserPreview(ParserBase):
    """
    cylindra preview [bold green]path[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]path[/u bold green]
        Path to the project/image file.

    [u bold cyan]options[/u bold cyan]
        [bold]--script, -s[/bold]
            Only preview script.py content.

        [bold]--wrap[/bold]
            Enable word wrap.
    """

    def __init__(self):
        super().__init__(
            prog="cylindra view", description="View a project, image or others."
        )
        self.add_argument("path", type=str, help="path to the file to view.")
        self.add_argument(
            "--script",
            "-s",
            nargs="*",
            action=ShowScriptAction,
        )
        self.add_argument("--wrap", action="store_true")

    def run_action(self, path: str, **kwargs):
        from magicgui.application import use_app

        _path = Path(path)
        if not _path.exists():
            raise FileNotFoundError(f"file not found: {_path}")
        match _path.suffix:
            case "" | ".tar" | ".zip" | ".json":
                print(f"Previewing project: {_path.as_posix()}")
                view_project(_path)
            case ".tif" | ".tiff" | ".mrc" | ".map":
                from cylindra._previews import view_image

                print(f"Previewing image: {_path.as_posix()}")
                if "*" in _path.as_posix():
                    view_image(glob(_path.as_posix()))
                else:
                    view_image(_path)
            case ".csv" | ".parquet":
                from cylindra._previews import view_tables

                print(f"Previewing table: {_path.as_posix()}")
                if "*" in _path.as_posix():
                    view_tables(glob(_path.as_posix()))
                else:
                    view_tables(_path)
            case _:
                raise ValueError(f"unknown file type: {_path.suffix}")
        if self.viewer is None:
            use_app().run()
