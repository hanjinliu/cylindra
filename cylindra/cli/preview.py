from __future__ import annotations

import argparse
import glob
from cylindra.cli._base import _ParserBase
from pathlib import Path
from cylindra.core import read_project, view_project


class ParserPreview(_ParserBase):
    """cylindra preview <path>"""

    class ShowScriptAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            ln = True
            if _path := getattr(namespace, "path", None):
                prj = read_project(_path)
                with prj.open_project() as dir:
                    txt = prj.script_py_path(dir).read_text()
                    self.print_rich(txt, line_numbers=ln)
            else:
                raise ValueError("path is not given.")
            return parser.exit()

        def print_rich(self, txt: str, line_numbers: bool = True):
            import rich
            from rich.syntax import Syntax

            syntax = Syntax(
                txt,
                "python",
                theme="monokai",
                line_numbers=line_numbers,
            )
            rich.print(syntax)

    def __init__(self):
        super().__init__(
            prog="cylindra view", description="View a project, image or others."
        )
        self.add_argument("path", type=str, help="path to the file to view.")
        self.add_argument(
            "--script",
            "-s",
            nargs="*",
            action=self.ShowScriptAction,
            help="Only preview script.py content.",
        )

    def run_action(self, path: str, **kwargs):
        from cylindra._previews import view_tables, view_image
        from magicgui.application import use_app

        _path = Path(path)
        if not _path.exists():
            raise FileNotFoundError(f"file not found: {_path}")
        match _path.suffix:
            case "" | ".tar" | ".zip" | ".json":
                print(f"Previewing project: {_path.as_posix()}")
                view_project(_path)
            case ".tif" | ".tiff" | ".mrc" | ".map":
                print(f"Previewing image: {_path.as_posix()}")
                if "*" in _path.as_posix():
                    view_image(glob(_path.as_posix()))
                else:
                    view_image(_path)
            case ".csv" | ".parquet":
                print(f"Previewing table: {_path.as_posix()}")
                if "*" in _path.as_posix():
                    view_tables(glob(_path.as_posix()))
                else:
                    view_tables(_path)
            case _:
                raise ValueError(f"unknown file type: {_path.suffix}")
        if self.viewer is None:
            use_app().run()
