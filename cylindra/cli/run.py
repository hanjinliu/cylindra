from __future__ import annotations

from pathlib import Path

from cylindra.cli._base import ParserBase, coerce_output_filename
from cylindra.core import read_project, start


class ParserRun(ParserBase):
    """
    cylindra run [bold green]path[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]path[/u bold green]
        Path to the project/image file.

    [u bold cyan]options[/u bold cyan]
        [bold]--headless[/bold]
            Run in headless mode.

        [bold]--output, -o[/bold]
            Output file name.
    """

    def __init__(self):
        super().__init__(prog="cylindra run", description="Run a script.")
        self.add_argument("path", type=str, help="Python script to run.")
        self.add_argument("--headless", action="store_true")
        self.add_argument("--output", "-o", type=str, default=None)

    def run_action(
        self, path: str, headless: bool = False, output: str = None, **kwargs
    ):
        from runpy import run_path

        ui = start(viewer=self.viewer, headless=headless)
        if Path(path).suffix in ("", ".tar", ".zip", ".json"):
            prj = read_project(path)
            with prj.open_project() as d:
                py_path = str(prj._script_py_path(d))
                out_globs = run_path(py_path, {"ui": ui})
            is_project = True
        else:
            out_globs = run_path(path, {"ui": ui})
            is_project = False
        if callable(main := out_globs.get("main")):
            main(ui)  # script.py style
        if output is None:
            if is_project:
                ui.overwrite_project()
            else:
                output = coerce_output_filename("output", ext="")
                ui.save_project(output, molecules_ext=".parquet")
        else:
            ui.save_project(output, molecules_ext=".parquet")
