from __future__ import annotations

from pathlib import Path
from cylindra.cli._base import ParserBase, coerce_output_filename
from cylindra.core import start, read_project


class ParserRun(ParserBase):
    """cylindra run <path>"""

    def __init__(self):
        super().__init__(prog="cylindra run", description="Run a script.")
        self.add_argument("path", type=str, help="Python script to run.")
        self.add_argument(
            "--headless", action="store_true", help="Run in headless mode."
        )
        self.add_argument(
            "--output", "-o", type=str, help="Output file name.", default=None
        )

    def run_action(self, path: str, headless: bool = False, output: str = None):
        from runpy import run_path

        ui = start(viewer=self.viewer, headless=headless)
        if Path(path).suffix in ("", ".tar", ".zip", ".json"):
            prj = read_project(path)
            with prj.open_project() as d:
                py_path = str(prj.script_py_path(d))
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
