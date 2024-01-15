from __future__ import annotations

from pathlib import Path

from macrokit import Expr, Head, parse

_MACRO_FORMAT = """import polars as pl
from cylindra.widgets import CylindraMainWidget
from cylindra import instance

def main(ui: CylindraMainWidget):
{}

if __name__ == "__main__":
    ui = instance(create=True)
    main(ui)
"""


def as_main_function(expr: Expr) -> str:
    txt = "\n".join(f"    {line}" for line in expr.args)
    return _MACRO_FORMAT.format(txt)


def extract(text: str) -> Expr:
    """Extract the content of main function."""

    macro_expr = parse(text, squeeze=False)
    if macro_expr.args[0].head is Head.import_:
        for line in macro_expr.args:
            if line.head is Head.function and str(line.args[0].args[0]) == "main":
                macro_expr = line.args[1]
                break

    return macro_expr


def get_project_file(path: str | Path):
    """Return the path to the project file."""
    path = Path(path)
    if path.is_dir():
        path = path / "project.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Directory {path} seems not a cylindra project directory. A "
                "project directory should contain a 'project.json' file."
            )
    return path
