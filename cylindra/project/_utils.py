from __future__ import annotations

from pathlib import Path
from typing import Iterable

from macrokit import Expr, Head, parse

_MACRO_FORMAT = """import polars as pl
from cylindra.widgets import CylindraMainWidget
from cylindra import instance
{}
def main(ui: CylindraMainWidget):
{}

if __name__ == "__main__":
    ui = instance(create=True)
    main(ui)
"""


def as_main_function(expr: Expr, imports: Iterable[str] = ()) -> str:
    """Convert a macro to a main(ui) function string."""
    txt = "\n".join(f"    {line}" for line in expr.args)
    if all(line.strip().startswith("#") for line in txt.splitlines()):
        txt += "\n    pass"
    import_statements = "\n".join(imports) + "\n"
    return _MACRO_FORMAT.format(import_statements, txt)


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


def as_relative(
    p: str | Path | None,
    project_dir: Path | None,
    absolute_ok: bool = False,
) -> Path | None:
    if p is None or project_dir is None:
        return None
    if isinstance(p, str):
        p = Path(p)
    if isinstance(p, Path):
        rel_path = Path("..")
        cur_path = project_dir
        for _ in range(5):
            cur_path = cur_path.parent
            if p.is_relative_to(cur_path):
                return rel_path.parent / p.relative_to(cur_path)
            rel_path = rel_path / ".."
        if absolute_ok:
            return p
    return None


def resolve_relative_paths(rel_path: Path, cur_dir: Path) -> Path | None:
    for part in Path(rel_path).parts:
        # NOTE: Path("./XX") -> Path("XX")
        # if part == ".":
        #     continue
        if part.startswith(".."):
            cur_dir = cur_dir.parent
        else:
            cur_dir = cur_dir / part
    if cur_dir.exists():
        return cur_dir
