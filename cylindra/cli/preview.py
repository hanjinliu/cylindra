from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cylindra.cli._base import ParserBase
from cylindra.core import read_project, view_project

if TYPE_CHECKING:
    import polars as pl


class ParserPreview(ParserBase):
    """
    cylindra preview [bold green]path[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]path[/u bold green]
        Path to the project file. You can use "::" to specify a file inside the project.
        e.g. `cylindra preview "./project.zip::spline-0.json"`

    [u bold cyan]options[/u bold cyan]
        --gui, -g  View the project in a GUI window.
    """

    def __init__(self):
        super().__init__(
            prog="cylindra view", description="View a project, image or others."
        )
        self.add_argument("path", type=str, help="path to the file to view.")
        self.add_argument("--gui", "-g", action="store_true")

    def run_action(self, path: str, gui: bool, **kwargs):
        import rich
        from magicgui.application import use_app

        if "::" in path:
            path, inner_filename = path.split("::")
        else:
            inner_filename = None

        _path = Path(path)
        if not _path.exists():
            raise FileNotFoundError(f"file not found: {_path}")
        if _path.suffix not in ("", ".tar", ".zip", ".json"):
            raise ValueError(f"{path} is not a project file.")
        if gui:
            if inner_filename is not None:
                rich.print(f"[yellow]::{inner_filename} is ignored for --gui.[/yellow]")
            print(f"Previewing project: {_path.as_posix()}")
            view_project(_path, show=not self._IS_TESTING)
            if self.viewer is None and gui:
                use_app().run()
            return
        prj = read_project(_path)
        with prj.open_project() as dir:
            if inner_filename is None:
                self.show_filetree(_path, dir)
            elif inner_filename.endswith(".py"):
                self.show_file(dir / inner_filename)
            elif inner_filename.endswith(".json"):
                self.show_file(dir / inner_filename, lang="json")
            elif inner_filename.endswith((".csv", ".parquet")):
                self.show_molecules(dir / inner_filename)
            else:
                raise ValueError(f"unknown file type: {inner_filename}")

    def show_file(self, path: Path, wrap: bool = False, lang: str = "python"):
        import rich
        from rich.syntax import Syntax

        txt = path.read_text()
        syntax = Syntax(
            txt,
            lang,
            theme="monokai",
            line_numbers=True,
            word_wrap=wrap,
        )
        rich.print(syntax)
        return

    def show_filetree(self, path: Path, dir: Path):
        import rich
        from rich.style import Style
        from rich.syntax import Syntax
        from rich.tree import Tree

        tree = Tree(f"[bold cyan]{path.as_posix()}[/bold cyan]")
        for f in dir.glob("*"):
            match f.suffix:
                case ".tar" | ".zip" | "":
                    tree.add(f.name, style=Style(color="blue"))
                case ".json":
                    if f.stem.startswith("spline-"):
                        tree.add(f.name, style=Style(color="red"))
                    else:
                        syntax = Syntax(
                            f.read_text(),
                            "json",
                            theme="monokai",
                            line_numbers=False,
                            word_wrap=False,
                        )
                        tree.add(boxed(syntax, f.name))
                case ".py":
                    syntax = Syntax(
                        f.read_text(),
                        "python",
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=False,
                    )
                    tree.add(boxed(syntax, f.name))
                case ".csv" | ".parquet":
                    df = read_table(f)
                    tree.add(boxed(render_dataframe(df), f.name))
                case _:
                    tree.add(f.name)

        rich.print(tree)
        return

    def show_molecules(self, path: Path):
        import rich

        df = read_table(path)
        rich.print(render_dataframe(df))
        return


def read_table(path: Path) -> pl.DataFrame:
    import polars as pl

    if path.suffix == ".csv":
        df = pl.read_csv(path)
    elif path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        raise ValueError(f"Cannot open file type {path.suffix}")
    return df


def render_dataframe(df: pl.DataFrame):
    if len(df.columns) < 6 or df.columns[:6] != ["z", "y", "x", "zvec", "yvec", "xvec"]:
        return pl_to_table(df.head(5), elide=len(df) > 5)
    elif len(df.columns) == 6:
        return f"[green]{len(df)} molecules with no features.[/green]"
    else:
        df = df.drop(["z", "y", "x", "zvec", "yvec", "xvec"])
        table = pl_to_table(df.head(5), elide=len(df) > 5)
        table.title = f"[green]{len(df)} molecules with features:[/green]"
        return table


def pl_to_table(df: pl.DataFrame, elide: bool = False):
    import polars as pl
    from rich.box import ROUNDED
    from rich.table import Column, Table

    table = Table(
        *[Column(c, min_width=min(len(c), 5)) for c in df.columns], box=ROUNDED
    )
    for row in df.cast(pl.Utf8).iter_rows():
        table.add_row(*row)
    if elide:
        table.add_row(*["…"] * len(df.columns))
    return table


def boxed(renderable, title: str):
    from rich.table import Table
    from rich.text import Text

    table = Table(title=Text(title, justify="left"), show_header=False, box=None)
    table.add_row(renderable)
    return table
