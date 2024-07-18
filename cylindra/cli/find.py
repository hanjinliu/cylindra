from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

from cylindra.cli._base import ParserBase, get_polars_expr
from cylindra.core import read_project


class ParserFind(ParserBase):
    """
    cylindra find [bold green]pattern[/bold green] [bold yellow]options[/bold yellow] [bold cyan]flags[/bold cyan]

    [u bold green]pattern[/u bold green] (positional)
        File pattern to match projects.
        e.g. `cylindra find **/*.tar`

    [u bold yellow]options[/u bold yellow]
        --date-before, --lt [grey50]DATE[/grey50]
            Date in YYMMDD format.
            DATE < date-before will be shown.

        --date-after, --gt [grey50]DATE[/grey50]
            Date in YYMMDD format.
            DATE > date-after will be shown.

        --called [grey50]METHOD[/grey50]
            Projects that called METHOD will be shown.
            e.g. `cylindra find --called filter_molecules`

        --props, -p [grey50]EXPR0, EXPR1, ...[grey50]
            Polars expression to filter.
            e.g. `cylindra find --props "col('npf')==13"`
            e.g. `cylindra find --props "col('npf')==13" "col('start')==3"`

    [u bold cyan]flags[/u bold cyan]
        --absolute, --abs
            Show absolute path.

        --image
            Show image path.

        --config
            Show config content.

        --description, --desc
            Show project description.
    """

    def __init__(self):
        super().__init__(prog="cylindra find", description="Find projects.")
        self.add_argument(
            "pattern",
            type=str,
            default="*",
            nargs="?",
            help="glob pattern for project files",
        )
        self.add_argument(
            "--date-before",
            "--lt",
            type=int,
            default=999999,
        )
        self.add_argument(
            "--date-after",
            "--gt",
            type=int,
            default=0,
        )
        self.add_argument(
            "--called",
            type=str,
            default=None,
        )
        self.add_argument(
            "--props",
            "-p",
            type=str,
            default=None,
            nargs="*",
        )
        self.add_argument("--absolute", "--abs", action="store_true")
        self.add_argument("--image", action="store_true")
        self.add_argument("--config", action="store_true")
        self.add_argument(
            "--description",
            "--desc",
            action="store_true",
        )

    def run_action(
        self,
        pattern: str,
        date_before: int = 999999,
        date_after: int = 0,
        called: str | None = None,
        props: list[str] = [],
        absolute: bool = False,
        image: bool = False,
        config: bool = False,
        description: bool = False,
        **kwargs,
    ):
        import rich

        if called is not None:
            if called.startswith("ui."):
                called = called[3:]
            ptn = re.compile(rf".*ui\.{called}\(.*\).*", flags=re.DOTALL)
        if props:
            pl_props = get_polars_expr(props[0])
            if len(props) > 1:
                for p in props[1:]:
                    pl_props = pl_props & get_polars_expr(p)

        for fp in glob(pattern, recursive=True):
            path = Path(fp)
            if path.is_dir():
                continue
            if path.name != "project.json" and path.suffix not in (".tar", ".zip"):
                continue
            try:
                prj = read_project(path)
            except Exception:
                continue

            if not date_after < get_date(prj.datetime) < date_before:
                continue
            with prj.open_project() as d:
                if called is not None:
                    if not ptn.match(prj._script_py_path(d).read_text()):
                        continue
                if props is not None:
                    for spl in prj.iter_load_splines(d, drop_columns=False):
                        try:
                            result = spl.props.get_glob(pl_props)
                        except Exception:
                            result = False
                        if result:
                            break
                    else:
                        continue

                if absolute:
                    path = path.absolute()
                if path.name == "project.json":
                    rich.print(f"[bold cyan]{path.parent.as_posix()}[/bold cyan]")
                else:
                    rich.print(f"[bold cyan]{path.as_posix()}[/bold cyan]")
                if image and (img_path := prj.image) is not None:
                    print(f"[image] {Path(img_path).as_posix()}")
                if config and (cpath := prj._default_spline_config_path(d)).exists():
                    txt = json.loads(cpath.read_text().strip())
                    print(f"[config] {txt}")
                if description:
                    if prj.project_description:
                        print(rf"\[description] [gray]{prj.project_description}[/gray]")
                    else:
                        rich.print(r"\[description] [italic]no description[/italic]")


def get_date(s: str) -> int:
    try:
        return int(s[2:4] + s[5:7] + s[8:10])
    except ValueError:
        raise ValueError(f"invalid date format: {s}, must be YY/MM/DD")
