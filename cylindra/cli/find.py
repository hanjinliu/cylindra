from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from cylindra.core import read_project
from cylindra.cli._base import _ParserBase, get_polars_expr


class ParserFind(_ParserBase):
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
            help="Date in YYMMDD format. date < date-before will be shown.",
        )
        self.add_argument(
            "--date-after",
            "--gt",
            type=int,
            default=0,
            help="Date in YYMMDD format. date > date-after will be shown.",
        )
        self.add_argument(
            "--called",
            type=str,
            default=None,
            help="Projects that called the given method will be shown.",
        )
        self.add_argument(
            "--props",
            "-p",
            type=str,
            default=None,
            nargs="*",
            help="Polars expression for spline global properties to filter output. If multiple expressions are given, they will be combined with `&`.",
        )
        self.add_argument(
            "--absolute", "--abs", action="store_true", help="Show absolute path."
        )
        self.add_argument("--image", action="store_true", help="Show image path.")

    def run_action(
        self,
        pattern: str,
        date_before: int = 999999,
        date_after: int = 0,
        called: str | None = None,
        props: list[str] = [],
        absolute: bool = False,
        image: bool = False,
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
                    if not ptn.match(prj.script_py_path(d).read_text()):
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
                rich.print(f"[cyan]{path.parent.as_posix()}[/cyan]")
            else:
                rich.print(f"[cyan]{path.as_posix()}[/cyan]")
            if image and (img_path := prj.image) is not None:
                print(f"[image] {Path(img_path).as_posix()}")


def get_date(s: str) -> int:
    try:
        return int(s[2:4] + s[5:7] + s[8:10])
    except ValueError:
        raise ValueError(f"invalid date format: {s}, must be YY/MM/DD")
