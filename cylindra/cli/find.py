from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from cylindra.core import read_project
from cylindra.cli._base import _ParserBase, get_polars_expr


class ParserFind(_ParserBase):
    def __init__(self):
        super().__init__(prog="cylindra find", description="Find projects.")
        self.add_argument("pattern", type=str, default="*", nargs="?")
        self.add_argument("--date-before", type=int, default=999999)
        self.add_argument("--date-after", type=int, default=0)
        self.add_argument("--called", type=str, default=None)
        self.add_argument("--props", type=str, default=None)

    def run_action(
        self,
        pattern: str,
        date_before: int = 999999,
        date_after: int = 0,
        called: str | None = None,
        props: str | None = None,
    ):
        if called is not None:
            if called.startswith("ui."):
                called = called[3:]
            ptn = re.compile(rf".*ui\.{called}\(.*\).*")
        if props:
            pl_props = get_polars_expr(props)

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

            if path.name == "project.json":
                print(path.parent.as_posix())
            else:
                print(path.as_posix())


def get_date(s: str) -> int:
    return int(s[2:4] + s[5:7] + s[8:10])
