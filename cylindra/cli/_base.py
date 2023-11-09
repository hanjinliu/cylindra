import argparse
from typing import Any
from pathlib import Path


class ParserBase(argparse.ArgumentParser):
    viewer: Any

    def parse(self, args=None):
        ns = self.parse_args(args)
        return self.run_action(**vars(ns))

    def run_action(self, *args, **kwargs):
        """The subcommand actions."""
        raise NotImplementedError


def coerce_output_filename(name: str, ext: str = ".tif"):
    cwd = Path.cwd()
    save_path = cwd / f"{name}{ext}"
    suffix = 0
    while save_path.exists():
        save_path = cwd / f"{name}-{suffix}{ext}"
        suffix += 1
    return save_path


def get_polars_expr(expr: str):
    import polars as pl
    import numpy as np

    ns = {
        "pl": pl,
        "col": pl.col,
        "when": pl.when,
        "format": pl.format,
        "int": int,
        "float": float,
        "str": str,
        "np": np,
        "__builtins__": {},
    }

    out = eval(expr, ns, {})
    if not isinstance(out, pl.Expr):
        raise TypeError(f"{expr!r} did not return an expression, got {type(out)}")
    return out


class HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import rich
        from rich.panel import Panel

        doc = parser.__doc__
        assert doc is not None
        lines = doc.splitlines()
        while lines[0].strip() == "":
            lines.pop(0)
        line0 = lines.pop(0)
        nindents = len(line0) - len(line0.lstrip())
        rich.print(Panel(line0[nindents:]))
        for line in lines:
            if "[" in line:
                rich.print(line[nindents:])
            else:
                print(line[nindents:])
        parser.exit()
