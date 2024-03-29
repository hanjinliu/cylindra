import argparse
import os
from pathlib import Path
from typing import Any


class ParserBase(argparse.ArgumentParser):
    viewer: Any
    _IS_TESTING = False  # patch for testing

    def __init__(self, prog: str, description: str, add_help: bool = False):
        super().__init__(
            prog=prog,
            description=description,
            add_help=add_help,
        )
        self.add_argument("-h", "--help", nargs=0, action=HelpAction)

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
    import numpy as np
    import polars as pl

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
        try:
            ncols = os.get_terminal_size().columns
        except OSError:
            # during testing
            ncols = 50
        lines = doc.splitlines()
        while lines[0].strip() == "":
            lines.pop(0)
        line0 = lines.pop(0)
        nindents = count_indent(line0)
        rich.print(Panel(line0[nindents:]))
        for line in lines:
            if "[" in line:
                rich.print(line[nindents:])
            else:
                indent_ = " " * count_indent(line[nindents:])
                # print considering the terminal size
                c = len(indent_)
                print(indent_, end="")
                for word in line.lstrip().split(" "):
                    if c + len(word) > ncols:
                        print()
                        print(indent_, end="")
                        c = len(indent_)
                    print(" " + word, end="")
                    c += len(word) + 1
                print()
        parser.exit()


def count_indent(line: str) -> int:
    return len(line) - len(line.lstrip())
