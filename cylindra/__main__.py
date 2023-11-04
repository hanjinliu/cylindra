from __future__ import annotations

import argparse
import sys
from glob import glob
from enum import Enum
from pathlib import Path
from typing import Any
from cylindra import start, view_project, read_project


class SubCommand(Enum):
    NONE = "none"
    OPEN = "open"
    VIEW = "view"
    PROCESS = "process"


class Namespace(argparse.Namespace):
    arg: Any | None
    init_config: bool
    subcommand: SubCommand
    file: str


class InfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        ...


class Args(argparse.ArgumentParser):
    def __init__(self, prog=None):
        from cylindra import __version__

        super().__init__(prog=prog, description="Command line interface of cylindra.")
        self.add_argument(
            "--init-config",
            action="store_true",
            help="Initialize the configuration file.",
        )
        self.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"cylindra version {__version__}",
            help="Show version.",
        )

    def prep_subcommands(self):
        subparsers = self.add_subparsers()
        sub_open = subparsers.add_parser("open")
        sub_open.set_defaults(subcommand=SubCommand.OPEN)
        subparsers.add_parser("view").set_defaults(subcommand=SubCommand.VIEW)
        sub = subparsers.add_parser("process")
        sub.set_defaults(subcommand=SubCommand.PROCESS)
        sub.add_argument(
            "-f",
            "--file",
            help="Python file containing the processing pipeline.",
        )

    @classmethod
    def from_args(cls) -> Namespace:
        self = cls()
        self.prep_subcommands()
        ns, argv = self.parse_known_args()
        ns: Namespace
        nargv = len(argv)
        if nargv == 0:
            ns.arg = None
        elif nargv == 1:
            ns.arg = argv[0]
        else:
            raise ValueError(f"too many arguments: {argv}")
        if not hasattr(ns, "subcommand"):
            ns.subcommand = SubCommand.NONE
        return ns


def view_csv(path: Path):
    from cylindra._previews import view_tables

    if "*" in path.as_posix():
        return view_tables(glob(path.as_posix()))
    else:
        return view_tables(path)


def view_image(path: Path):
    from cylindra._previews import view_image

    if "*" in path.as_posix():
        return view_image(glob(path.as_posix()))
    else:
        return view_image(path)


def main(viewer=None):  # "viewer" is used for testing only
    """The main function of the CLI."""
    args = Args.from_args()
    block = viewer is None

    if args.init_config:
        if args.subcommand is not SubCommand.NONE:
            raise ValueError("cannot use --init-config with subcommands.")
        from cylindra._config import init_config

        return init_config(force=True)

    match args.subcommand:
        case SubCommand.NONE:
            ui = start(args.arg, viewer=viewer)
            ui.parent_viewer.show(block=block)
        case SubCommand.OPEN:
            read_project(args.arg)  # check if the project is valid
            ui = start(args.arg, viewer=viewer)
            ui.parent_viewer.show(block=block)
        case SubCommand.VIEW:
            path = Path(args.arg)
            if not path.exists():
                raise FileNotFoundError(f"file not found: {path}")
            match path.suffix:
                case "" | ".tar" | ".zip":
                    view_project(path)
                case ".tif" | ".tiff" | ".mrc" | ".map":
                    view_image(path).show()
                case ".csv" | ".parquet":
                    view_csv(path).show()
                case _:
                    raise ValueError(f"unknown file type: {path.suffix}")
            if block:
                from magicgui.application import use_app

                use_app().run()
        case SubCommand.PROCESS:
            from runpy import run_path

            ui = start(args.arg, viewer=viewer, headless=True)
            out_globs = run_path(args.file, {"ui": ui})
            if callable(main := out_globs.get("main")):
                main(ui)  # script.py style
            ui.overwrite_project()
        case _:
            raise ValueError(f"unknown subcommand: {args.subcommand}")
    sys.exit()


if __name__ == "__main__":
    main()
