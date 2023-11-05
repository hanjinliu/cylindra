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


class ParserBase(argparse.ArgumentParser):
    viewer: Any

    def parse(self, args=None):
        ns = self.parse_args(args)
        return self.run_action(**vars(ns))

    def run_action(self, *args, **kwargs):
        """The subcommand actions."""
        raise NotImplementedError


class ParserNone(ParserBase):
    def __init__(self):
        from cylindra import __version__

        super().__init__(
            prog="cylindra", description="Command line interface of cylindra."
        )
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

    def run_action(self, init_config: bool):
        if init_config:
            from cylindra._config import init_config

            return init_config(force=True)
        else:
            ui = start(viewer=self.viewer)
            ui.parent_viewer.show(block=self.viewer is None)
            return None


class ParserOpen(ParserBase):
    """cylindra open <path>"""

    def __init__(self):
        super().__init__(prog="cylindra open", description="Open a project.")
        self.add_argument("path", type=str, help="path to the project file.")

    def run_action(self, path: str):
        read_project(path)  # check if the project is valid
        ui = start(path, viewer=self.viewer)
        ui.parent_viewer.show(block=self.viewer is None)


class ParserView(ParserBase):
    """cylindra view <path>"""

    def __init__(self):
        super().__init__(
            prog="cylindra view", description="View a project, image or others."
        )
        self.add_argument("path", type=str, help="path to the file to view.")

    def run_action(self, path: str):
        _path = Path(path)
        if not _path.exists():
            raise FileNotFoundError(f"file not found: {_path}")
        match _path.suffix:
            case "" | ".tar" | ".zip" | ".json":
                view_project(_path)
            case ".tif" | ".tiff" | ".mrc" | ".map":
                view_image(_path).show()
            case ".csv" | ".parquet":
                view_csv(_path).show()
            case _:
                raise ValueError(f"unknown file type: {_path.suffix}")
        if self.viewer is None:
            from magicgui.application import use_app

            use_app().run()


class ParserRun(ParserBase):
    """cylindra run <path>"""

    def __init__(self):
        super().__init__(prog="cylindra run", description="Run a script.")
        self.add_argument("path", type=str)
        self.add_argument(
            "--headless", action="store_true", help="Run in headless mode."
        )

    def run_action(self, path: str, headless: bool):
        from runpy import run_path

        ui = start(viewer=self.viewer, headless=headless)
        out_globs = run_path(path, {"ui": ui})
        if callable(main := out_globs.get("main")):
            main(ui)  # script.py style
        ui.overwrite_project()


def main(viewer=None):
    argv = sys.argv[1:]
    ParserBase.viewer = viewer
    match argv:
        case ("open", *args):
            ParserOpen().parse(args)
        case ("view", *args):
            ParserView().parse(args)
        case ("run", *args):
            ParserRun().parse(args)
        case args:
            ParserNone().parse(args)


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


if __name__ == "__main__":
    main()
    sys.exit()
