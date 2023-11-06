from __future__ import annotations

import argparse
import sys
import warnings
from glob import glob
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
import math
from cylindra import start, view_project, read_project, collect_projects


class HelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(parser.__doc__)
        parser.exit()


class ParserBase(argparse.ArgumentParser):
    viewer: Any

    def parse(self, args=None):
        ns = self.parse_args(args)
        return self.run_action(**vars(ns))

    def run_action(self, *args, **kwargs):
        """The subcommand actions."""
        raise NotImplementedError


class ParserNone(ParserBase):
    """
    Usage: cylindra [OPTIONS] COMMAND [ARGUMENTS]

    Options:
        --init-config   Initialize the configuration file.
        -v, --version   Show version.
        -h, --help      Show this message and exit.

    Commands:
        open       Open a project or an image.
        preview    View a project, image or others.
        run        Run a script.
        average    Average images.
        new        Create a new project.
        add        Add splines by coordinates.
    """

    def __init__(self):
        from cylindra import __version__

        super().__init__(
            prog="cylindra",
            description="Command line interface of cylindra.",
            add_help=False,
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
        self.add_argument(
            "-h",
            "--help",
            nargs=0,
            action=HelpAction,
        )

    def run_action(self, init_config: bool, **kwargs):
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
        super().__init__(
            prog="cylindra open",
            description="Open a project or an image file.",
        )
        self.add_argument(
            "path",
            type=str,
            help="path to the project/image file.",
        )
        self.add_argument(
            "--scale",
            "-s",
            type=float,
            help=(
                "Scale (nm/pixel) of the image, if an image file is given as the "
                "first argument. If not given, it will be read from the image "
                "file. This parameter will be ignored if a project is given."
            ),
            default=None,
        )
        self.add_argument(
            "--missing_wedge",
            "--mw",
            nargs=2,
            type=float,
            help="Min/max tilt angles. This parameter will be ignored if a project is given.",
            default=None,
        )

    def run_action(
        self,
        path: str,
        scale: float | None = None,
        missing_wedge: tuple[float, float] | None = None,
    ):
        fp = Path(path)
        match fp.suffix:
            case ".tif" | ".tiff" | ".mrc" | ".map":
                print(f"Opening image: {fp.as_posix()}")
                import impy as ip

                ui = start(viewer=self.viewer)
                img = ip.lazy.imread(path)
                scale = img.scale.x
                bin_size = int(math.ceil(0.96 / scale))
                ui.open_image(
                    fp, scale=scale, bin_size=[bin_size], tilt_range=missing_wedge
                )
            case "" | ".tar" | ".zip" | ".json":
                read_project(fp)  # check if the project is valid
                print(f"Opening project: {fp.as_posix()}")
                if not (scale is None and missing_wedge is None):
                    warnings.warn(
                        "scale and tilt are ignored for project input.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                ui = start(fp, viewer=self.viewer)
            case _:
                raise ValueError(f"unknown file type: {fp.suffix}")
        ui.parent_viewer.show(block=self.viewer is None)


class ParserNew(ParserBase):
    def __init__(self):
        super().__init__(prog="cylindra new", description="Create a new project.")
        self.add_argument("output", type=str, help="The project file")
        self.add_argument("--image", type=str, help="image file.")
        self.add_argument(
            "--multiscales",
            nargs="+",
            type=float,
            help="Bin sizes.",
            default=None,
        )
        self.add_argument(
            "--scale",
            type=float,
            help="scale (nm/pixel) of the image.",
            default=None,
        )
        self.add_argument(
            "--missing_wedge",
            "--mw",
            nargs=2,
            type=float,
            help="min/max tilt angles in degree that defines the missing wedge.",
            default=None,
        )
        self.add_argument(
            "--molecules",
            "-m",
            nargs="*",
            help="molecules files.",
            default=[],
        )

    def run_action(
        self,
        output: str,
        image: str,
        multiscales: list[float] | None = None,
        scale: float | None = None,
        missing_wedge: tuple[float, float] | None = None,
        molecules: list[str] = [],
    ):
        from cylindra.project import CylindraProject
        from acryo import Molecules

        path = Path(output)
        prj = CylindraProject.new(image, scale, multiscales, missing_wedge, path)
        for mole in molecules:
            mole = Molecules.from_file(mole)
        prj.save(path)
        return print(f"Project created at: {path.as_posix()}")


class ParserPreview(ParserBase):
    """cylindra preview <path>"""

    def __init__(self):
        super().__init__(
            prog="cylindra view", description="View a project, image or others."
        )
        self.add_argument("path", type=str, help="path to the file to view.")

    def run_action(self, path: str):
        from cylindra._previews import view_tables, view_image
        from magicgui.application import use_app

        _path = Path(path)
        if not _path.exists():
            raise FileNotFoundError(f"file not found: {_path}")
        match _path.suffix:
            case "" | ".tar" | ".zip" | ".json":
                print(f"Previewing project: {_path.as_posix()}")
                view_project(_path)
            case ".tif" | ".tiff" | ".mrc" | ".map":
                print(f"Previewing image: {_path.as_posix()}")
                if "*" in _path.as_posix():
                    view_image(glob(_path.as_posix()))
                else:
                    view_image(_path)
            case ".csv" | ".parquet":
                print(f"Previewing table: {_path.as_posix()}")
                if "*" in _path.as_posix():
                    view_tables(glob(_path.as_posix()))
                else:
                    view_tables(_path)
            case _:
                raise ValueError(f"unknown file type: {_path.suffix}")
        if self.viewer is None:
            use_app().run()


class ParserRun(ParserBase):
    """cylindra run <path>"""

    def __init__(self):
        super().__init__(prog="cylindra run", description="Run a script.")
        self.add_argument("path", type=str, help="Python script to run.")
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


class ParserAverage(ParserBase):
    def __init__(self):
        super().__init__(prog="cylindra average", description="Average images.")
        self.add_argument(
            "project",
            type=str,
            help="Path to the project file. Can be a pattern containing `*`.",
        )
        self.add_argument(
            "--molecules",
            "-m",
            type=str,
            help="File name pattern of molecules.",
        )
        self.add_argument(
            "--size",
            "-s",
            type=float,
            help="Size of subtomograms in nm.",
        )
        self.add_argument(
            "--output",
            "-o",
            type=str,
            help="Output file name for the averaged image.",
            default=None,
        )

    def run_action(
        self,
        project: str,
        molecules: str,
        size: float,
        output: str | None = None,
    ):
        import impy as ip

        col = collect_projects(project)
        if "*" in molecules:
            name_filter = lambda n: fnmatch(n, molecules)
        else:
            name_filter = lambda n: n == molecules
        loader = col.sta_loader(name_filter)
        shape = (int(round(size / loader.scale)),) * 3
        print("Averaging...")
        avg = loader.average(output_shape=shape)
        avg = ip.asarray(avg, axes="zyx").set_scale(xyz=loader.scale, unit="nm")
        if output is None:
            cwd = Path.cwd()
            save_path = cwd / "AVG.tif"
            suffix = 0
            while save_path.exists():
                save_path = cwd / f"AVG-{suffix}.tif"
                suffix += 1
            output = save_path
        avg.imsave(output)
        print(f"Average image saved at: {output}")


def main(viewer=None):
    argv = sys.argv[1:]
    ParserBase.viewer = viewer
    match argv:
        case ("open", *args):
            ParserOpen().parse(args)
        case ("preview", *args):
            ParserPreview().parse(args)
        case ("run", *args):
            ParserRun().parse(args)
        case ("average", *args):
            ParserAverage().parse(args)
        case ("new", *args):
            ParserNew().parse(args)
        case args:
            ParserNone().parse(args)


if __name__ == "__main__":
    main()
    sys.exit()
