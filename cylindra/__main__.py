from __future__ import annotations

import argparse
import sys
import warnings
import json
from glob import glob
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, TYPE_CHECKING
import math
from cylindra import start, view_project, read_project, collect_projects

if TYPE_CHECKING:
    from cylindra.project import CylindraProject


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

    def run_action(self, **kwargs):
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

    class ShowScriptAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            ln = True
            if _path := getattr(namespace, "path", None):
                prj = read_project(_path)
                with prj.open_project() as dir:
                    txt = prj.script_py_path(dir).read_text()
                    self.print_rich(txt, line_numbers=ln)
            else:
                raise ValueError("path is not given.")
            return parser.exit()

        def print_rich(self, txt: str, line_numbers: bool = True):
            import rich
            from rich.syntax import Syntax

            syntax = Syntax(
                txt,
                "python",
                theme="monokai",
                line_numbers=line_numbers,
            )
            rich.print(syntax)

    def __init__(self):
        super().__init__(
            prog="cylindra view", description="View a project, image or others."
        )
        self.add_argument("path", type=str, help="path to the file to view.")
        self.add_argument(
            "--script",
            "-s",
            nargs="*",
            action=self.ShowScriptAction,
            help="Only preview script.py content.",
        )

    def run_action(self, path: str, **kwargs):
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
        self.add_argument(
            "--output", "-o", type=str, help="Output file name.", default=None
        )

    def run_action(self, path: str, headless: bool = False, output: str = None):
        from runpy import run_path

        ui = start(viewer=self.viewer, headless=headless)
        if Path(path).suffix in ("", ".tar", ".zip", ".json"):
            prj = read_project(path)
            with prj.open_project() as d:
                py_path = str(prj.script_py_path(d))
                out_globs = run_path(py_path, {"ui": ui})
            is_project = True
        else:
            out_globs = run_path(path, {"ui": ui})
            is_project = False
        if callable(main := out_globs.get("main")):
            main(ui)  # script.py style
        if output is None:
            if is_project:
                ui.overwrite_project()
            else:
                output = _coerce_output_filename("output", ext="")
                ui.save_project(output, molecules_ext=".parquet")
        else:
            ui.save_project(output, molecules_ext=".parquet")


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
            output = _coerce_output_filename("AVG", ext=".tif")
        avg.imsave(output)
        print(f"Average image saved at: {output}")


class ParserConfig(ParserBase):
    class InitAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            from cylindra._config import init_config

            init_config(force=True)
            parser.exit()

    class ListAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            from cylindra._config import get_config
            import rich

            for path in get_config().list_config_paths():
                with open(path) as f:
                    js = json.load(f)
                assert isinstance(js, dict)

                rich.print(f"[bold green]{path.stem}[/bold green]:")
                for k, v in js.items():
                    print(f"    {k} = {v!r}")
            parser.exit()

    def __init__(self):
        super().__init__(prog="cylindra config", description="Configure cylindra.")
        self.add_argument("path")
        self.add_argument(
            "--remove",
            "-r",
            action="store_true",
            help="Remove the configuration specific keyword arguments. If a project path is given, also remove the default configuration file.",
        )
        self.add_argument(
            "--init",
            "-i",
            action=self.InitAction,
            nargs=0,
            help="Initialize the default configuration directory. This operation will not remove the user-defined files",
        )
        self.add_argument("--list", "-l", action=self.ListAction, nargs=0)

    def run_action(self, path: str, remove: bool = False, **kwargs):
        _path = Path(path)
        if remove:
            if _path.suffix == ".py":
                if "*" in str(_path):
                    for py_path in glob(_path.as_posix()):
                        self.remove_project_config_kwargs(Path(py_path))
                else:
                    self.remove_project_config_kwargs(_path)
            else:
                for prj in collect_projects(path):
                    with prj.open_project() as dir:
                        py_path = prj.script_py_path(dir)
                        self.remove_project_config_kwargs(Path(py_path))
                        cfg_path = prj.default_spline_config_path(dir)
                        if cfg_path.exists():
                            cfg_path.unlink()
                            print(f"Removed: {cfg_path.as_posix()}")
        else:
            for prj in collect_projects(path):
                self.show_project_default_config(prj)

    def show_project_default_config(self, prj: CylindraProject):
        import rich

        with prj.open_project() as dir:
            if project_path := prj.project_path:
                _p = Path(project_path).as_posix()
                rich.print(f"[bold green]{_p}[/bold green]:")
            with open(prj.default_spline_config_path(dir)) as f:
                js = json.load(f)
            assert isinstance(js, dict)
            for k, v in js.items():
                print(f"    {k} = {v!r}")

    def remove_project_config_kwargs(self, py_path: Path):
        import re

        ptn = re.compile(r", config=\{.*\}")
        _edit_prefix = (
            "ui.register_path(",
            "ui.protofilaments_to_spline(",
            "ui.molecules_to_spline(",
        )
        original = py_path.read_text()
        lines = original.splitlines()
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped.startswith(_edit_prefix):
                continue
            lines[i] = ptn.sub("", line)

        edited = "\n".join(lines)
        if original != edited:
            py_path.write_text(edited)
            print(f"Processed: {py_path.as_posix()}")


def _coerce_output_filename(name: str, ext: str = ".tif"):
    cwd = Path.cwd()
    save_path = cwd / f"{name}{ext}"
    suffix = 0
    while save_path.exists():
        save_path = cwd / f"{name}-{suffix}{ext}"
        suffix += 1
    return save_path


def main(viewer=None, ignore_sys_exit: bool = False):
    argv = sys.argv[1:]
    ParserBase.viewer = viewer
    try:
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
            case ("config", *args):
                ParserConfig().parse(args)
            case args:
                ParserNone().parse(args)
    except SystemExit as e:
        if ignore_sys_exit:
            return
        else:
            raise e


if __name__ == "__main__":
    main()
    sys.exit()
