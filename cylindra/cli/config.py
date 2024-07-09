from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

from cylindra.cli._base import ParserBase
from cylindra.core import collect_projects

if TYPE_CHECKING:
    from cylindra.project import CylindraProject


class InitAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        from cylindra._config import init_config

        init_config(force=True)
        parser.exit()


def list_all_configs():
    import rich
    from rich.panel import Panel

    from cylindra._config import get_config

    for path in get_config().list_config_paths():
        with open(path) as f:
            js = json.load(f)
        assert isinstance(js, dict)

        vals = list[str]()
        for k, v in js.items():
            vals.append(f" {k} = {v!r}")
        title = f"[bold green]{path.stem}[/bold green]"
        rich.print(Panel("\n".join(vals), title=title, border_style="green"))


class ListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        list_all_configs()
        parser.exit()


class ImportAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = Path(namespace.path)
        if not path.exists():
            raise FileNotFoundError(path)
        from cylindra._config import get_config

        save_path = get_config().spline_config_path(path.stem)
        if save_path.exists():
            raise FileExistsError(f"{save_path.as_posix()} already exists.")
        with open(path) as f:
            js = json.load(f)
        assert isinstance(js, dict)
        from cylindra.components import SplineConfig

        cfg = SplineConfig().updated(**js)
        with open(save_path, mode="w") as f:
            json.dump(cfg.asdict(), f, indent=4, separators=(", ", ": "))
        print(f"Config file imported: {save_path.as_posix()}")


class ParserConfig(ParserBase):
    """
    cylindra config [bold green]path[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]path[/u bold green]
        Path to the project/image file.

    [u bold cyan]options[/u bold cyan]
        [bold]--remove, -r[/bold]
            Remove the configuration specific keyword arguments. If a project path is given, also remove the default configuration file.

        [bold]--init, -i[/bold]
            Initialize the default configuration directory. This operation will not remove the user-defined files

        [bold]--list, -l[/bold]
            List up all the available configurations.

        [bold]--import[/bold]
            Import a configuration from a file.
    """

    def __init__(self):
        super().__init__(prog="cylindra config", description="Configure cylindra.")
        self.add_argument("path")
        self.add_argument("--remove", "-r", action="store_true")
        self.add_argument("--init", "-i", action=InitAction, nargs=0)
        self.add_argument("--list", "-l", action=ListAction, nargs=0)
        self.add_argument("--import", action=ImportAction, nargs=0)

    def run_action(self, path: str, remove: bool = False, **kwargs):
        _path = Path(path)
        if path == "list" and not _path.exists():
            return list_all_configs()
        if remove:
            if _path.suffix == ".py":
                if "*" in str(_path):
                    for py_path in glob(_path.as_posix()):
                        self.remove_project_config_kwargs(Path(py_path))
                else:
                    self.remove_project_config_kwargs(_path)
            else:
                for prj in collect_projects(path):
                    assert prj.project_path is not None
                    with prj.open_project() as dir:
                        py_path = prj._script_py_path(dir)
                        self.remove_project_config_kwargs(Path(py_path))
                        cfg_path = prj._default_spline_config_path(dir)
                        if cfg_path.exists():
                            cfg_path.unlink()
                        prj.rewrite(dir)
        else:
            for prj in collect_projects(path):
                self.show_project_default_config(prj)

    def show_project_default_config(self, prj: CylindraProject):
        import rich

        with prj.open_project() as dir:
            if project_path := prj.project_path:
                _p = Path(project_path).as_posix()
                rich.print(f"[bold green]{_p}[/bold green]:")
            with open(prj._default_spline_config_path(dir)) as f:
                js = json.load(f)
            assert isinstance(js, dict)
            for k, v in js.items():
                print(f"    {k} = {v!r}")

    def remove_project_config_kwargs(self, py_path: Path):
        """Remove the config={...} kwargs from the given script.py file."""
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
