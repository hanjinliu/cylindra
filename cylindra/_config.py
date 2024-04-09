from __future__ import annotations

import atexit
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

from appdirs import user_config_dir

VAR_PATH = Path(user_config_dir("variables", "cylindra"))
SETTINGS_DIR = Path(user_config_dir("settings", "cylindra"))
WORKFLOWS_DIR = Path(user_config_dir("workflows", "cylindra"))
STASH_DIR = Path(user_config_dir("stash", "cylindra"))
TEMPLATE_PATH_HIST = SETTINGS_DIR / "template_path_hist.txt"
RECOVERY_PATH = SETTINGS_DIR / "last_project.zip"

USER_SETTINGS = SETTINGS_DIR / "user-settings.json"

WORKFLOW_TEMPLATE = """import numpy as np
import impy as ip
import polars as pl
from pathlib import Path
from cylindra.widgets import CylindraMainWidget

def main(ui: 'CylindraMainWidget'):
    {}

"""


@dataclass
class AppConfig:
    """Application configuration."""

    default_spline_config: str = "eukaryotic_MT"
    dask_chunk: tuple[int, int, int] = (256, 256, 256)
    point_size: float = 4.2
    molecules_color: str = "#00EA00"
    molecules_ndim: int = 3
    autosave_interval: float = 60.0
    default_dask_n_workers: int | None = None
    use_gpu: bool = True

    @classmethod
    def from_user_dir(cls, ignore_error: bool = False) -> AppConfig:
        if not USER_SETTINGS.exists():
            return cls()
        with open(USER_SETTINGS) as f:
            js = json.load(f)
            if "dask_chunk" in js:
                js["dask_chunk"] = tuple(js["dask_chunk"])
                assert len(js["dask_chunk"]) == 3
        try:
            self = AppConfig(**js)
        except Exception as e:
            if ignore_error:
                print("Failed to load user settings. Initialize AppConfig.")
                self = AppConfig()
            else:
                raise e
        return self

    def to_user_dir(self):
        with open(USER_SETTINGS, mode="w") as f:
            json.dump(asdict(self), f, indent=4, separators=(", ", ": "))
        return None

    @property
    def default_spline_config_path(self) -> Path:
        return self.spline_config_path(self.default_spline_config)

    def spline_config_path(self, name: str) -> Path:
        """Get the spline config path for the given config name."""
        return VAR_PATH / f"{name}.json"

    def list_config_paths(self) -> list[Path]:
        """List up all the available configs."""
        return list(VAR_PATH.glob("*.json"))

    def list_workflow_paths(self) -> list[Path]:
        """List up all the available workflows."""
        return list(WORKFLOWS_DIR.glob("*.py"))


_APP_CONFIG: AppConfig | None = None


@atexit.register
def _save_config():  # pragma: no cover
    if _APP_CONFIG is not None:
        try:
            _APP_CONFIG.to_user_dir()
        except Exception as e:
            print(f"Failed to save user settings: {e}")


def autosave_path() -> Path:
    return RECOVERY_PATH


def get_config() -> AppConfig:
    global _APP_CONFIG
    if _APP_CONFIG is None:
        _APP_CONFIG = AppConfig.from_user_dir(ignore_error=True)
    return _APP_CONFIG


def workflow_path(name: str | Path) -> Path:
    """Path to the workflow file."""
    out = WORKFLOWS_DIR / name
    if out.suffix != ".py":
        out = out.with_suffix(".py")
    return out


@contextmanager
def patch_workflow_path(dir: str | Path):
    """Temporarily change the workflow directory."""
    global WORKFLOWS_DIR

    dir = Path(dir)
    assert dir.is_dir()
    old_dir = WORKFLOWS_DIR
    WORKFLOWS_DIR = dir
    try:
        yield
    finally:
        WORKFLOWS_DIR = old_dir


@contextmanager
def patch_stash_dir(dir: str | Path):
    """Temporarily change the stash directory."""
    global STASH_DIR

    dir = Path(dir)
    assert dir.is_dir()
    old_dir = STASH_DIR
    STASH_DIR = dir
    try:
        yield
    finally:
        STASH_DIR = old_dir


@contextmanager
def patch_config_dir(dir: str | Path):
    """Temporarily change the config directory."""
    global VAR_PATH

    dir = Path(dir)
    assert dir.is_dir()
    old_dir = VAR_PATH
    VAR_PATH = dir
    try:
        yield
    finally:
        VAR_PATH = old_dir


def get_main_function(filename: str | Path) -> Callable:
    """Get the main function object from the file."""
    from runpy import run_path

    path = Path(filename)
    if not path.exists():
        path = workflow_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
    if path.is_dir():
        raise ValueError("You must specify a file.")

    ns = run_path(str(path))
    if callable(main := ns.get("main")):

        @wraps(main)
        def _main(*args, **kwargs):
            # TODO: stream the output to the logger optionally.
            out = main(*args, **kwargs)
            return out

        return _main
    raise ValueError(f"No main function found in file {path.as_posix()}.")


def init_config(force: bool = False):  # pragma: no cover
    # Initialize user config directory.
    if force or not VAR_PATH.exists() or _is_empty(VAR_PATH):
        try:
            if not VAR_PATH.exists():
                VAR_PATH.mkdir(parents=True)

            _data_dir = Path(__file__).parent / "_data"
            for fp in _data_dir.glob("*.json"):
                with open(fp) as f:
                    js = json.load(f)

                with open(VAR_PATH / fp.name, mode="w") as f:
                    json.dump(js, f, indent=4, separators=(", ", ": "))

        except Exception as e:
            print("Failed to initialize config directory.")
            print(e)
        else:
            print(f"Config directory initialized at {VAR_PATH}.")

    # Initialize user settings directory.
    if not SETTINGS_DIR.exists() or _is_empty(SETTINGS_DIR):
        try:
            if not SETTINGS_DIR.exists():
                SETTINGS_DIR.mkdir(parents=True)
            AppConfig().to_user_dir()
        except Exception as e:
            print("Failed to initialize settings directory.")
            print(e)
        else:
            print(f"Settings directory initialized at {SETTINGS_DIR}.")

    if not WORKFLOWS_DIR.exists():
        try:
            WORKFLOWS_DIR.mkdir(parents=True)
        except Exception as e:
            print("Failed to initialize workflows directory.")
            print(e)
        else:
            print(f"Workflows directory initialized at {WORKFLOWS_DIR}.")
        Path(WORKFLOWS_DIR).joinpath("example_workflow.py").write_text(
            WORKFLOW_TEMPLATE.format(
                "\n    ".join(
                    [
                        "nsplines = len(ui.splines)",
                        "for i in range(nsplines):",
                        "    spl = ui.splines[i]",
                        "    print(f'{i}: {spl.length():.2f} nm')",
                    ]
                )
            )
        )


def _is_empty(path: Path) -> bool:
    """Check if a directory is empty."""
    it = path.glob("*")
    try:
        next(it)
    except StopIteration:
        return True
    return False


def get_template_path_hist() -> list[Path]:
    if TEMPLATE_PATH_HIST.exists():
        out = list[Path]()
        for line in TEMPLATE_PATH_HIST.read_text().splitlines():
            if line.strip() == "":
                continue
            try:
                out.append(Path(line))
            except ValueError:
                pass
        return out
    return []


def set_template_path_hist(paths: list[str]):
    path = TEMPLATE_PATH_HIST

    def _is_file_and_exists(p: str) -> bool:
        p0 = Path(p)
        return p0.is_file() and p0.exists()

    try:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        path.write_text("\n".join([p for p in paths if _is_file_and_exists(p)]) + "\n")
    except Exception:
        pass
    return None


def get_stash_list() -> list[str]:
    """Get the list of stashed project names."""
    if STASH_DIR.exists():
        return [path.name for path in STASH_DIR.glob("*")]
    return []


def get_stash_dir() -> Path:
    """Get the stash directory."""
    if not STASH_DIR.exists():
        STASH_DIR.mkdir(parents=True)
    return STASH_DIR
