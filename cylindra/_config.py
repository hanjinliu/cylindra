from __future__ import annotations
from pathlib import Path
import json
from typing import Callable

from appdirs import user_config_dir


VAR_PATH = Path(user_config_dir("variables", "cylindra"))
SETTINGS_DIR = Path(user_config_dir("settings", "cylindra"))
WORKFLOWS_DIR = Path(user_config_dir("workflows", "cylindra"))
TEMPLATE_PATH_HIST = "template_path_hist.txt"
DEFAULT_VARIABLES = "default_variables"

USER_SETTINGS = SETTINGS_DIR / "user-settings.json"


def workflow_path(name: str | Path) -> Path:
    """Path to the workflow file."""
    out = WORKFLOWS_DIR / name
    if out.suffix != ".py":
        out = out.with_suffix(".py")
    return out


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
        return main
    raise ValueError(f"No main function found in file {path.as_posix()}.")


def init_config(force: bool = False):  # pragma: no cover
    # Initialize user config directory.
    if force:
        import shutil

        if VAR_PATH.exists():
            shutil.rmtree(VAR_PATH)
        if SETTINGS_DIR.exists():
            shutil.rmtree(SETTINGS_DIR)
        if WORKFLOWS_DIR.exists():
            shutil.rmtree(WORKFLOWS_DIR)

    if not VAR_PATH.exists() or _is_empty(VAR_PATH):
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

            settings_js = {DEFAULT_VARIABLES: "eukaryotic_MT"}
            with open(USER_SETTINGS, mode="w") as f:
                json.dump(settings_js, f, indent=4, separators=(", ", ": "))
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


def _is_empty(path: Path) -> bool:
    """Check if a directory is empty."""
    it = path.glob("*")
    try:
        next(it)
    except StopIteration:
        return True
    return False


def get_template_path_hist() -> list[Path]:
    path = Path(SETTINGS_DIR / TEMPLATE_PATH_HIST)
    if path.exists():
        out = list[Path]()
        for line in path.read_text().splitlines():
            if line.strip() == "":
                continue
            try:
                out.append(Path(line))
            except ValueError:
                pass
        return out
    return []


def set_template_path_hist(paths: list[str]):
    path = Path(SETTINGS_DIR / TEMPLATE_PATH_HIST)
    try:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        path.write_text("\n".join([p for p in paths if Path(p).is_file()]) + "\n")
    except Exception:
        pass
    return None
