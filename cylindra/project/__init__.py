from pathlib import Path
from ._single import CylindraProject
from ._batch import CylindraBatchProject
from ._widgets import ComponentsViewer
from ._sequence import ProjectSequence

__all__ = [
    "CylindraProject",
    "CylindraBatchProject",
    "ComponentsViewer",
    "ProjectSequence",
    "get_project_json",
]


def get_project_json(path: "str | Path"):
    """Return the path to the project.json file."""
    path = Path(path)
    if path.is_dir():
        path = path / "project.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Directory {path} seems not a cylindra project directory. A "
                "project directory should contain a 'project.json' file."
            )
    return path
