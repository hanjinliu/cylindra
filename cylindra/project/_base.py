import json
from typing import Any, Union
from typing_extensions import Self
import io
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from cylindra.project._utils import get_project_file


def json_encoder(obj):
    """An enhanced encoder."""
    import numpy as np
    import pandas as pd
    import polars as pl

    if isinstance(obj, Enum):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pl.DataFrame):
        return obj.to_dict(as_series=False)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        # return as a POSIX path
        if obj.is_absolute():
            return obj.as_posix()
        else:
            return "./" + obj.as_posix()
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")


PathLike = Union[Path, str, bytes]


class BaseProject(BaseModel):
    """The basic project class."""

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    project_path: Union[Path, None] = None

    def _post_init(self):
        pass

    def resolve_path(self, file_dir: PathLike):
        pass

    def dict(self, **kwargs) -> dict[str, Any]:
        """Return a dict."""
        d = super().dict(**kwargs)
        d.pop("project_path")
        return d

    def to_json(self, path: "str | Path | io.IOBase") -> None:
        """Save project as a json file."""
        if isinstance(path, io.IOBase):
            return self._dump(path)
        with open(path, mode="w") as f:
            self._dump(f)
        return None

    def _dump(self, f: io.IOBase) -> None:
        """Dump the project to a file."""
        json.dump(
            self.dict(), f, indent=4, separators=(",", ": "), default=json_encoder
        )
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.project_path!r})"

    @classmethod
    def from_file(cls, path: "str | Path") -> Self:
        """Construct a project from a file."""
        path = Path(path)
        if path.is_dir():
            return cls.from_json(get_project_file(path))
        elif path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix == ".tar":
            return cls.from_tar(path)
        elif path.suffix == ".zip":
            return cls.from_zip(path)
        raise ValueError(f"Cannot construct a project from {path!r}.")

    @classmethod
    def from_json(cls, path: "str | Path") -> Self:
        """Construct a project from a json file."""
        path = get_project_file(path)
        with open(str(path).strip("'").strip('"')) as f:
            js: dict = json.load(f)
        self = cls(**js, project_path=path.parent)
        self._post_init()
        self.resolve_path(path.parent)
        return self

    @classmethod
    def from_tar(cls, path: "str | Path") -> Self:
        """Construct a project from a tar file."""
        import tarfile

        with tarfile.open(path) as tar:
            f = tar.extractfile("project.json")
            js = json.load(f)
        self = cls(**js, project_path=path)
        self._post_init()
        self.resolve_path(path.parent)
        return self

    @classmethod
    def from_zip(cls, path: "str | Path") -> Self:
        """Construct a project from a zip file."""
        import zipfile, tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as zip:
                zip.extractall(tmpdir)
            self = cls.from_json(tmpdir)
        return self


_void = object()


def resolve_path(
    path: Union[str, Path, None],
    root: Path,
    *,
    default: "Path | None" = _void,
) -> "Path | None":
    """Resolve a relative path to an absolute path."""
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    path_joined = root / path
    if path_joined.exists():
        return path_joined
    if default is _void:
        raise ValueError(
            f"Path {path} was resolved to be {path_joined} but does not exist."
        )
    return default


class MissingWedge(BaseModel):
    """The missing wedge model."""

    params: dict[str, Any]
    kind: str = "y"

    @classmethod
    def parse(self, obj):
        if isinstance(obj, MissingWedge):
            return MissingWedge(**obj.dict())
        elif isinstance(obj, dict):
            return MissingWedge(**obj)
        elif isinstance(obj, (tuple, list)) and len(obj) == 2:
            return MissingWedge(params={"min": obj[0], "max": obj[1]})
        elif obj is None:
            return MissingWedge(params={}, kind="none")
        raise TypeError(f"Cannot parse {obj!r} as a MissingWedge.")

    def as_param(self):
        """As the input parameter for tomogram creation."""
        if self.kind == "y":
            return (self.params["min"], self.params["max"])
        elif self.kind == "none":
            return None
        raise NotImplementedError("Only y-axis rotation is supported now.")
