import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from cylindra.project._json import project_json_encoder
from cylindra.project._utils import get_project_file

if TYPE_CHECKING:
    from pydantic import BaseModel
else:
    from pydantic_compat import BaseModel

PathLike = Path | str | bytes


class BaseProject(BaseModel):
    """The basic project class."""

    datetime: str
    version: str
    dependency_versions: dict[str, str]
    project_path: Path | None = None

    def _post_init(self):
        pass

    def resolve_path(self, file_dir: PathLike):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Return a dict."""
        d = super().model_dump(**kwargs)
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
            self.model_dump(),
            f,
            indent=4,
            separators=(",", ": "),
            default=project_json_encoder,
        )
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.project_path!r})"

    @classmethod
    def from_file(cls, path: "str | Path") -> Self:
        """Construct a project from a file."""
        path = Path(path).absolute()
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
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as zip:
                zip.extractall(tmpdir)
            self = cls.from_json(Path(tmpdir) / "project.json")
            self.project_path = path
        return self


_void = object()


def resolve_path(
    path: str | Path | None,
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
    """
    Missing wedge parameters.
    {"min": float, "max": float} for kind = "x" or "y".
    {"xrange": {"min": float, "max": float}, "yrange": {"min": float, "max": float}}
    for kind = "dual".
    """

    kind: str = "y"

    @classmethod
    def parse(self, obj):
        """Parse an object as a MissingWedge model."""
        if isinstance(obj, MissingWedge):
            # make a copy of the input model
            return MissingWedge(**obj.model_dump())
        elif isinstance(obj, dict):
            # must be one of the following:
            # {"kind": "none"}
            # {"kind": "x", "range": (-40, 40)}
            # {"kind": "y", "range": (-40, 40)}
            # {"kind": "dual", "xrange": (-40, 40), "yrange": (-50, 50)}
            kind = obj["kind"]
            if kind == "none":
                params = {}
            elif kind in ("x", "y"):
                _min, _max = obj["range"]
                params = {"min": _min, "max": _max}
            elif kind == "dual":
                _xrange = {"min": obj["xrange"][0], "max": obj["xrange"][1]}
                _yrange = {"min": obj["yrange"][0], "max": obj["yrange"][1]}
                params = {"xrange": _xrange, "yrange": _yrange}
            else:
                raise ValueError(f"Unknown missing wedge kind {kind!r}.")
            return MissingWedge(kind=kind, params=params)
        elif isinstance(obj, (tuple, list)) and len(obj) == 2:
            return MissingWedge(params={"min": obj[0], "max": obj[1]})
        elif obj is None:
            return MissingWedge(params={}, kind="none")
        raise TypeError(f"Cannot parse {obj!r} as a MissingWedge.")

    def as_param(self):
        """As the input parameter for tomogram creation."""
        if self.kind == "none":
            return None
        elif self.kind in ("x", "y"):
            return {
                "kind": self.kind,
                "range": (self.params["min"], self.params["max"]),
            }
        elif self.kind == "dual":
            return {
                "kind": self.kind,
                "xrange": (self.params["xrange"]["min"], self.params["xrange"]["max"]),
                "yrange": (self.params["yrange"]["min"], self.params["yrange"]["max"]),
            }
        else:
            raise ValueError(f"Unknown missing wedge kind {self.kind!r}.")
