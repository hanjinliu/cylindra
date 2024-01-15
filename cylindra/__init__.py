from cylindra.core import (
    collect_projects,
    instance,
    read_molecules,
    read_project,
    read_spline,
    start,
    view_project,
)

__NAMESPACE = {}

__version__: str
__author__: str
__email__: str


def __getattr__(name: str):
    if name not in ["__version__", "__author__", "__email__"]:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in __NAMESPACE:
        from importlib.metadata import PackageNotFoundError, metadata, version

        try:
            _version = version("cylindra")
        except PackageNotFoundError:
            _version = "uninstalled"
        _author = metadata("cylindra")["Author"]
        _email = metadata("cylindra")["Author-email"]
        __NAMESPACE["__version__"] = _version
        __NAMESPACE["__author__"] = _author
        __NAMESPACE["__email__"] = _email
    return __NAMESPACE[name]


__all__ = [
    "start",
    "instance",
    "view_project",
    "read_project",
    "read_molecules",
    "read_spline",
    "collect_projects",
]
