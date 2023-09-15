from __future__ import annotations

import argparse
from typing import Any


class Namespace(argparse.Namespace):
    arg: Any | None
    project: str
    view: str
    init_config: bool
    debug: bool


class Args(argparse.ArgumentParser):
    def __init__(self):
        from cylindra import __version__

        super().__init__(description="Command line interface of cylindra.")
        self.add_argument("--project", type=str, default="None")
        self.add_argument("--view", type=str, default="None")
        self.add_argument("--debug", action="store_true")
        self.add_argument("--init-config", action="store_true")
        self.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"cylindra version {__version__}",
        )

    @classmethod
    def from_args(cls) -> Namespace:
        ns, argv = cls().parse_known_args()
        nargv = len(argv)
        if nargv == 0:
            ns.arg = None
        elif nargv == 1:
            ns.arg = argv[0]
        else:
            raise ValueError(f"too many arguments: {argv}")
        return ns


def main(viewer=None):  # "viewer" is used for testing only
    """The main function of the CLI."""
    args = Args.from_args()
    block = viewer is None

    project_file = None if args.project == "None" else args.project
    view_file = None if args.view == "None" else args.view

    if args.init_config:
        from cylindra._config import init_config

        return init_config(force=True)

    if view_file:
        from cylindra import view_project

        return view_project(view_file, run=block)

    log_level = "DEBUG" if args.debug else "INFO"

    from cylindra import start

    ui = start(
        project_file=project_file,
        viewer=viewer,
        log_level=log_level,
    )

    return ui.parent_viewer.show(block=block)


if __name__ == "__main__":
    main()
