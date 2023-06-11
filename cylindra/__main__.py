import argparse


class Namespace(argparse.Namespace):
    project: str
    view: str
    globals: str
    debug: bool


class Args(argparse.ArgumentParser):
    def __init__(self):
        from cylindra import __version__

        super().__init__(description="Command line interface of cylindra.")
        self.add_argument("--project", type=str, default="None")
        self.add_argument("--view", type=str, default="None")
        self.add_argument("--globals", type=str, default="None")
        self.add_argument("--debug", action="store_true")
        self.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"cylindra version {__version__}",
        )

    @classmethod
    def from_args(cls) -> Namespace:
        return cls().parse_args()


def main(viewer=None):  # "viewer" is used for testing only
    """The main function of the CLI."""
    args = Args.from_args()
    block = viewer is None

    project_file = None if args.project == "None" else args.project
    view_file = None if args.view == "None" else args.view
    globals_file = None if args.globals == "None" else args.globals

    if view_file:
        from cylindra import view_project

        return view_project(view_file, run=block)

    log_level = "DEBUG" if args.debug else "INFO"

    from cylindra import start

    ui = start(
        project_file=project_file,
        globals_file=globals_file,
        viewer=viewer,
        log_level=log_level,
    )

    return ui.parent_viewer.show(block=block)


if __name__ == "__main__":
    main()
