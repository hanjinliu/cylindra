from __future__ import annotations

from cylindra.cli._base import ParserBase
from cylindra.core import start


class ParserNone(ParserBase):
    """
    cylindra [bold green]options[/bold green] [bold cyan]commands[/bold cyan] [italic]arguments[/italic]

    [u bold green]options[/u bold green]
        -v, --version   Show version.
        -h, --help      Show this message and exit.

    [u bold cyan]commands[/u bold cyan]
        [bold]open[/bold]     Open a project or an image.
        [bold]preview[/bold]  View a project, image etc.
        [bold]find[/bold]     Find cylindra projects.
        [bold]run[/bold]      Run a script.
        [bold]config[/bold]   Edit/view the configuration.
        [bold]average[/bold]  Average images.
        [bold]new[/bold]      Create a new project.
        [bold]add[/bold]      Add splines by coordinates.
    """

    def __init__(self):
        from cylindra import __version__

        super().__init__(
            prog="cylindra",
            description="Command line interface of cylindra.",
        )
        self.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"cylindra version {__version__}",
            help="Show version.",
        )

    def run_action(self, **kwargs):
        ui = start(viewer=self.viewer)
        if not self._IS_TESTING:  # pragma: no cover
            ui.parent_viewer.show(block=self.viewer is None)
        return None
