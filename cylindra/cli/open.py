from __future__ import annotations

import math
import warnings
from pathlib import Path

from cylindra.cli._base import ParserBase
from cylindra.core import read_project, start


class ParserOpen(ParserBase):
    """
    cylindra open [bold green]path[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]path[/u bold green]
        Path to the project/image file.

    [u bold cyan]options[/u bold cyan]
        [bold]--scale, -s[/bold]
            Scale (nm/pixel) of the image, if an image file is given as the first argument. If not given, it will be read from the image file. This parameter will be ignored if a project is given.

        [bold]--missing_wedge, --mw[/bold]
            Min/max tilt angles. This parameter will be ignored if a project is given.
    """

    def __init__(self):
        super().__init__(
            prog="cylindra open",
            description="Open a project or an image file.",
        )
        self.add_argument("path", type=str)
        self.add_argument("--scale", "-s", type=float, default=None)
        self.add_argument(
            "--missing_wedge",
            "--mw",
            nargs=2,
            type=float,
            default=None,
        )
        self.add_argument(
            "--no-reference",
            "--nr",
            action="store_true",
            help="Do not calculate the reference image.",
        )

    def run_action(
        self,
        path: str,
        scale: float | None = None,
        missing_wedge: tuple[float, float] | None = None,
        no_reference: bool = False,
        **kwargs,
    ):
        fp = Path(path)
        match fp.suffix:
            case ".tif" | ".tiff" | ".mrc" | ".map":
                print(f"Opening image: {fp.as_posix()}")
                import impy as ip

                ui = start(viewer=self.viewer)
                img = ip.lazy.imread(path)
                scale = img.scale.x
                bin_size = int(math.ceil(0.96 / scale))
                ui.open_image(
                    fp, scale=scale, bin_size=[bin_size], tilt_range=missing_wedge
                )
            case "" | ".tar" | ".zip" | ".json":
                read_project(fp)  # check if the project is valid
                print(f"Opening project: {fp.as_posix()}")
                if not (scale is None and missing_wedge is None):
                    warnings.warn(
                        "scale and tilt are ignored for project input.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                ui = start(viewer=self.viewer)
                ui.load_project(fp, read_image=not no_reference)
            case _:
                raise ValueError(f"invalid file type: {fp.suffix}")
        if not self._IS_TESTING:  # pragma: no cover
            ui.parent_viewer.show(block=self.viewer is None)
