from __future__ import annotations

from pathlib import Path
import math
import warnings
from cylindra.core import read_project, start
from cylindra.cli._base import _ParserBase


class ParserOpen(_ParserBase):
    """cylindra open <path>"""

    def __init__(self):
        super().__init__(
            prog="cylindra open",
            description="Open a project or an image file.",
        )
        self.add_argument(
            "path",
            type=str,
            help="path to the project/image file.",
        )
        self.add_argument(
            "--scale",
            "-s",
            type=float,
            help=(
                "Scale (nm/pixel) of the image, if an image file is given as the "
                "first argument. If not given, it will be read from the image "
                "file. This parameter will be ignored if a project is given."
            ),
            default=None,
        )
        self.add_argument(
            "--missing_wedge",
            "--mw",
            nargs=2,
            type=float,
            help="Min/max tilt angles. This parameter will be ignored if a project is given.",
            default=None,
        )

    def run_action(
        self,
        path: str,
        scale: float | None = None,
        missing_wedge: tuple[float, float] | None = None,
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
                ui = start(fp, viewer=self.viewer)
            case _:
                raise ValueError(f"invalid file type: {fp.suffix}")
        ui.parent_viewer.show(block=self.viewer is None)
