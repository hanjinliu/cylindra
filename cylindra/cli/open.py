from __future__ import annotations

import math
import warnings
from pathlib import Path

from cylindra import _io
from cylindra.cli._base import ParserBase
from cylindra.core import read_batch_project, read_project, start


class ParserOpen(ParserBase):
    """cylindra open [bold green]path[/bold green] [bold cyan]options[/bold cyan]

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
            "--invert", action="store_true", help="Invert the image contrast."
        )
        self.add_argument(
            "--filter",
            type=str,
            default=None,
            help="Image filter to apply when opening the image.",
        )
        self.add_argument(
            "--bin-size",
            nargs="+",
            type=int,
            default=None,
            help="Bin sizes to generate multiscale images.",
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
        invert: bool = False,
        filter: str | None = None,
        bin_size: list[int] | None = None,
        no_reference: bool = False,
        **kwargs,
    ):
        fp = Path(path)
        if fp.suffix in ["", ".tar", ".zip", ".json"]:
            try:
                read_project(fp)  # check if the project is valid
                _is_batch = False
            except Exception:
                read_batch_project(fp)  # check if it's a batch project
                _is_batch = True
            print(f"Opening project: {fp.as_posix()}")
            if not (scale is None and missing_wedge is None):
                warnings.warn(
                    "scale and tilt are ignored for project input.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            ui = start(viewer=self.viewer)
            if _is_batch:
                ui.batch.load_batch_project(fp)
            else:
                ui.load_project(fp, read_image=not no_reference)
        else:
            print(f"Opening image: {fp.as_posix()}")

            from cylindra.project import TomogramDefaults

            ref_image = None
            inv_ref = False
            if defaults := TomogramDefaults.from_dir(fp.parent):
                print(".cylindra-defaults.toml found, applying default settings.")
                scale = scale or defaults.scale
                missing_wedge = missing_wedge or defaults.missing_wedge.as_param()
                invert = defaults.invert
                filter = filter or defaults.filter
                bin_size = defaults.bin_size
                ref_image = defaults.resolve_reference_path(fp)
                inv_ref = defaults.invert_reference

            ui = start(viewer=self.viewer)
            img_meta = _io.read_header(path)
            scale = scale or img_meta.scale.x
            bin_size = bin_size or [int(math.ceil(0.96 / scale))]
            if ref_image:
                ui.open_image_with_reference(
                    fp, ref_image, scale=scale, bin_size=bin_size,
                    tilt_range=missing_wedge, invert=invert, filter=filter,
                    invert_reference=inv_ref,
                )  # fmt: skip
            else:
                ui.open_image(
                    fp, scale=scale, bin_size=bin_size, tilt_range=missing_wedge,
                    invert=invert, filter=filter,
                )  # fmt: skip
        if not self._IS_TESTING:  # pragma: no cover
            ui.parent_viewer.show(block=self.viewer is None)
