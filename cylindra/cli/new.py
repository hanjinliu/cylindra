from __future__ import annotations

from glob import glob
from pathlib import Path

from cylindra.cli._base import ParserBase


class ParserNew(ParserBase):
    """
    cylindra new [bold green]output[/bold green] [bold cyan]options[/bold cyan]

    [u bold green]output[/u bold green]
        Path of the project file. It can be a directory or a zip/tar file.

    [u bold cyan]options[/u bold cyan]
        [bold]--image[/bold]
            Image file for this project.

        [bold]--multiscales, --binsize[/bold]
            Bin sizes of this project.

        [bold]--scale, -s[/bold]
            Scale (nm/pixel) of the image. If not given, it will be read from the image header.

        [bold]--missing_wedge, --mw[/bold]
            Min/max tilt angles in degree that defines the missing wedge.

        [bold]--molecules, -m[/bold]
            Molecules files that will be copied to the project.

        [bold]--like[/bold]
            Copy parameters as defaults from another project.
    """

    def __init__(self):
        super().__init__(prog="cylindra new", description="Create a new project.")
        self.add_argument("output", type=str)
        self.add_argument("--image", type=str, default=None)
        self.add_argument(
            "--multiscales", "--binsize", nargs="+", type=float, default=None
        )
        self.add_argument("--scale", "-s", type=float, default=None)
        self.add_argument("--missing_wedge", "--mw", nargs=2, type=float, default=None)
        self.add_argument("--molecules", "-m", nargs="*", default=[])
        self.add_argument("--like", type=str, default=None)

    def run_action(
        self,
        output: str,
        image: str | None = None,
        multiscales: list[float] | None = None,
        scale: float | None = None,
        missing_wedge: tuple[float, float] | None = None,
        molecules: list[str] = [],
        like: str | None = None,
        **kwargs,
    ):
        from cylindra.project import CylindraProject

        if like:
            prj = CylindraProject.from_file(like)
            if image is None:
                image = prj.image
            if multiscales is None:
                multiscales = prj.multiscales
            if scale is None:
                scale = prj.scale
            if missing_wedge is None:
                missing_wedge = prj.missing_wedge.as_param()
        if image is None:
            raise ValueError("Image file is required.")
        path = Path(output)
        prj = CylindraProject.new(image, scale, multiscales, missing_wedge, path)
        mole_input = {}
        if molecules:
            from acryo import Molecules

            for mole in molecules:
                if "*" in mole:
                    if "**" in mole:
                        raise ValueError("Recursive glob is not supported.")
                    for m in glob(mole):
                        mole_input[Path(m).name] = Molecules.from_file(m)
                else:
                    mole_input[Path(mole).name] = Molecules.from_file(mole)
        prj.save(path, molecules=mole_input)
        return print(f"Project created at: {path.absolute().as_posix()}")
