from __future__ import annotations

from cylindra.cli._base import ParserBase
from pathlib import Path


class ParserNew(ParserBase):
    def __init__(self):
        super().__init__(prog="cylindra new", description="Create a new project.")
        self.add_argument("output", type=str, help="The project file")
        self.add_argument("--image", type=str, help="image file.")
        self.add_argument(
            "--multiscales",
            nargs="+",
            type=float,
            help="Bin sizes.",
            default=None,
        )
        self.add_argument(
            "--scale",
            type=float,
            help="scale (nm/pixel) of the image.",
            default=None,
        )
        self.add_argument(
            "--missing_wedge",
            "--mw",
            nargs=2,
            type=float,
            help="min/max tilt angles in degree that defines the missing wedge.",
            default=None,
        )
        self.add_argument(
            "--molecules",
            "-m",
            nargs="*",
            help="molecules files.",
            default=[],
        )

    def run_action(
        self,
        output: str,
        image: str,
        multiscales: list[float] | None = None,
        scale: float | None = None,
        missing_wedge: tuple[float, float] | None = None,
        molecules: list[str] = [],
    ):
        from cylindra.project import CylindraProject
        from acryo import Molecules

        path = Path(output)
        prj = CylindraProject.new(image, scale, multiscales, missing_wedge, path)
        for mole in molecules:
            mole = Molecules.from_file(mole)
        prj.save(path)
        return print(f"Project created at: {path.as_posix()}")
