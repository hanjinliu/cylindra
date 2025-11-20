from __future__ import annotations

from glob import glob
from pathlib import Path

from cylindra.cli._base import ParserBase


class ParserPrep(ParserBase):
    """cylindra prep [bold green]pattern[/bold green] [bold green]output[/bold green]

    [u bold green]pattern[/u bold green]
        Path pattern to match tomograms or projects.

    [u bold green]output[/u bold green]
        Output directory to save the batch project and the individual projects.
    """

    def __init__(self):
        super().__init__(
            prog="cylindra collect",
            description="Collect tomograms or projects into a batch project.",
        )
        self.add_argument("pattern", type=str)
        self.add_argument("output", type=str)

    def run_action(
        self,
        pattern: str,
        output: str,
        **kwargs,
    ):
        from cylindra.project import (
            ChildProjectInfo,
            CylindraBatchProject,
            CylindraProject,
            TomogramDefaults,
        )

        output_dir = Path(output).expanduser().resolve()
        if not output_dir.exists():
            output_dir.mkdir()

        children: list[ChildProjectInfo] = []
        for each_path in glob(pattern):
            each_path = Path(each_path).expanduser().resolve()
            if each_path.suffix in ["", ".json", ".tar", ".zip"]:
                # incoming path is a project.
                prj_path = each_path
            else:
                # incoming path is a tomogram.
                prj_dir = output_dir / each_path.stem
                defaults = TomogramDefaults.from_dir(output_dir.parent)
                if defaults is None:
                    defaults = TomogramDefaults()
                prj = CylindraProject.new(
                    each_path,
                    scale=defaults.scale,
                    image_reference=defaults.image_reference,
                    multiscales=defaults.bin_size or [1],
                    missing_wedge=defaults.missing_wedge,
                    invert=defaults.invert,
                    invert_reference=defaults.invert_reference or False,
                )
                prj.save(prj_dir)
                prj_path = prj_dir / "project.json"
            children.append(ChildProjectInfo(path=prj_path))

        batch_prj = CylindraBatchProject.from_children(children)
        batch_prj.save(output_dir)
