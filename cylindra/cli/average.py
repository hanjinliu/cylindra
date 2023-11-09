from __future__ import annotations

from fnmatch import fnmatch
from cylindra.cli._base import ParserBase, coerce_output_filename, get_polars_expr
from cylindra.core import collect_projects


class ParserAverage(ParserBase):
    def __init__(self):
        super().__init__(prog="cylindra average", description="Average images.")
        self.add_argument(
            "project",
            type=str,
            help="Path to the project file. Can be a pattern containing `*`.",
        )
        self.add_argument(
            "--molecules",
            "-m",
            type=str,
            help="File name pattern of molecules.",
        )
        self.add_argument(
            "--size",
            "-s",
            type=float,
            help="Size of subtomograms in nm.",
        )
        self.add_argument(
            "--output",
            "-o",
            type=str,
            help="Output file name for the averaged image.",
            default=None,
        )
        self.add_argument(
            "--filter",
            "-f",
            type=str,
            help="Polars-style filter to apply to molecules.",
            default=None,
        )
        self.add_argument(
            "--order", type=int, help="Order of interpolation.", default=1
        )
        self.add_argument(
            "--split", action="store_true", help="Split the averaging into two."
        )
        self.add_argument(
            "--seed",
            type=int,
            help="Random seed used to determine the split.",
            default=None,
        )

    def run_action(
        self,
        project: str,
        molecules: str,
        size: float,
        output: str | None = None,
        filter: str | None = None,
        order: int = 3,
        split: bool = False,
        seed: int | None = None,
    ):
        import impy as ip

        loader = get_loader(project, molecules, size, filter, order)
        print("Averaging...")
        if split:
            avg = loader.average_split(seed=seed)
            avg = ip.asarray(avg, axes="tzyx").set_scale(xyz=loader.scale, unit="nm")
        else:
            avg = loader.average()
            avg = ip.asarray(avg, axes="zyx").set_scale(xyz=loader.scale, unit="nm")
        if output is None:
            output = coerce_output_filename("AVG", ext=".tif")
        avg.imsave(output)
        print(f"Average image saved at: {output}")


def get_loader(
    project: str,
    molecules: str,
    size: float,
    filter: str | None = None,
    order: int = 3,
):
    col = collect_projects(project)
    if "*" in molecules:
        name_filter = lambda n: fnmatch(n, molecules)
    else:
        name_filter = lambda n: n == molecules
    loader = col.sta_loader(name_filter)
    shape = (int(round(size / loader.scale)),) * 3
    if filter:
        filt_expr = get_polars_expr(filter)
        nmole_before = len(loader.molecules)
        loader = loader.filter(filt_expr)
        nmole_after = len(loader.molecules)
        print(f"Molecules filtered: {nmole_before} --> {nmole_after}")
    return loader.replace(output_shape=shape, order=order)
