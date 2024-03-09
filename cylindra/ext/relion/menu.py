import warnings
from typing import Annotated

import pandas as pd
import polars as pl
from acryo import Molecules
from magicclass import field, magicmenu, set_design
from magicclass.types import Path
from magicclass.widgets import Separator

from cylindra.const import FileFilter
from cylindra.types import MoleculesLayer
from cylindra.widget_utils import add_molecules, capitalize
from cylindra.widgets.subwidgets._child_widget import ChildWidget

POS_COLUMNS = ["rlnCoordinateZ", "rlnCoordinateY", "rlnCoordinateX"]
ROT_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
POS_ORIGIN_COLUMNS = ["rlnOriginZAngst", "rlnOriginYAngst", "rlnOriginXAngst"]
RELION_TUBE_ID = "rlnHelicalTubeID"


@magicmenu
class RELION(ChildWidget):
    """File IO for IMOD softwares."""

    @set_design(text=capitalize)
    def load_molecules(
        self,
        path: Annotated[Path.Read[FileFilter.STAR], {"label": "Path to star file"}],
    ):  # fmt: skip
        """
        Read monomer coordinates and angles from RELION .star file.
        """
        path = Path(path)
        mole = self._read_star(path)
        return add_molecules(self.parent_viewer, mole, path.name, source=None)

    @set_design(text=capitalize)
    def load_splines(
        self,
        path: Annotated[Path.Read[FileFilter.STAR], {"label": "Path to star file"}],
    ):
        """Read a star file and register all the tubes as splines."""
        mole = self._read_star(path)
        main = self._get_main()
        if RELION_TUBE_ID not in mole.features.columns:
            warnings.warn(
                f"{RELION_TUBE_ID!r} not found in star file. Use all points as a "
                "single spline.",
                UserWarning,
                stacklevel=2,
            )
            main.register_path(mole.pos, err_max=1e-8)
        else:
            for _, each in mole.group_by(RELION_TUBE_ID):
                main.register_path(each.pos, err_max=1e-8)
        return None

    sep0 = field(Separator)

    @set_design(text=capitalize)
    def save_molecules(
        self, save_path: Path.Save[FileFilter.STAR], layer: MoleculesLayer
    ):
        """Save the current molecules to a RELION .star file."""
        save_path = Path(save_path)
        mole = layer.molecules
        euler_angle = mole.euler_angle(seq="ZYZ", degrees=True)
        out_dict = {
            POS_COLUMNS[2]: mole.pos[:, 2] / self.scale * 10,
            POS_COLUMNS[1]: mole.pos[:, 1] / self.scale * 10,
            POS_COLUMNS[0]: mole.pos[:, 0] / self.scale * 10,
            ROT_COLUMNS[0]: -euler_angle[:, 0],
            ROT_COLUMNS[1]: -euler_angle[:, 1],
            ROT_COLUMNS[2]: -euler_angle[:, 2],
        }
        for col in mole.features.columns:
            out_dict[col] = mole.features[col]
        df = pd.DataFrame(out_dict)
        write_star(df, save_path)
        return None

    @set_design(text=capitalize)
    def save_splines(
        self,
        save_path: Path.Save[FileFilter.STAR],
        interval: Annotated[
            float, {"min": 0.01, "max": 1000.0, "label": "Sampling interval (nm)"}
        ] = 10.0,
    ):
        """Save the current splines to a RELION .star file."""

        if interval <= 1e-4:
            raise ValueError("Interval must be larger than 1e-4.")
        save_path = Path(save_path)
        main = self._get_main()
        data_list: list[pl.DataFrame] = []
        for i, spl in enumerate(main.splines):
            num = int(spl.length() / interval)
            coords = spl.partition(num) / self.scale * 10
            df = pl.DataFrame(
                {
                    POS_COLUMNS[2]: coords[:, 2],
                    POS_COLUMNS[1]: coords[:, 1],
                    POS_COLUMNS[0]: coords[:, 0],
                    ROT_COLUMNS[0]: 0.0,
                    ROT_COLUMNS[1]: 0.0,
                    ROT_COLUMNS[2]: 0.0,
                    RELION_TUBE_ID: i,
                }
            )
            data_list.append(df)
        data_all = pl.concat(data_list, how="vertical")
        df = pl.concat(data_all, how="vertical").to_pandas()
        write_star(df, save_path)

    @property
    def scale(self) -> float:
        return self._get_main().tomogram.scale

    def _read_star(self, path):
        try:
            import starfile
        except ImportError:
            raise ImportError(
                "`starfile` is required to read RELION star files. Please\n"
                "$ pip install starfile"
            )

        star = starfile.read(path)
        if not isinstance(star, dict):
            star = {"particles": star}  # assume particles block

        particles = star["particles"]
        if not isinstance(particles, pd.DataFrame):
            raise NotImplementedError("Particles block must be a dataframe")

        scale = self.scale  # default scale to use
        if "optics" in star:
            opt = star["optics"]
            if not isinstance(opt, pd.DataFrame):
                raise NotImplementedError("Optics block must be a dataframe")
            particles = particles.merge(opt, on="rlnOpticsGroup")
            if "rlnTomoTiltSeriesPixelSize" in particles.columns:
                pixel_sizes = particles["rlnTomoTiltSeriesPixelSize"] / 10
                for col in POS_COLUMNS:
                    particles[col] *= pixel_sizes  # update positions in place
                scale = 1  # because already updated
        particles[POS_COLUMNS] *= scale
        if all(c in particles.columns for c in POS_ORIGIN_COLUMNS):
            for target, source in zip(POS_COLUMNS, POS_ORIGIN_COLUMNS, strict=True):
                particles[target] += particles[source] / 10
            particles.drop(columns=POS_ORIGIN_COLUMNS, inplace=True)
        pos = particles[POS_COLUMNS].to_numpy()
        euler = particles[ROT_COLUMNS].to_numpy()
        features = particles.drop(columns=POS_COLUMNS + ROT_COLUMNS)
        mole = Molecules.from_euler(
            pos, euler, seq="ZYZ", degrees=True, order="xyz", features=features
        )
        return mole


def write_star(df: pd.DataFrame, path: str):
    try:
        import starfile
    except ImportError:
        raise ImportError(
            "`starfile` is required to save RELION star files. Please\n"
            "$ pip install starfile"
        )

    return starfile.write(df, path)
