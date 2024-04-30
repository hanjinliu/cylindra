import warnings
from typing import Annotated

import numpy as np
import pandas as pd
import polars as pl
from acryo import Molecules
from magicclass import field, magicmenu, set_design
from magicclass.types import Path
from magicclass.widgets import Separator

from cylindra.const import FileFilter
from cylindra.widget_utils import add_molecules, capitalize
from cylindra.widgets._annotated import MoleculesLayersType, assert_list_of_layers
from cylindra.widgets.subwidgets._child_widget import ChildWidget

POS_COLUMNS = ["rlnCoordinateZ", "rlnCoordinateY", "rlnCoordinateX"]
ROT_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
POS_ORIGIN_COLUMNS = ["rlnOriginZAngst", "rlnOriginYAngst", "rlnOriginXAngst"]
RELION_TUBE_ID = "rlnHelicalTubeID"
MOLE_ID = "MoleculeGroupID"
PIXEL_SIZE = "rlnTomoTiltSeriesPixelSize"
OPTICS_GROUP = "rlnOpticsGroup"


@magicmenu
class RELION(ChildWidget):
    """File IO for RELION."""

    @set_design(text=capitalize)
    def load_molecules(self, path: Path.Read[FileFilter.STAR]):
        """
        Read monomer coordinates and angles from RELION .star file.

        Parameters
        ----------
        path : path-like
            The path to the star file.
        """
        path = Path(path)
        moles = self._read_star(path)
        for i, mole in enumerate(moles):
            add_molecules(self.parent_viewer, mole, f"{path.name}-{i}", source=None)

    @set_design(text=capitalize)
    def load_splines(self, path: Path.Read[FileFilter.STAR]):
        """
        Read a star file and register all the tubes as splines.

        The "rlnHelicalTubeID" column will be used to group the points into splines.

        Parameters
        ----------
        path : path-like
            The path to the star file.
        """
        mole = Molecules.concat(self._read_star(path))
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
        self,
        save_path: Path.Save[FileFilter.STAR],
        layers: MoleculesLayersType,
        save_features: bool = True,
    ):
        """
        Save the selected molecules to a RELION .star file.

        If multiple layers are selected, the `MoleculeGroupID` column will be added
        to the star file to distinguish the layers.

        Parameters
        ----------
        save_path : path-like
            The path to save the star file.
        layers : sequence of MoleculesLayer
            The layers to save.
        save_features : bool, default True
            Whether to save the features of the molecules.
        """
        save_path = Path(save_path)
        layers = assert_list_of_layers(layers, self.parent_viewer)
        mole = Molecules.concat([layer.molecules for layer in layers])
        euler_angle = mole.euler_angle(seq="ZYZ", degrees=True)

        out_dict = {
            POS_COLUMNS[2]: mole.pos[:, 2] / self.scale,
            POS_COLUMNS[1]: mole.pos[:, 1] / self.scale,
            POS_COLUMNS[0]: mole.pos[:, 0] / self.scale,
            ROT_COLUMNS[0]: euler_angle[:, 0],
            ROT_COLUMNS[1]: euler_angle[:, 1],
            ROT_COLUMNS[2]: euler_angle[:, 2],
            OPTICS_GROUP: np.ones(mole.count(), dtype=np.uint32),
        }
        if len(layers) > 1:
            out_dict[MOLE_ID] = np.concatenate(
                [
                    np.full(layer.molecules.count(), i, dtype=np.uint32)
                    for i, layer in enumerate(layers)
                ]
            )
        if save_features:
            for col in mole.features.columns:
                out_dict[col] = mole.features[col]
        df = pd.DataFrame(out_dict)
        self._write_star(df, save_path)
        return None

    @set_design(text=capitalize)
    def save_splines(
        self,
        save_path: Path.Save[FileFilter.STAR],
        interval: Annotated[float, {"min": 0.01, "max": 1000.0, "label": "Sampling interval (nm)"}] = 10.0,
    ):  # fmt: skip
        """
        Save the current splines to a RELION .star file.

        Parameters
        ----------
        save_path : path-like
            The path to save the star file.
        interval : float, default 10.0
            Sampling interval along the splines. For example, if interval=10.0 and the
            length of a spline is 100.0, 11 points will be sampled.
        """

        if interval <= 1e-4:
            raise ValueError("Interval must be larger than 1e-4.")
        save_path = Path(save_path)
        main = self._get_main()
        data_list: list[pl.DataFrame] = []
        for i, spl in enumerate(main.splines):
            num = int(spl.length() / interval)
            coords = spl.partition(num) / self.scale
            df = pl.DataFrame(
                {
                    POS_COLUMNS[2]: coords[:, 2],
                    POS_COLUMNS[1]: coords[:, 1],
                    POS_COLUMNS[0]: coords[:, 0],
                    ROT_COLUMNS[0]: 0.0,
                    ROT_COLUMNS[1]: 0.0,
                    ROT_COLUMNS[2]: 0.0,
                    RELION_TUBE_ID: i,
                    OPTICS_GROUP: np.ones(coords.shape[0], dtype=np.uint32),
                }
            )
            data_list.append(df)
        df = pl.concat(data_list, how="vertical").to_pandas()
        self._write_star(df, save_path)

    @property
    def scale(self) -> float:
        return self._get_main().tomogram.scale

    def _read_star(self, path) -> list[Molecules]:
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

        scale = self.scale / 10  # default scale to use
        if "optics" in star:
            opt = star["optics"]
            if not isinstance(opt, pd.DataFrame):
                raise NotImplementedError("Optics block must be a dataframe")
            particles = particles.merge(opt, on=OPTICS_GROUP)
            if PIXEL_SIZE in particles.columns:
                pixel_sizes = particles[PIXEL_SIZE] / 10
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
        if MOLE_ID in mole.features.columns:
            return [m.drop_features(MOLE_ID) for _, m in mole.group_by(MOLE_ID)]
        return [mole]

    def _write_star(self, df: pd.DataFrame, path: str):
        try:
            import starfile
        except ImportError:
            raise ImportError(
                "`starfile` is required to save RELION star files. Please\n"
                "$ pip install starfile"
            )

        head = pd.DataFrame({OPTICS_GROUP: [1], PIXEL_SIZE: [self.scale * 10]})

        return starfile.write({"optics": head, "particles": df}, path)
