import warnings
from typing import Annotated

import numpy as np
import pandas as pd
import polars as pl
from acryo import Molecules
from magicclass.types import Path

from cylindra.const import FileFilter
from cylindra.plugin import register_function
from cylindra.widget_utils import add_molecules
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets._annotated import MoleculesLayersType, assert_list_of_layers

TOMO_NAME = "rlnTomoName"
POS_COLUMNS = [
    "rlnCenteredCoordinateZAngst",
    "rlnCenteredCoordinateYAngst",
    "rlnCenteredCoordinateXAngst",
]
ROT_COLUMNS = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
POS_ORIGIN_COLUMNS = ["rlnOriginZAngst", "rlnOriginYAngst", "rlnOriginXAngst"]
RELION_TUBE_ID = "rlnHelicalTubeID"
MOLE_ID = "MoleculeGroupID"
PIXEL_SIZE = "rlnTomoTiltSeriesPixelSize"
OPTICS_GROUP = "rlnOpticsGroup"
IMAGE_PIXEL_SIZE = "rlnImagePixelSize"


@register_function(name="Load molecules")
def load_molecules(ui: CylindraMainWidget, path: Path.Read[FileFilter.STAR]):
    """Read monomer coordinates and angles from RELION .star file.

    Parameters
    ----------
    path : path-like
        The path to the star file.
    """
    path = Path(path)
    moles = _read_star(path, ui.tomogram.scale)
    for i, mole in enumerate(moles):
        add_molecules(ui.parent_viewer, mole, f"{path.name}-{i}", source=None)


@register_function(name="Load splines")
def load_splines(ui: CylindraMainWidget, path: Path.Read[FileFilter.STAR]):
    """Read a star file and register all the tubes as splines.

    The "rlnHelicalTubeID" column will be used to group the points into splines.

    Parameters
    ----------
    path : path-like
        The path to the star file.
    """
    mole = Molecules.concat(_read_star(path, ui.tomogram.scale))
    if RELION_TUBE_ID not in mole.features.columns:
        warnings.warn(
            f"{RELION_TUBE_ID!r} not found in star file. Use all points as a "
            "single spline.",
            UserWarning,
            stacklevel=2,
        )
        ui.register_path(mole.pos, err_max=1e-8)
    else:
        for _, each in mole.group_by(RELION_TUBE_ID):
            ui.register_path(each.pos, err_max=1e-8)


@register_function(name="Save molecules")
def save_molecules(
    ui: CylindraMainWidget,
    save_path: Path.Save[FileFilter.STAR],
    layers: MoleculesLayersType,
    save_features: bool = True,
    tomo_name_override: str = "",
    shift_by_origin: bool = True,
):
    """Save the selected molecules to a RELION .star file.

    If multiple layers are selected, the `MoleculeGroupID` column will be added
    to the star file to distinguish the layers. This method is RELION 5 compliant.

    Parameters
    ----------
    save_path : path-like
        The path to save the star file.
    layers : sequence of MoleculesLayer
        The layers to save.
    save_features : bool, default True
        Whether to save the features of the molecules.
    tomo_name_override : str, default ""
        If provided, this will override the tomogram name identifier (the rlnTomoName
        column) in the star file.
    shift_by_origin : bool, default True
        If True, the positions will be shifted by the origin of the tomogram. This
        option is required if you picked molecules in a trimmed tomogram.
    """
    save_path = Path(save_path)
    layers = assert_list_of_layers(layers, ui.parent_viewer)
    mole = Molecules.concat([layer.molecules for layer in layers])
    euler_angle = mole.euler_angle(seq="ZYZ", degrees=True)
    scale = ui.tomogram.scale
    orig = ui.tomogram.origin
    centerz, centery, centerx = (np.array(ui.tomogram.image.shape) / 2 - 1) * scale
    tomo_name = tomo_name_override or _strip_relion5_prefix(ui.tomogram.image.name)
    if not shift_by_origin:
        orig = type(orig)(0.0, 0.0, 0.0)
    out_dict = {
        TOMO_NAME: [tomo_name] * mole.count(),
        POS_COLUMNS[2]: (mole.pos[:, 2] - centerx + orig.x) * 10,  # Angstrom
        POS_COLUMNS[1]: (mole.pos[:, 1] - centery + orig.y) * 10,  # Angstrom
        POS_COLUMNS[0]: (mole.pos[:, 0] - centerz + orig.z) * 10,  # Angstrom
        ROT_COLUMNS[0]: euler_angle[:, 0],
        ROT_COLUMNS[1]: euler_angle[:, 1],
        ROT_COLUMNS[2]: euler_angle[:, 2],
        IMAGE_PIXEL_SIZE: scale * 10,  # convert to Angstrom
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
    _write_star(df, save_path, scale)


@register_function(name="Save splines")
def save_splines(
    ui: CylindraMainWidget,
    save_path: Path.Save[FileFilter.STAR],
    interval: Annotated[float, {"min": 0.01, "max": 1000.0, "label": "Sampling interval (nm)"}] = 10.0,
    tomo_name_override: str = "",
    shift_by_origin: bool = True,
):  # fmt: skip
    """Save the current splines to a RELION .star file.

    Parameters
    ----------
    save_path : path-like
        The path to save the star file.
    interval : float, default 10.0
        Sampling interval along the splines. For example, if interval=10.0 and the
        length of a spline is 100.0, 11 points will be sampled.
    tomo_name_override : str, default ""
        If provided, this will override the tomogram name identifier (the rlnTomoName
        column) in the star file.
    shift_by_origin : bool, default True
        If True, the positions will be shifted by the origin of the tomogram. This
        option is required if you picked molecules in a trimmed tomogram.
    """

    if interval <= 1e-4:
        raise ValueError("Interval must be larger than 1e-4.")
    save_path = Path(save_path)
    data_list: list[pl.DataFrame] = []
    orig = ui.tomogram.origin
    if not shift_by_origin:
        orig = type(orig)(0.0, 0.0, 0.0)
    tomo_name = tomo_name_override or ui.tomogram.image.name
    scale = ui.tomogram.scale
    centerz, centery, centerx = (np.array(ui.tomogram.image.shape) / 2 - 1) * scale
    for i, spl in enumerate(ui.splines):
        num = int(spl.length() / interval)
        coords = spl.partition(num)
        mole_count = coords.shape[0]
        df = pl.DataFrame(
            {
                TOMO_NAME: [tomo_name] * mole_count,
                POS_COLUMNS[2]: (coords[:, 2] - centerx + orig.x) * 10,  # Angstrom
                POS_COLUMNS[1]: (coords[:, 1] - centery + orig.y) * 10,  # Angstrom
                POS_COLUMNS[0]: (coords[:, 0] - centerz + orig.z) * 10,  # Angstrom
                ROT_COLUMNS[0]: 0.0,
                ROT_COLUMNS[1]: 0.0,
                ROT_COLUMNS[2]: 0.0,
                RELION_TUBE_ID: i,
                IMAGE_PIXEL_SIZE: ui.tomogram.scale * 10,  # convert to Angstrom
                OPTICS_GROUP: np.ones(mole_count, dtype=np.uint32),
            }
        )
        data_list.append(df)
    df = pl.concat(data_list, how="vertical").to_pandas()
    _write_star(df, save_path, ui.tomogram.scale)


def open_relion_job(ui: CylindraMainWidget, path: Path.Dir):
    """Open a RELION job folder.

    Parameters
    ----------
    path : path-like
        The path to the RELION job folder.
    """
    import starfile

    path = Path(path)
    if not (path / "job.star").exists():
        raise ValueError(f"Directory {path} is not a RELION job folder.")
    if (tomogram_star_path := path / "tomograms.star").exists():
        # Reconstruct Tomogram job
        tomogram_star = starfile.read(tomogram_star_path)
        print(tomogram_star)


def _read_star(path: str, scale: float) -> list[Molecules]:
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

    scale_a = scale / 10  # default scale (A) to use
    if "optics" in star:
        opt = star["optics"]
        if not isinstance(opt, pd.DataFrame):
            raise NotImplementedError("Optics block must be a dataframe")
        particles = particles.merge(opt, on=OPTICS_GROUP)
        if PIXEL_SIZE in particles.columns:
            pixel_sizes = particles[PIXEL_SIZE] / 10
            for col in POS_COLUMNS:
                particles[col] *= pixel_sizes  # update positions in place
            scale_a = 1  # because already updated
    particles[POS_COLUMNS] *= scale_a
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


def _write_star(df: pd.DataFrame, path: str, scale: float):
    try:
        import starfile
    except ImportError:
        raise ImportError(
            "`starfile` is required to save RELION star files. Please\n"
            "$ pip install starfile"
        )

    return starfile.write(df, path)


def _strip_relion5_prefix(name: str):
    """Strip the RELION 5.0 rec prefix from the name."""
    if name.startswith("rec_"):
        name = name[4:]
    if "." in name:
        name = name.split(".")[0]
    return name
