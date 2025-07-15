import warnings
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
import polars as pl
from acryo import Molecules
from magicclass.types import Path

try:
    import starfile
except ImportError:
    starfile = None

from cylindra.const import FileFilter, nm
from cylindra.plugin import register_function
from cylindra.widget_utils import add_molecules
from cylindra.widgets import CylindraMainWidget
from cylindra.widgets._annotated import MoleculesLayersType, assert_list_of_layers

if TYPE_CHECKING:
    from cylindra.components.tomogram import CylTomogram

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
REC_TOMO_PATH = "rlnTomoReconstructedTomogram"
REC_TOMO_HALF1_PATH = "rlnTomoReconstructedTomogramHalf1"


@register_function(name="Load molecules")
def load_molecules(ui: CylindraMainWidget, path: Path.Read[FileFilter.STAR]):
    """Read monomer coordinates and angles from RELION .star file.

    Parameters
    ----------
    path : path-like
        The path to the star file.
    """
    path = Path(path)
    moles = _read_star(path, ui.tomogram)
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
    mole = Molecules.concat(_read_star(path, ui.tomogram))
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
    centerz, centery, centerx = _shape_to_center_zyx(ui.tomogram.image.shape, scale)
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


@register_function(name="Open RELION job")
def open_relion_job(
    ui: CylindraMainWidget,
    path: Path.Dir,
    invert: bool = True,
):
    """Open a RELION tomogram reconstruction job folder.

    Parameters
    ----------
    path : path-like
        The path to the RELION job folder.
    """

    path = Path(path)
    rln_project_path = _relion_project_path(path)
    if not (path / "job.star").exists():
        raise ValueError(f"Directory {path} is not a RELION job folder.")
    jobtype = _get_job_type(path)
    if jobtype == "relion.reconstructtomograms":
        # Reconstruct Tomogram job
        tomogram_star_path = path / "tomograms.star"
        tomo_paths, scale_nm = _parse_tomo_star(tomogram_star_path)
        ui.batch.constructor._new_projects_from_table(
            path=[rln_project_path / p for p in tomo_paths],
            scale=scale_nm,
            invert=[invert] * len(tomo_paths),
            save_root=path / "cylindra",
        )
    elif (opt_star_path := path / "optimisation_set.star").exists():
        opt = starfile.read(opt_star_path)
        assert isinstance(opt, pd.DataFrame)
        tomo_paths = opt["rlnTomoTomogramsFile"]
        # XXX: WIP!
        raise NotImplementedError
        mole_dict = {}
        for particle_path, tomo_path in zip(
            opt["rlnTomoParticlesFile"], tomo_paths, strict=False
        ):
            tomo_paths, scale_nm = _parse_tomo_star(tomo_path)
            moles = _read_star(rln_project_path / particle_path, scale_nm)
            mole_dict[particle_path.stem] = moles

        # ui.batch.constructor.new_projects(
        #     [rln_project_path / p for p in tomo_paths],
        #     save_root=path / "cylindra",
        #     invert=invert,
        #     scale=scale,
        # )
    else:
        raise ValueError(f"Job {path.name} is not a tomogram reconstruction job.")


def _relion_project_path(path: Path) -> Path:
    return path.parent.parent


def _get_job_type(job_dir: Path) -> str:
    """Determine the type of RELION job based on the directory structure."""
    if (job_star_path := job_dir / "job.star").exists():
        return starfile.read(job_star_path, always_dict=True)["job"]["rlnJobTypeLabel"]
    raise ValueError(f"{job_dir} is not a RELION job folder.")


def _parse_tomo_star(path: Path) -> tuple[pd.Series, np.ndarray]:
    df = starfile.read(path)
    assert isinstance(df, pd.DataFrame)
    if REC_TOMO_PATH in df:
        tomo_paths = df[REC_TOMO_PATH]
    elif REC_TOMO_HALF1_PATH in df:
        tomo_paths = df[REC_TOMO_HALF1_PATH]
    else:
        raise ValueError(
            "No tomogram paths found in the tomograms.star file. Expected either "
            "'rlnTomoReconstructedTomogram' or 'rlnTomoReconstructedTomogramHalf1' "
            "column."
        )
    tomo_orig_scale = df["rlnTomoTiltSeriesPixelSize"]
    tomo_bin = df["rlnTomoTomogramBinning"]
    scale_nm = np.asarray(tomo_orig_scale / 10 * tomo_bin)
    return tomo_paths, scale_nm


def _read_star(path: str, tomo: "CylTomogram") -> list[Molecules]:
    try:
        import starfile
    except ImportError:
        raise ImportError(
            "`starfile` is required to read RELION star files. Please\n"
            "$ pip install starfile"
        )

    center_zyx = _shape_to_center_zyx(tomo.image.shape, tomo.scale)
    star = starfile.read(path)
    if not isinstance(star, dict):
        star = {"particles": star}  # assume particles block

    particles = star["particles"]  # angstrom
    for ix in range(3):
        particles[POS_COLUMNS[ix]] += center_zyx[ix] * 10
    if not isinstance(particles, pd.DataFrame):
        raise NotImplementedError("Particles block must be a dataframe")

    if "optics" in star:
        opt = star["optics"]
        if not isinstance(opt, pd.DataFrame):
            raise NotImplementedError("Optics block must be a dataframe")
        particles = particles.merge(opt, on=OPTICS_GROUP)

    if all(c in particles.columns for c in POS_ORIGIN_COLUMNS):
        for target, source in zip(POS_COLUMNS, POS_ORIGIN_COLUMNS, strict=True):
            particles[target] += particles[source]
        particles.drop(columns=POS_ORIGIN_COLUMNS, inplace=True)
    # particles are in Angstrom, convert to nm
    pos = particles[POS_COLUMNS].to_numpy() / 10
    euler = particles[ROT_COLUMNS].to_numpy()
    features = particles.drop(columns=POS_COLUMNS + ROT_COLUMNS)
    mole = Molecules.from_euler(
        pos, euler, seq="ZYZ", degrees=True, order="xyz", features=features
    )
    if MOLE_ID in mole.features.columns:
        return [m.drop_features(MOLE_ID) for _, m in mole.group_by(MOLE_ID)]
    return [mole]


def _write_star(df: pd.DataFrame, path: str, scale: float):
    return starfile.write(df, path)


def _shape_to_center_zyx(shape: tuple[int, int, int], scale: nm) -> np.ndarray:
    return (np.array(shape) / 2 - 1) * scale


def _strip_relion5_prefix(name: str):
    """Strip the RELION 5.0 rec prefix from the name."""
    if name.startswith("rec_"):
        name = name[4:]
    if "." in name:
        name = name.split(".")[0]
    return name
