import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Iterator

import impy as ip
import numpy as np
import pandas as pd
import polars as pl
from acryo import Molecules
from magicclass.types import Optional, Path

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
IMPORT_PARTICLE_FILE = "rlnTomoImportParticleFile"
POS = [
    "rlnCoordinateZ",
    "rlnCoordinateY",
    "rlnCoordinateX",
]
POS_CENTERED = [
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
REC_TOMO_PATH = "rlnTomoReconstructedTomogram"
REC_TOMO_HALF1_PATH = "rlnTomoReconstructedTomogramHalf1"
REC_TOMO_DENOISED_PATH = "rlnTomoReconstructedTomogramDenoised"


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
    for i, mole in enumerate(moles.values()):
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
    mole = Molecules.concat(_read_star(path, ui.tomogram).values())
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
    save_features: bool = False,
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
    tomo_name = tomo_name_override or _strip_relion5_prefix(ui.tomogram.image.name)
    df = _mole_to_star_df(
        [layer.molecules for layer in layers],
        ui.tomogram,
        tomo_name,
        save_features,
        shift_by_origin,
    )
    starfile.write(df, save_path)


def _get_loader_paths(*_):
    from cylindra import instance

    ui = instance()
    return ui.batch._get_loader_paths(*_)


@register_function(name="Save molecules for import", record=False)
def save_molecules_for_import(
    ui: CylindraMainWidget,
    coordinates_path: Path.Save[FileFilter.STAR],
    path_sets: Annotated[Any, {"bind": _get_loader_paths}],
    save_features: bool = False,
    shift_by_origin: bool = True,
    centered: bool = True,
):
    """Save the batch analyzer state as star files for "Import" job.

    Parameters
    ----------
    coordinates_path : path-like
        The path to save the star file containing the particles.
    path_sets : sequence of PathInfo
        The path sets to the tomograms and molecules.
    save_features : bool, default False
        Whether to save the features of the molecules to the star file.
    shift_by_origin : bool, default True
        If True, the positions will be shifted by the origin of the tomogram. This
        option is required if you picked molecules in a trimmed tomogram.
    centered : bool, default True
        If True, the positions will be centered around the tomogram center, and columns
        "rlnCenteredCoordinateX/Y/ZAngst" will be used. If False, columns
        "rlnCoordinateX/Y/Z" will be used.
    """
    coordinates_path = Path(coordinates_path)
    save_dir = coordinates_path.parent / f"{coordinates_path.stem}_particles"
    save_dir.mkdir(exist_ok=True)

    tomo_names = list[str]()
    particles_paths = list[str]()
    particles_dfs = list[pd.DataFrame]()
    for tomo_name, df in _iter_dataframe_from_path_sets(
        path_sets,
        save_features=save_features,
        shift_by_origin=shift_by_origin,
        centered=centered,
    ):
        particles_path = save_dir / f"{tomo_name}_particles.star"
        particles_dfs.append(df)
        particles_paths.append(particles_path)
        tomo_names.append(tomo_name)
        starfile.write(df, particles_path)

    df_opt = pd.DataFrame(
        {
            TOMO_NAME: tomo_names,
            IMPORT_PARTICLE_FILE: particles_paths,
        }
    )
    starfile.write(df_opt, coordinates_path)


@register_function(name="Save molecules for extract", record=False)
def save_molecules_for_extract(
    ui: CylindraMainWidget,
    coordinates_path: Path.Save[FileFilter.STAR],
    path_sets: Annotated[Any, {"bind": _get_loader_paths}],
    save_features: bool = False,
    shift_by_origin: bool = True,
    centered: bool = True,
):
    """Save the batch analyzer state as a star file for "Extract subtomo" job.

    The output star file should be provided as the "I/O > Input particle set" with
    "OR: use direct entries" set to "Yes".

    Parameters
    ----------
    coordinates_path : path-like
        The path to save the star file containing the particles.
    path_sets : sequence of PathInfo
        The path sets to the tomograms and molecules.
    save_features : bool, default False
        Whether to save the features of the molecules to the star file.
    shift_by_origin : bool, default True
        If True, the positions will be shifted by the origin of the tomogram. This
        option is required if you picked molecules in a trimmed tomogram.
    centered : bool, default True
        If True, the positions will be centered around the tomogram center, and columns
        "rlnCenteredCoordinateX/Y/ZAngst" will be used. If False, columns
        "rlnCoordinateX/Y/Z" will be used.
    """
    particles_dfs = list[pd.DataFrame]()
    for _, df in _iter_dataframe_from_path_sets(
        path_sets,
        save_features=save_features,
        shift_by_origin=shift_by_origin,
        centered=centered,
    ):
        particles_dfs.append(df)
    particles_df = pd.concat(particles_dfs)
    starfile.write({"particles": particles_df}, coordinates_path)


def _iter_dataframe_from_path_sets(
    path_sets,
    save_features: bool = False,
    shift_by_origin: bool = True,
    centered: bool = True,
) -> Iterator[tuple[str, pd.DataFrame]]:
    from cylindra.components.tomogram import CylTomogram
    from cylindra.widgets.batch._sequence import PathInfo
    from cylindra.widgets.batch._utils import TempFeatures

    _temp_feat = TempFeatures()

    for path_info in path_sets:
        path_info = PathInfo(*path_info)
        prj = path_info.project_instance(missing_ok=False)
        tomo_name = _strip_relion5_prefix(path_info.image.stem)
        img = path_info.lazy_imread()
        tomo = CylTomogram.from_image(
            img,
            scale=prj.scale,
            tilt=prj.missing_wedge.as_param(),
            compute=False,
        )
        moles = list(path_info.iter_molecules(_temp_feat, prj.scale))
        if len(moles) > 0:
            df = _mole_to_star_df(
                moles,
                tomo,
                tomo_name,
                save_features,
                shift_by_origin,
                centered=centered,
            )
            yield tomo_name, df


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
                POS_CENTERED[2]: (coords[:, 2] - centerx + orig.x) * 10,  # Angstrom
                POS_CENTERED[1]: (coords[:, 1] - centery + orig.y) * 10,  # Angstrom
                POS_CENTERED[0]: (coords[:, 0] - centerz + orig.z) * 10,  # Angstrom
                ROT_COLUMNS[0]: 0.0,
                ROT_COLUMNS[1]: 0.0,
                ROT_COLUMNS[2]: 0.0,
                RELION_TUBE_ID: i,
            }
        )
        data_list.append(df)
    df = pl.concat(data_list, how="vertical").to_pandas()
    starfile.write(df, save_path)


@register_function(name="Open RELION job")
def open_relion_job(
    ui: CylindraMainWidget,
    path: Path.Read[FileFilter.STAR_JOB],
    project_root: Optional[Path.Save] = None,
    invert: bool = True,
    bin_size: list[int] = [1],
):
    """Open a RELION tomogram reconstruction job folder.

    Parameters
    ----------
    path : path-like
        The path to the RELION job.star file.
    project_root : path-like, optional
        The root directory to save the projects for each tomogram. If not provided,
        it will be automatically created in the RELION job directory.
    invert : bool, default True
        Set to True if the tomograms are light backgroud.
    bin_size : list[int], default [1]
        The multiscale binning size for the tomograms.
    """
    path = Path(path)
    if path.name != "job.star" or not path.is_file() or not path.exists():
        raise ValueError(f"Path must be an existing RELION job.star file, got {path}")
    job_dir_path = Path(path).parent
    rln_project_path = _relion_project_path(job_dir_path)
    jobtype = _get_job_type(job_dir_path)
    if project_root is None:
        project_root = job_dir_path / "cylindra"
    if jobtype in ("relion.reconstructtomograms", "relion.denoisetomo"):
        # Reconstruct Tomogram job
        tomogram_star_path = job_dir_path / "tomograms.star"
        if not tomogram_star_path.exists():
            raise FileNotFoundError(
                f"tomogram.star file {tomogram_star_path} does not exist. Make sure "
                "the input job has an tomogram output."
            )
        col = (
            REC_TOMO_DENOISED_PATH if jobtype == "relion.denoisetomo" else REC_TOMO_PATH
        )
        _, tomo_paths, scales = _parse_tomo_star(tomogram_star_path, col)
        paths = [rln_project_path / p for p in tomo_paths]
        moles = None
    elif jobtype in ("relion.picktomo", "relion.pseudosubtomo"):
        if not (opt_star_path := job_dir_path / "optimisation_set.star").exists():
            raise ValueError(
                f"Optimisation set star file not found in {job_dir_path}. "
                "Please ensure the job is a RELION 5.0 pick-particles job."
            )
        paths, scales, moles = _parse_optimisation_star(opt_star_path, rln_project_path)
    elif jobtype in ("relion.initialmodel.tomo", "relion.refine3d.tomo"):
        opt_set_path_list = sorted(
            job_dir_path.glob("run_it*_optimisation_set.star"),
            key=lambda p: p.stem,
        )
        if len(opt_set_path_list) == 0:
            raise ValueError(
                f"No optimisation set star files found in {job_dir_path}. "
                "Please ensure at least one iteration has finished."
            )
        opt_star_path = opt_set_path_list[-1]
        paths, scales, moles = _parse_optimisation_star(opt_star_path, rln_project_path)
    else:
        raise ValueError(f"Job {job_dir_path.name} is not a supported RELION job.")
    # TODO: parse tilt angles
    ui.batch._new_projects_from_table(
        paths,
        save_root=project_root,
        scale=scales,
        invert=[invert] * len(paths),
        molecules=moles,
        bin_size=[bin_size] * len(paths),
    )


def _relion_project_path(path: Path) -> Path:
    return path.parent.parent


def _get_job_type(job_dir: Path) -> str:
    """Determine the type of RELION job based on the directory structure."""
    if (job_star_path := job_dir / "job.star").exists():
        return starfile.read(job_star_path, always_dict=True)["job"]["rlnJobTypeLabel"]
    raise ValueError(f"{job_dir} is not a RELION job folder.")


def _parse_tomo_star(
    path: Path,
    col: str = REC_TOMO_PATH,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    df = starfile.read(path)
    assert isinstance(df, pd.DataFrame)
    if col in df:
        tomo_paths = df[col]
    elif REC_TOMO_HALF1_PATH in df:
        tomo_paths = df[REC_TOMO_HALF1_PATH]
    else:
        raise ValueError(
            "No tomogram paths found in the tomograms.star file. Expected either "
            f"{col!r} or 'rlnTomoReconstructedTomogramHalf1' "
            "column."
        )
    tomo_orig_scale = df["rlnTomoTiltSeriesPixelSize"]
    tomo_bin = df["rlnTomoTomogramBinning"]
    tomo_names = df[TOMO_NAME]
    scale_nm = np.asarray(tomo_orig_scale / 10 * tomo_bin)
    return tomo_names, tomo_paths, scale_nm


def _parse_optimisation_star(opt_star_path: Path, rln_project_path: Path):
    paths = []
    scales = []
    molecules = []
    for item in _iter_from_optimisation_star(opt_star_path, rln_project_path):
        paths.append(rln_project_path / item.tomo_path)
        scales.append(item.scale)
        molecules.append(item.molecules)
    return paths, scales, molecules


@dataclass
class OptimizationSetItem:
    tomo_id: str
    tomo_path: Path
    scale: nm
    molecules: dict[str, Molecules]


def _iter_from_optimisation_star(
    path: Path,
    rln_project_path: Path,
) -> "Iterator[OptimizationSetItem]":
    opt_star_df = starfile.read(path)
    if isinstance(opt_star_df, pd.DataFrame):
        tomo_star_path: str = opt_star_df["rlnTomoTomogramsFile"][0]
        particles_path: str = opt_star_df["rlnTomoParticlesFile"][0]
    else:
        tomo_star_path: str = opt_star_df["rlnTomoTomogramsFile"]
        particles_path: str = opt_star_df["rlnTomoParticlesFile"]
    tomo_names, tomo_paths, scale_nm = _parse_tomo_star(
        rln_project_path / tomo_star_path
    )
    tomo_paths = [rln_project_path / p for p in tomo_paths]  # resolve relative paths
    particles_df = starfile.read(rln_project_path / particles_path)
    if isinstance(particles_df, dict):
        particles_df = particles_df["particles"]
    assert isinstance(particles_df, pd.DataFrame)
    name_to_center_map = {
        tomo_name: _shape_to_center_zyx(ip.lazy.imread(tomo_path).shape, sc_nm)
        for tomo_name, tomo_path, sc_nm in zip(
            tomo_names, tomo_paths, scale_nm, strict=False
        )
    }
    name_to_path_map = dict(zip(tomo_names, tomo_paths, strict=False))
    name_to_scale_map = dict(zip(tomo_names, scale_nm, strict=False))
    for tomo_id, particles in particles_df.groupby(TOMO_NAME):
        center_zyx = name_to_center_map.get(tomo_id)
        if center_zyx is None:
            warnings.warn(
                f"Tomogram {tomo_id} not found in the tomograms.star file. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        scale = name_to_scale_map[tomo_id]
        yield OptimizationSetItem(
            tomo_id,
            Path(name_to_path_map[tomo_id]),
            scale=scale,
            molecules=_particles_to_molecules(particles, center_zyx, scale=scale),
        )


def _read_star(path: str, tomo: "CylTomogram") -> dict[str, Molecules]:
    fpath = Path(path)
    center_zyx = _shape_to_center_zyx(tomo.image.shape, tomo.scale)
    star = starfile.read(fpath)
    if not isinstance(star, dict):
        star = {"particles": star}  # assume particles block

    particles = star["particles"]  # angstrom
    if not isinstance(particles, pd.DataFrame):
        raise NotImplementedError("Particles block must be a dataframe")

    return _particles_to_molecules(particles, center_zyx, fpath.stem, tomo.scale)


def _particles_to_molecules(
    particles: pd.DataFrame,
    center_zyx,
    default_key: str = "Mole-0",
    scale: nm = 1.0,
) -> dict[str, Molecules]:
    center_zyx = np.asarray(center_zyx, dtype=np.float32)
    if all(c in particles.columns for c in POS_CENTERED):
        pos = particles[POS_CENTERED].to_numpy() / 10 + center_zyx[np.newaxis, :]
        particles = particles.drop(columns=POS_CENTERED)
    elif all(c in particles.columns for c in POS):
        pos = particles[POS] * scale + center_zyx[np.newaxis, :]
        particles = particles.drop(columns=POS)
    else:
        raise ValueError("No position columns found in the DataFrame.")

    if all(c in particles.columns for c in POS_ORIGIN_COLUMNS):
        for target, source in enumerate(POS_ORIGIN_COLUMNS):
            pos[:, target] += particles[source] / 10
        particles.drop(columns=POS_ORIGIN_COLUMNS, inplace=True)

    # TODO: should optics properties be included?
    # if "optics" in star:
    #     opt = star["optics"]
    #     if not isinstance(opt, pd.DataFrame):
    #         raise NotImplementedError("Optics block must be a dataframe")
    #     particles = particles.merge(opt, on=OPTICS_GROUP)

    if all(c in particles.columns for c in ROT_COLUMNS):
        euler = particles[ROT_COLUMNS].to_numpy()
        features = particles.drop(columns=ROT_COLUMNS)
    else:
        euler = np.zeros((particles.shape[0], 3), dtype=np.float32)
        features = particles
    mole = Molecules.from_euler(
        pos, euler, seq="ZYZ", degrees=True, order="xyz", features=features
    )
    if MOLE_ID in mole.features.columns:
        return {
            mole_id: m.drop_features(MOLE_ID) for mole_id, m in mole.group_by(MOLE_ID)
        }
    return {default_key: mole}


def _shape_to_center_zyx(shape: tuple[int, int, int], scale: nm) -> np.ndarray:
    return (np.array(shape) / 2 - 1) * scale


def _strip_relion5_prefix(name: str):
    """Strip the RELION 5.0 rec prefix from the name."""
    if name.startswith("rec_"):
        name = name[4:]
    if "." in name:
        name = name.split(".")[0]
    return name


def _mole_to_star_df(
    moles: list[Molecules],
    tomo: "CylTomogram",
    tomo_name: str,
    save_features: bool = False,
    shift_by_origin: bool = True,
    centered: bool = True,
) -> pd.DataFrame:
    mole = Molecules.concat(moles)
    euler_angle = mole.euler_angle(seq="ZYZ", degrees=True)
    scale = tomo.scale
    orig = tomo.origin
    if not shift_by_origin:
        orig = type(orig)(0.0, 0.0, 0.0)
    out_dict = {TOMO_NAME: [tomo_name] * mole.count()}
    _rot_dict = {
        ROT_COLUMNS[0]: euler_angle[:, 0],
        ROT_COLUMNS[1]: euler_angle[:, 1],
        ROT_COLUMNS[2]: euler_angle[:, 2],
    }
    if centered:
        centerz, centery, centerx = _shape_to_center_zyx(tomo.image.shape, scale)
        # Angstrom
        _pos_dict = {
            POS_CENTERED[2]: (mole.pos[:, 2] - centerx + orig.x) * 10,
            POS_CENTERED[1]: (mole.pos[:, 1] - centery + orig.y) * 10,
            POS_CENTERED[0]: (mole.pos[:, 0] - centerz + orig.z) * 10,
        }
    else:
        # pixels
        _pos_dict = {
            POS[2]: (mole.pos[:, 2] + orig.x) / scale,
            POS[1]: (mole.pos[:, 1] + orig.y) / scale,
            POS[0]: (mole.pos[:, 0] + orig.z) / scale,
        }
    out_dict.update(_pos_dict)
    out_dict.update(_rot_dict)
    if len(moles) > 1:
        out_dict[MOLE_ID] = np.concatenate(
            [
                np.full(each_mole.count(), i, dtype=np.uint32)
                for i, each_mole in enumerate(moles)
            ]
        )
    if save_features:
        for col in mole.features.columns:
            out_dict[col] = mole.features[col]
    return pd.DataFrame(out_dict)
