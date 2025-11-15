from typing import Annotated, Any

import impy as ip
import numpy as np
import polars as pl
from acryo import Molecules
from magicclass.types import Optional, Path

from cylindra import instance
from cylindra.const import FileFilter, ImageFilter
from cylindra.plugin import register_function
from cylindra.utils import find_tilt_angles
from cylindra.widget_utils import add_molecules
from cylindra.widgets._annotated import (
    MoleculesLayersType,
    MoleculesLayerType,
    assert_layer,
    assert_list_of_layers,
)
from cylindra.widgets.main import CylindraMainWidget
from cylindra_builtins.imod.cmd import read_edf, read_mod, save_angles, save_mod


@register_function(name="Load molecules")
def load_molecules(
    ui: CylindraMainWidget,
    mod_path: Annotated[Path.Read[FileFilter.MOD], {"label": "Path to MOD file"}],
    ang_path: Annotated[Path.Read[FileFilter.CSV], {"label": "Path to csv file"}],
    shift_mol: Annotated[bool, {"label": "Apply shifts to monomers if offsets are available."}] = True,
):  # fmt: skip
    """Read molecule coordinates and angles from IMOD .mod files.

    Parameters
    ----------
    mod_path : Path
        Path to the mod file that contains molecule coordinates.
    ang_path : Path
        Path to the text file that contains molecule angles in Euler angles.
    shift_mol : bool, default True
        In PEET output csv there may be xOffset, yOffset, zOffset columns that can
        be directly applied to the molecule coordinates.
    """
    mod_path = Path(mod_path)
    df = read_mod(mod_path)
    mod = df.select("z", "y", "x").to_numpy(writable=True)
    mod[:, 1:] -= 0.5  # shift to center of voxel
    shifts, angs = _read_shift_and_angle(ang_path)
    scale = ui.tomogram.scale
    mol = Molecules.from_euler(pos=mod * scale, angles=angs, degrees=True)
    if shift_mol:
        mol.translate(shifts * scale, copy=False)

    return add_molecules(ui.parent_viewer, mol, mod_path.name, source=None)


@register_function(name="Load splines")
def load_splines(
    ui: CylindraMainWidget,
    mod_path: Annotated[Path.Read[FileFilter.MOD], {"label": "Path to MOD file"}],
):
    """Read a mod file and register all the contours as splines."""
    df = read_mod(mod_path)
    for _, sub in df.group_by("object_id", "contour_id", maintain_order=True):
        coords = sub.select("z", "y", "x").to_numpy(writable=True)
        coords[:, 1:] -= 0.5  # shift YX to center of voxel
        ui.register_path(coords * ui.tomogram.scale, err_max=1e-8)


@register_function(name="Save molecules", record=False)
def save_molecules(
    ui: CylindraMainWidget, save_dir: Path.Dir, layers: MoleculesLayersType
):
    """Save monomer positions and angles in the PEET format.

    Parameters
    ----------
    save_dir : Path
        Saving path.
    layers : sequence of MoleculesLayer
        Select the layers to save. All the molecules will be concatenated.
    """
    save_dir = Path(save_dir)
    layers = assert_list_of_layers(layers, ui.parent_viewer)
    mol = Molecules.concat([l.molecules for l in layers])
    return _save_molecules(save_dir=save_dir, mol=mol, scale=ui.tomogram.scale)


@register_function(name="Save splines", record=False)
def save_splines(
    ui: CylindraMainWidget,
    save_path: Path.Save[FileFilter.MOD],
    interval: Annotated[float, {"min": 0.01, "max": 1000.0, "label": "Sampling interval (nm)"}] = 10.0,
):  # fmt: skip
    """Save splines as a mod file.

    This function will sample coordinates along the splines and save the coordinates
    as a mod file. The mod file will be labeled with object_id=1 and contour_id=i+1,
    where i is the index of the spline.

    Parameters
    ----------
    save_path : Path
        Saving path.
    interval : float, default 10.0
        Sampling interval along the splines. For example, if interval=10.0 and the
        length of a spline is 100.0, 11 points will be sampled.
    """
    if interval <= 1e-4:
        raise ValueError("Interval must be larger than 1e-4.")
    data_list = []
    for i, spl in enumerate(ui.splines):
        num = int(spl.length() / interval)
        coords = spl.partition(num) / ui.tomogram.scale
        df = pl.DataFrame(
            {
                "object_id": 1,
                "contour_id": i + 1,
                "x": coords[:, 2] + 0.5,
                "y": coords[:, 1] + 0.5,
                "z": coords[:, 0],
            }
        )
        data_list.append(df)
    data_all = pl.concat(data_list, how="vertical")
    save_mod(save_path, data_all)


@register_function(name="Shift molecules")
def shift_molecules(
    ui: CylindraMainWidget,
    ang_path: Annotated[Path.Read[FileFilter.CSV], {"label": "Path to csv file"}],
    layer: MoleculesLayerType,
    update: bool = False,
):
    """Shift monomer coordinates in PEET format.

    Parameters
    ----------
    ang_path : Path
        Path of offset file.
    layer : MoleculesLayer
        Points layer of target monomers.
    update : bool, default False
        Check if update monomer coordinates in place.
    """
    mol = layer.molecules
    shifts, angs = _read_shift_and_angle(ang_path)
    mol_shifted = mol.translate(shifts * ui.tomogram.scale)
    mol_shifted = Molecules.from_euler(pos=mol_shifted.pos, angles=angs, degrees=True)

    vector_data = np.stack([mol_shifted.pos, mol_shifted.z], axis=1)
    if update:
        layer.data = mol_shifted.pos
        vector_layer = None
        vector_layer_name = layer.name + " Z-axis"
        for l in ui.parent_viewer.layers:
            if l.name == vector_layer_name:
                vector_layer = l
                break
        if vector_layer is not None:
            vector_layer.data = vector_data
        else:
            ui.parent_viewer.add_vectors(
                vector_data,
                edge_width=0.3,
                edge_color="crimson",
                length=2.4,
                name=vector_layer_name,
            )
        layer.molecules = mol_shifted
    else:
        add_molecules(ui.parent_viewer, mol_shifted, name="Molecules from PEET")


def _get_template_path(*_) -> Path:
    return instance().sta._template_param()


def _get_mask_params(*_):
    return instance().sta._get_mask_params()


@register_function(name="Open image from an IMOD project", record=False)
def open_image_from_imod_project(
    ui: CylindraMainWidget,
    edf_path: Annotated[Path.Read[FileFilter.EDF], {"label": "IMOD edf file"}],
    scale_override: Annotated[
        Optional[float],
        {"text": "Use header scale", "options": {"step": 0.0001, "value": 1.0}},
    ] = None,
    bin_size: list[int] = [4],
    filter: ImageFilter | None = ImageFilter.Lowpass,
    invert: bool = True,
    eager: Annotated[bool, {"label": "Load the entire image into memory"}] = False,
    cache_image: Annotated[bool, {"label": "Cache image on SSD"}] = False,
):
    """Open an image from an IMOD project.

    Parameters
    ----------
    edf_path : Path
        Path to the edf file.
    scale_override : float, default None
        Override the scale used for all the tomograms inside cylindra.
    bin_size : list of int, default [1]
        Bin sizes to load the tomograms.
    filter : ImageFilter, default ImageFilter.Lowpass
        Filter to apply when binning the image.
    invert : bool, default False
        If true, invert the intensity of the image.
    eager : bool, default False
        If true, the image will be loaded immediately. Otherwise, it will be loaded
        lazily.
    cache_image : bool, default False
        If true, the image will first be copied to the cache directory before
        loading.
    """
    res = _edf_to_tomo_and_tilt(edf_path)
    if res is None:
        raise ValueError(f"Could not find tomogram in the IMOD project {edf_path}.")
    tomo_path, tilt_model = res
    ui.open_image(
        tomo_path,
        scale=scale_override,
        invert=invert,
        tilt_range=tilt_model,
        bin_size=bin_size,
        filter=filter,
        eager=eager,
        cache_image=cache_image,
    )


@register_function(name="Import IMOD projects", record=False)
def import_imod_projects(
    ui: CylindraMainWidget,
    edf_path: Annotated[Path.Read[FileFilter.EDF], {"label": "IMOD edf file(s)"}],
    project_root: Optional[Path.Save] = None,
    scale_override: Annotated[Optional[float], {"text": "Use header scale", "options": {"step": 0.0001, "value": 1.0}}] = None,
    invert: bool = True,
    bin_size: list[int] = [1],
):  # fmt: skip
    """Import IMOD projects as batch analyzer entries.

    Parameters
    ----------
    edf_path : Path
        Path to the edf file(s). Path can contain wildcards.
    project_root : Path, default None
        Root directory to save the cylindra project folders. If None, a new directory
        will be created under the same level as the first IMOD project.
    scale_override : float, default None
        Override the scale used for all the tomograms inside cylindra.
    invert : bool, default False
        If true, invert the intensity of the image.
    bin_size : list of int, default [1]
        Bin sizes to load the tomograms.
    """
    from cylindra.widgets.batch._utils import unwrap_wildcard

    tomo_paths: list[Path] = []
    tilt_models: list[dict | None] = []
    for each in unwrap_wildcard(edf_path):
        if (res := _edf_to_tomo_and_tilt(each)) is not None:
            tomo_path, tilt_model = res
            tomo_paths.append(tomo_path)
            tilt_models.append(tilt_model)
    if len(tomo_paths) == 0:
        raise ValueError(f"No tomograms found with the given path input: {edf_path}")
    if scale_override is not None:
        scales = [scale_override] * len(tomo_paths)
    else:
        scales = None

    if project_root is None:
        project_root = tomo_paths[0].parent.parent / "cylindra"

    ui.batch._new_projects_from_table(
        tomo_paths,
        save_root=project_root,
        scale=scales,
        tilt_model=tilt_models,
        invert=[invert] * len(tomo_paths),
        bin_size=[bin_size] * len(tomo_paths),
    )


def _edf_to_tomo_and_tilt(edf_path: Path) -> tuple[Path, dict] | None:
    edf_path = Path(edf_path)
    edf = read_edf(edf_path)
    if dataset_name := edf.get("Setup.DatasetName", None):
        if (rec_path := edf_path.parent / f"{dataset_name}_rec.mrc").exists():
            tomo_path = rec_path
        elif (rec_path := edf_path.parent / f"{dataset_name}.rec").exists():  # old IMOD
            tomo_path = rec_path
        else:
            return None
        if (tilt_arr := find_tilt_angles(edf_path.parent)) is not None:
            tilt_min = round(float(tilt_arr.min()), 1)
            tilt_max = round(float(tilt_arr.max()), 1)
            tilt_model = {"kind": "y", "range": (tilt_min, tilt_max)}
        else:
            tilt_model = None
    else:
        return None
    return tomo_path, tilt_model


@register_function(name="Export project", record=False)
def export_project(
    ui: CylindraMainWidget,
    layer: MoleculesLayerType,
    save_dir: Path.Dir,
    template_path: Annotated[str, {"bind": _get_template_path}],
    mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
    project_name: str = "project-0",
):
    """Export cylindra state as a PEET prm file.

    Molecules and images will be exported to a directory that can be
    directly used by PEET.

    Parameters
    ----------
    layer : MoleculesLayer
        Molecules layer to export.
    save_dir : Path
        Directory to save the files needed for a PEET project.
    template_path : str
        Path to the template image.
    mask_params : Any, default None
        Mask parameters.
    project_name : str, default "project-0"
        Name of the PEET project.
    """
    save_dir = Path(save_dir)
    layer = assert_layer(layer, ui.parent_viewer)
    if not save_dir.exists():
        save_dir.mkdir()
    loader = ui.tomogram.get_subtomogram_loader(
        Molecules.empty(),
        binsize=1,
    )
    template_image, mask_image = loader.normalize_input(
        template=ui.sta.params._norm_template_param(template_path),
        mask=ui.sta.params._get_mask(params=mask_params),
    )

    if template_image is None:
        raise ValueError("Template image is not loaded.")

    # paths
    coordinates_path = "./coordinates.mod"
    angles_path = "./angles.csv"
    template_path = "./template-image.mrc"
    mask_path = "./mask-image.mrc"
    prm_path = save_dir / f"{project_name}.prm"

    txt = PEET_TEMPLATE.format(
        tomograms=repr(ui.tomogram.source),
        coordinates=repr(coordinates_path),
        angles=repr(angles_path),
        tilt_range=list(ui.tomogram.tilt["range"]),
        template=template_path,
        project_name=project_name,
        shape=list(template_image.shape),
        mask_type=mask_path,
    )

    # save files
    prm_path.write_text(txt)
    mol = layer.molecules
    _save_molecules(save_dir=save_dir, mol=mol, scale=ui.tomogram.scale)
    ip.asarray(template_image, axes="zyx").set_scale(
        zyx=ui.tomogram.scale, unit="nm"
    ).imsave(save_dir / template_path)
    if mask_image is not None:
        ip.asarray(mask_image, axes="zyx").set_scale(
            zyx=ui.tomogram.scale, unit="nm"
        ).imsave(save_dir / mask_path)


def _get_loader_paths(*_):
    from cylindra import instance

    ui = instance()
    return ui.batch._get_loader_paths(*_)


@register_function(name="Export project as batch", record=False)
def export_project_batch(
    ui: CylindraMainWidget,
    save_dir: Path.Dir,
    path_sets: Annotated[Any, {"bind": _get_loader_paths}],
    project_name: str = "project-0",
    size: Annotated[
        float, {"label": "Subtomogram size (nm)", "min": 1.0, "max": 1000.0}
    ] = 10.0,  # fmt: skip
):
    """Export cylindra batch analyzer state as a PEET prm file.

    A epe file will be generated, which can directly be used by `etomo <name>.epe`.

    Parameters
    ----------
    save_dir : Path
        Directory to save the files needed for a PEET project.
    path_sets : Any
        Path sets of the tomograms and coordinates.
    project_name : str, default "project-0"
        Name of the PEET project.
    size : float, default 10.0
        Size of the subtomograms in nanometers.
    """
    from cylindra.widgets.batch._sequence import PathInfo
    from cylindra.widgets.batch._utils import TempFeatures

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()

    _temp_feat = TempFeatures()

    _tomogram_list = list[str]()
    _coords_list = list[str]()
    _angle_list = list[str]()
    _tilt_list = list[str]()
    _count = 0
    scales = []
    for path_info in path_sets:
        path_info = PathInfo(*path_info)
        prj = path_info.project_instance(missing_ok=False)
        moles = list(path_info.iter_molecules(_temp_feat, prj.scale))
        if len(moles) > 0:
            _tomogram_list.append(repr(path_info.image.as_posix()))
            mod_name = f"coordinates-{_count:0>3}_{path_info.image.stem}.mod"
            csv_name = f"angles-{_count:0>3}_{path_info.image.stem}.csv"
            _save_molecules(
                save_dir=save_dir,
                mol=Molecules.concat(moles),
                scale=prj.scale,
                mod_name=mod_name,
                csv_name=csv_name,
            )
            _coords_list.append(f"'./{mod_name}'")
            _angle_list.append(f"'./{csv_name}'")
            if mw_dict := prj.missing_wedge.as_param():
                if "range" in mw_dict:
                    tilt_range = mw_dict["range"]
                    _tilt_list.append(f"[{tilt_range[0]}, {tilt_range[1]}]")
            scales.append(prj.scale)
            _count += 1

    # determine shape using the average scale
    if len(scales) == 0:
        raise ValueError("No tomograms found in the project.")
    scale = np.mean(scales)
    shape = [int(round(size / scale / 2)) * 2] * 3  # must be even

    # paths
    prm_path = save_dir / f"{project_name}.prm"
    epe_path = save_dir / f"{project_name}.epe"

    prm_txt = PEET_TEMPLATE.format(
        tomograms=", ".join(_tomogram_list),
        coordinates=", ".join(_coords_list),
        angles=", ".join(_angle_list),
        tilt_range=", ".join(_tilt_list),
        template="",
        project_name=project_name,
        shape=shape,
        mask_type="none",
    )

    # save files
    prm_path.write_text(prm_txt)
    epe_path.write_text(f"Peet.RootName={project_name}\n")


def _read_angle(ang_path: str) -> np.ndarray:
    import pandas as pd

    line1 = str(pd.read_csv(ang_path, nrows=1).values[0, 0])  # determine sep
    if "\t" in line1:
        sep = "\t"
    else:
        sep = ","

    csv = pd.read_csv(ang_path, sep=sep)

    if csv.shape[1] == 3:
        try:
            header = np.array(csv.columns).astype(np.float64)
            csv_data = np.concatenate([header.reshape(1, 3), csv.values], axis=0)
        except ValueError:
            csv_data = csv.values
    elif "CCC" in csv.columns:
        csv_data = -csv[["EulerZ(1)", "EulerX(2)", "EulerZ(3)"]].values
    else:
        raise ValueError(
            f"Could not interpret data format of {ang_path}:\n{csv.head(5)}"
        )
    return csv_data


def _read_shift_and_angle(path: str) -> tuple["np.ndarray | None", np.ndarray]:
    """Read offsets and angles from PEET project"""
    import pandas as pd

    csv: pd.DataFrame = pd.read_csv(path)
    if "CCC" in csv.columns:
        ang_data = -csv[["EulerZ(1)", "EulerX(2)", "EulerZ(3)"]].values
        shifts_data = csv[["zOffset", "yOffset", "xOffset"]].values
    else:
        ang_data = _read_angle(path)
        shifts_data = None
    return shifts_data, ang_data


def _save_molecules(
    save_dir: Path,
    mol: Molecules,
    scale: float,
    mod_name: "str | None" = None,
    csv_name: "str | None" = None,
):
    if mod_name is None:
        mod_name = "coordinates.mod"
    elif not mod_name.endswith(".mod"):
        mod_name += ".mod"
    if csv_name is None:
        csv_name = "angles.csv"
    elif not csv_name.endswith(".csv"):
        csv_name += ".csv"

    pos = mol.pos[:, ::-1] / scale
    pos[:, :2] += 0.5  # shift XY to center of voxel
    if not save_dir.exists():  # will happen if methods are called programmatically
        save_dir.mkdir()
    save_mod(
        save_dir / mod_name,
        pl.DataFrame({"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}),
    )
    save_angles(save_dir / csv_name, mol.euler_angle("ZXZ", degrees=True))


PEET_TEMPLATE = """
fnVolume = {{{tomograms}}}
fnModParticle = {{{coordinates}}}
initMOTL = {{{angles}}}
tiltRange = {{{tilt_range!r}}}
dPhi = {{0:0:0}}
dTheta = {{0:0:0}}
dPsi = {{0:0:0}}
searchRadius = {{[4]}}
lowCutoff = {{[0, 0.05]}}
hiCutoff = {{[0.9, 0.05]}}
refThreshold = {{100}}
duplicateShiftTolerance = [0]
duplicateAngularTolerance = [0]
reference = {template!r}
fnOutput = {project_name!r}
szVol = {shape!r}
alignedBaseName = ''
debugLevel = 3
lstThresholds = [40000:1000:45000]
refFlagAllTom = 1
lstFlagAllTom = 1
particlePerCPU = 3
yaxisType = 0
yaxisObjectNum = NaN
yaxisContourNum = NaN
flgWedgeWeight = 1
sampleSphere = 'none'
sampleInterval = NaN
maskType = {mask_type!r}
maskModelPts = []
insideMaskRadius = 0
outsideMaskRadius = NaN
nWeightGroup = 14
flgRemoveDuplicates = 0
flgAlignAverages = 0
flgFairReference = 0
flgAbsValue = 0
flgStrictSearchLimits = 0
flgNoReferenceRefinement = 0
flgRandomize = 0
cylinderHeight = NaN
maskBlurStdDev = NaN
flgVolNamesAreTemplates = 0
edgeShift = 1
"""
