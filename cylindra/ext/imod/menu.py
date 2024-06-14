from typing import Annotated, Any

import impy as ip
import numpy as np
import pandas as pd
import polars as pl
from acryo import Molecules
from magicclass import field, magicmenu, set_design
from magicclass.types import Path
from magicclass.widgets import Separator

from cylindra.const import FileFilter
from cylindra.widget_utils import add_molecules, capitalize
from cylindra.widgets._annotated import (
    MoleculesLayersType,
    MoleculesLayerType,
    assert_layer,
    assert_list_of_layers,
)
from cylindra.widgets.subwidgets._child_widget import ChildWidget


@magicmenu
class IMOD(ChildWidget):
    """File IO for IMOD softwares."""

    @set_design(text=capitalize)
    def load_molecules(
        self,
        mod_path: Annotated[Path.Read[FileFilter.MOD], {"label": "Path to MOD file"}],
        ang_path: Annotated[Path.Read[FileFilter.CSV], {"label": "Path to csv file"}],
        shift_mol: Annotated[bool, {"label": "Apply shifts to monomers if offsets are available."}] = True,
    ):  # fmt: skip
        """
        Read molecule coordinates and angles from IMOD .mod files.

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
        from .cmd import read_mod

        mod_path = Path(mod_path)
        df = read_mod(mod_path)
        mod = df.select("z", "y", "x").to_numpy(writable=True)
        mod[:, 1:] -= 0.5  # shift to center of voxel
        shifts, angs = _read_shift_and_angle(ang_path)
        mol = Molecules.from_euler(pos=mod * self.scale, angles=angs, degrees=True)
        if shift_mol:
            mol.translate(shifts * self.scale, copy=False)

        return add_molecules(self.parent_viewer, mol, mod_path.name, source=None)

    @set_design(text=capitalize)
    def load_splines(
        self,
        mod_path: Annotated[Path.Read[FileFilter.MOD], {"label": "Path to MOD file"}],
    ):
        """Read a mod file and register all the contours as splines."""
        from .cmd import read_mod

        df = read_mod(mod_path)
        main = self._get_main()
        for _, sub in df.group_by("object_id", "contour_id", maintain_order=True):
            coords = sub.select("z", "y", "x").to_numpy(writable=True)
            coords[:, 1:] -= 0.5  # shift YX to center of voxel
            main.register_path(coords * self.scale, err_max=1e-8)

    sep0 = field(Separator)

    @set_design(text=capitalize)
    def save_molecules(self, save_dir: Path.Dir, layers: MoleculesLayersType):
        """
        Save monomer positions and angles in the PEET format.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        layers : sequence of MoleculesLayer
            Select the layers to save. All the molecules will be concatenated.
        """
        save_dir = Path(save_dir)
        layers = assert_list_of_layers(layers, self.parent_viewer)
        mol = Molecules.concat([l.molecules for l in layers])
        return _save_molecules(save_dir=save_dir, mol=mol, scale=self.scale)

    @set_design(text=capitalize)
    def save_splines(
        self,
        save_path: Path.Save[FileFilter.MOD],
        interval: Annotated[float, {"min": 0.01, "max": 1000.0, "label": "Sampling interval (nm)"}] = 10.0,
    ):  # fmt: skip
        """
        Save splines as a mod file.

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
        from .cmd import save_mod

        if interval <= 1e-4:
            raise ValueError("Interval must be larger than 1e-4.")
        main = self._get_main()
        data_list = []
        for i, spl in enumerate(main.splines):
            num = int(spl.length() / interval)
            coords = spl.partition(num) / self.scale
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

    @set_design(text=capitalize)
    def shift_molecules(
        self,
        ang_path: Annotated[Path.Read[FileFilter.CSV], {"label": "Path to csv file"}],
        layer: MoleculesLayerType,
        update: bool = False,
    ):
        """
        Shift monomer coordinates in PEET format.

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
        mol_shifted = mol.translate(shifts * self.scale)
        mol_shifted = Molecules.from_euler(
            pos=mol_shifted.pos, angles=angs, degrees=True
        )

        vector_data = np.stack([mol_shifted.pos, mol_shifted.z], axis=1)
        if update:
            layer.data = mol_shifted.pos
            vector_layer = None
            vector_layer_name = layer.name + " Z-axis"
            for l in self.parent_viewer.layers:
                if l.name == vector_layer_name:
                    vector_layer = l
                    break
            if vector_layer is not None:
                vector_layer.data = vector_data
            else:
                self.parent_viewer.add_vectors(
                    vector_data,
                    edge_width=0.3,
                    edge_color="crimson",
                    length=2.4,
                    name=vector_layer_name,
                )
            layer.molecules = mol_shifted
        else:
            add_molecules(self.parent_viewer, mol_shifted, name="Molecules from PEET")
        return None

    def _get_template_path(self, *_) -> Path:
        return self._get_main().sta._template_param()

    def _get_mask_params(self, *_):
        return self._get_main().sta._get_mask_params()

    @set_design(text=capitalize)
    def export_project(
        self,
        layer: MoleculesLayerType,
        save_dir: Path.Dir,
        template_path: Annotated[str, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        project_name: str = "project-0",
    ):
        """
        Export cylindra state as a PEET prm file.

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
        layer = assert_layer(layer, self.parent_viewer)
        if not save_dir.exists():
            save_dir.mkdir()
        main = self._get_main()
        loader = main.tomogram.get_subtomogram_loader(
            Molecules.empty(),
            binsize=1,
        )
        template_image, mask_image = loader.normalize_input(
            template=main.sta.params._norm_template_param(template_path),
            mask=main.sta.params._get_mask(params=mask_params),
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
            tomograms=str(main.tomogram.source),
            coordinates=coordinates_path,
            angles=angles_path,
            tilt_range=list(main.tomogram.tilt["range"]),
            template=template_path,
            project_name=project_name,
            shape=list(template_image.shape),
            mask_type=mask_path,
        )

        # save files
        prm_path.write_text(txt)
        mol = layer.molecules
        _save_molecules(save_dir=save_dir, mol=mol, scale=self.scale)
        ip.asarray(template_image, axes="zyx").set_scale(
            zyx=self.scale, unit="nm"
        ).imsave(save_dir / template_path)
        if mask_image is not None:
            ip.asarray(mask_image, axes="zyx").set_scale(
                zyx=self.scale, unit="nm"
            ).imsave(save_dir / mask_path)

        return None

    @property
    def scale(self) -> float:
        return self._get_main().tomogram.scale


def _read_angle(ang_path: str) -> np.ndarray:
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
    from .cmd import save_angles, save_mod

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
    save_mod(
        save_dir / mod_name,
        pl.DataFrame({"z": pos[:, 0], "y": pos[:, 1], "x": pos[:, 2]}),
    )
    save_angles(save_dir / csv_name, mol.euler_angle("ZXZ", degrees=True))
    return None


PEET_TEMPLATE = """
fnVolume = {{{tomograms!r}}}
fnModParticle = {{{coordinates!r}}}
initMOTL = {{{angles!r}}}
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
