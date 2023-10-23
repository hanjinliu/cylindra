from typing import Annotated, Any
import numpy as np
import pandas as pd

from magicclass import magicmenu
from magicclass.types import Path
from acryo import Molecules
import impy as ip

from cylindra.types import get_monomer_layers, MoleculesLayer
from cylindra.const import FileFilter
from cylindra.widgets.widget_utils import add_molecules
from cylindra.widgets.subwidgets._child_widget import ChildWidget


@magicmenu
class PEET(ChildWidget):
    """File IO for PEET software."""

    def read_monomers(
        self,
        mod_path: Annotated[Path.Read[FileFilter.MOD], {"label": "Path to MOD file"}],
        ang_path: Annotated[Path.Read[FileFilter.CSV], {"label": "Path to csv file"}],
        shift_mol: Annotated[bool, {"label": "Apply shifts to monomers if offsets are available."}] = True,
    ):  # fmt: skip
        """
        Read monomer coordinates and angles from IMOD .mod files.

        Parameters
        ----------
        mod_path : Path
            Path to the mod file that contains monomer coordinates.
        ang_path : Path
            Path to the text file that contains monomer angles in Euler angles.
        shift_mol : bool, default is True
            In PEET output csv there may be xOffset, yOffset, zOffset columns that can be directly applied to
            the molecule coordinates.
        """
        from .cmd import read_mod

        mod = read_mod(mod_path).values
        mod[:, 1:] -= 0.5  # shift to center of voxel
        shifts, angs = _read_shift_and_angle(ang_path)
        mol = Molecules.from_euler(pos=mod * self.scale, angles=angs, degrees=True)
        if shift_mol:
            mol.translate(shifts * self.scale, copy=False)

        return add_molecules(
            self.parent_viewer, mol, "Molecules from PEET", source=None
        )

    def save_monomers(self, save_dir: Path.Dir, layer: MoleculesLayer):
        """
        Save monomer positions and angles in the PEET format.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        layer : Points
            Select the Vectors layer to save.
        """
        save_dir = Path(save_dir)
        mol = layer.molecules
        return _save_molecules(save_dir=save_dir, mol=mol, scale=self.scale)

    def save_all_monomers(self, save_dir: Path.Dir):
        """
        Save monomer angles in PEET format.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        """
        save_dir = Path(save_dir)
        layers = get_monomer_layers(self)
        if len(layers) == 0:
            raise ValueError("No monomer found.")
        mol = Molecules.concat([l.molecules for l in layers])
        return _save_molecules(save_dir=save_dir, mol=mol, scale=self.scale)

    def shift_monomers(
        self,
        ang_path: Annotated[Path.Read[FileFilter.CSV], {"label": "Path to csv file"}],
        layer: MoleculesLayer,
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
        update : bool, default is False
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

    def _get_molecules_layers(self, *_) -> list[MoleculesLayer]:
        try:
            main = self._get_main()
            return get_monomer_layers(main)
        except Exception:
            return []

    def _get_template_path(self, *_) -> Path:
        return self._get_main().sta._get_template_path()

    def _get_mask_params(self, *_):
        return self._get_main().sta._get_mask_params()

    def export_project(
        self,
        layer: MoleculesLayer,
        save_dir: Path.Dir,
        template_path: Annotated[str, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        project_name: str = "MyProject",
    ):
        """
        Export cylindra state as a PEET prm file.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        """
        save_dir = Path(save_dir)
        main = self._get_main()
        loader = main.tomogram.get_subtomogram_loader(
            Molecules.empty(),
            binsize=1,
        )
        template_image, mask_image = loader.normalize_input(
            template=main.sta.params._get_template(path=template_path),
            mask=main.sta.params._get_mask(params=mask_params),
        )

        if template_image is None:
            raise ValueError("Template image is not loaded.")

        # paths
        coordinates_path = save_dir / "coordinates.mod"
        angles_path = save_dir / "angles.csv"
        template_path = save_dir / "template-image.mrc"
        mask_path = save_dir / "mask-image.mrc"
        prm_path = save_dir / f"{project_name}.prm"

        txt = PEET_TEMPLATE.format(
            tomograms=str(main.tomogram.source),
            coordinates=str(coordinates_path),
            angles=str(angles_path),
            tilt_range=list(main.tomogram.tilt_range),
            template=str(template_path),
            project_name=project_name,
            shape=list(template_image.shape),
            mask_type=str(mask_path),
        )

        # save files
        prm_path.write_text(txt)
        mol = layer.molecules
        _save_molecules(save_dir=save_dir, mol=mol, scale=self.scale)
        ip.asarray(template_image).imsave(template_path)
        if mask_image is not None:
            ip.asarray(mask_image).imsave(mask_path)

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
    from .cmd import save_mod, save_angles

    if mod_name is None:
        mod_name = "coordinates.mod"
    elif not mod_name.endswith(".mod"):
        mod_name += ".mod"
    if csv_name is None:
        csv_name = "angles.csv"
    elif not csv_name.endswith(".csv"):
        csv_name += ".csv"

    pos = mol.pos[:, ::-1] / scale
    pos[:, 1:] += 0.5
    save_mod(save_dir / mod_name, pos)
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
