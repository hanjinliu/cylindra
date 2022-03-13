from typing import Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from magicclass import magicmenu, set_options, MagicTemplate

from ...components import Molecules, MtSpline
from ...utils import roundint
from ...const import EulerAxes, H, Mole
from ...types import MOLECULES, MonomerLayer, get_monomer_layers
from ...widgets.widget_utils import add_molecules

@magicmenu
class PEET(MagicTemplate):
    """PEET extension."""
    @set_options(mod_path={"label": "Path to MOD file", "mode": "r", "filter": "Model files (*.mod);;All files (*.txt;*.csv)"},
                 ang_path={"label": "Path to csv file", "mode": "r", "filter": "*.csv;*.txt"},
                 shift_mol={"label": "Apply shifts to monomers if offsets are available."})
    def Read_monomers(self, mod_path: Path, ang_path: Path, shift_mol: bool = True):
        """
        Read monomer coordinates and angles from PEET-format files.

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
        shifts, angs = _read_shift_and_angle(ang_path)
        mol = Molecules.from_euler(pos=mod*self.scale, angles=angs, degrees=True)
        if shift_mol:
            mol.translate(shifts*self.scale, copy=False)
        
        add_molecules(self.parent_viewer, mol, "Molecules from PEET", source=None)
    
    @set_options(save_dir={"label": "Save at", "mode": "d"})
    def Save_monomers(self, 
                      save_dir: Path,
                      layer: MonomerLayer,
                      save_protofilaments_separately: bool = False):
        """
        Save monomer angles in PEET format.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        layer : Points
            Select the Vectors layer to save.
        save_protofilaments_separately : bool, default is False
            Check if you want to save monomers on each protofilament in separate files.
        """        
        save_dir = Path(save_dir)
        mol: Molecules = layer.metadata[MOLECULES]
        from .cmd  import save_mod, save_angles
        if save_protofilaments_separately:
            npf = roundint(max(layer.features[Mole.pf]) + 1)
            for pf in range(npf):
                sl = slice(pf, None, npf)
                save_mod(save_dir/f"coordinates-PF{pf:0>2}.mod", mol.pos[sl, ::-1]/self.scale)
                save_angles(save_dir/f"angles-PF{pf:0>2}.csv", mol.euler_angle(EulerAxes.ZXZ, degrees=True)[sl])
        else:
            save_mod(save_dir/"coordinates.mod", mol.pos[:, ::-1]/self.scale)
            save_angles(save_dir/"angles.csv", mol.euler_angle(EulerAxes.ZXZ, degrees=True))
        return None
    
    @set_options(save_dir={"label": "Save at", "mode": "d"})
    def Save_all_monomers(self, 
                          save_dir: Path):
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
        mol = Molecules.concat([l.metadata[MOLECULES] for l in layers])
        from .cmd  import save_mod, save_angles
        save_mod(save_dir/"coordinates.mod", mol.pos[:, ::-1]/self.scale)
        save_angles(save_dir/"angles.csv", mol.euler_angle(EulerAxes.ZXZ, degrees=True))
        return None
    
    @set_options(ang_path={"label": "Path to csv file", "mode": "r", "filter": "*.csv;*.txt"})
    def Shift_monomers(self, ang_path: Path, layer: MonomerLayer, update: bool = False):
        """
        Shift monomer coordinates in PEET format.

        Parameters
        ----------
        ang_path : Path
            Path of offset file.
        layer : MonomerLayer
            Points layer of target monomers.
        update : bool, default is False
            Check if update monomer coordinates in place.
        """       
        mol: Molecules = layer.metadata[MOLECULES]
        shifts, angs = _read_shift_and_angle(ang_path)
        mol_shifted = mol.translate(shifts*self.scale)
        mol_shifted = Molecules.from_euler(pos=mol_shifted.pos, angles=angs, degrees=True)
        
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
                    vector_data, edge_width=0.3, edge_color="crimson", length=2.4,
                    name=vector_layer_name,
                    )
            layer.metadata[MOLECULES] = mol_shifted
        else:
            add_molecules(self.parent_viewer, mol_shifted, name="Molecules from PEET")
    
    @property
    def scale(self) -> float:
        from ...widgets import MTPropsWidget
        return self.find_ancestor(MTPropsWidget).tomogram.scale


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
        raise ValueError(f"Could not interpret data format of {ang_path}:\n{csv.head(5)}")
    return csv_data

def _read_shift_and_angle(path: str) -> Tuple[Union[np.ndarray, None], np.ndarray]:
    """Read offsets and angles from PEET project"""
    csv: pd.DataFrame = pd.read_csv(path)
    if "CCC" in csv.columns:
        ang_data = -csv[["EulerZ(1)", "EulerX(2)", "EulerZ(3)"]].values
        shifts_data = csv[["zOffset", "yOffset", "xOffset"]].values
    else:
        ang_data = _read_angle(path)
        shifts_data = None
    return shifts_data, ang_data
