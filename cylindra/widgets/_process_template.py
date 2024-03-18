from typing import Annotated, Literal

import impy as ip
import polars as pl
from acryo import pipe
from magicclass import MagicTemplate, do_not_record, magicmenu, set_design
from magicclass.types import Optional, Path

from cylindra.const import FileFilter, nm
from cylindra.widgets._main_utils import degrees_to_rotator
from cylindra.widgets._widget_ext import SingleRotationEdit


@magicmenu(name="Template image")
class TemplateImage(MagicTemplate):
    """Process/create template images for subtomogram alignment."""

    @set_design(text="Center by centroid")
    @do_not_record
    def center_by_centroid(
        self,
        path: Path.Read[FileFilter.IMAGE],
        save_path: Annotated[Optional[Path.Save[FileFilter.IMAGE]], {"text": "Overwrite"}] = None,
    ):  # fmt: skip
        """
        Center the image by its centroid.

        Parameters
        ----------
        path : Path
            Path to the input image file.
        save_path : Path, default None
            Path to save the output image file. If None, the input file will be
            overwritten.
        """
        if save_path is None:
            save_path = path
        img = ip.imread(path)
        out = pipe.center_by_mass().convert(img.value, img.scale.y)
        out = ip.asarray(out, like=img)
        return out.imsave(save_path)

    @set_design(text="PDB to image file")
    @do_not_record
    def convert_pdb_to_image(
        self,
        pdb_path: Path.Read[FileFilter.PDB],
        save_path: Path.Save[FileFilter.IMAGE],
        degrees: Annotated[
            list[tuple[Literal["z", "y", "x"], float]],
            {"layout": "vertical", "options": {"widget_type": SingleRotationEdit}},
        ] = (),
        sigma: nm = 0.1,
        scale: nm = 0.2,
    ):
        """
        Convert a PDB file to an 3D image file.

        This method uses 3D histogram to make an approximate image of the PDB file.

        Parameters
        ----------
        pdb_path : Path
            Path to the PDB file.
        save_path : Path
            Path to save the image file.
        degrees : list of (str, float)
            Rotation angles in degree.
        sigma : float, default 0.1
            Standard deviation of the Gaussian filter in nm.
        scale : float, default 0.2
            Scale of the output image in nm/pixel.
        """
        rotator = degrees_to_rotator(degrees)
        img = pipe.from_pdb(pdb_path, rotator).provide(scale)
        return (
            ip.asarray(img, axes="zyx")
            .set_scale(zyx=scale, unit="nm")
            .gaussian_filter(sigma / scale)
            .imsave(save_path)
        )

    @set_design(text="CSV to image file")
    @do_not_record
    def convert_csv_to_image(
        self,
        csv_path: Path.Read[FileFilter.CSV],
        save_path: Path.Save[FileFilter.IMAGE],
        sigma: nm = 0.1,
        scale: nm = 0.2,
    ):
        """
        Convert a CSV file to an 3D image file.

        The header must contain "x", "y", "z" and optionally "weights" columns.

        Parameters
        ----------
        csv_path : Path
            Path to the CSV file.
        save_path : Path
            Path to save the image file.
        sigma : float, default 0.1
            Standard deviation of the Gaussian filter in nm.
        scale : float, default 0.2
            Scale of the output image in nm/pixel.
        """
        df = pl.read_csv(csv_path)
        if not {"x", "y", "z"}.issubset(df.columns):
            raise ValueError("The CSV file must contain 'x', 'y' and 'z' columns.")
        atoms = df.select(["z", "y", "x"]).to_numpy()
        if "weights" in df.columns:
            weights = df["weights"].to_numpy()
        else:
            weights = None
        img = pipe.from_atoms(atoms, weights=weights).provide(scale)
        return (
            ip.asarray(img, axes="zyx")
            .set_scale(zyx=scale, unit="nm")
            .gaussian_filter(sigma / scale)
            .imsave(save_path)
        )
