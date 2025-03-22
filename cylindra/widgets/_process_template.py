from typing import Annotated, Literal

import impy as ip
import numpy as np
import polars as pl
from acryo import pipe
from acryo.simulator import TomogramSimulator
from magicclass import (
    do_not_record,
    get_function_gui,
    impl_preview,
    magicmenu,
    set_design,
)
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.vispy import Vispy3DCanvas
from magicclass.logging import getLogger
from magicclass.types import Optional, Path

from cylindra.components import CylSpline
from cylindra.const import FileFilter, nm
from cylindra.const import PropertyNames as H
from cylindra.core import ACTIVE_WIDGETS
from cylindra.utils import ceilint, roundint
from cylindra.widgets._main_utils import degrees_to_rotator
from cylindra.widgets._widget_ext import SingleRotationEdit
from cylindra.widgets.subwidgets._child_widget import ChildWidget

_Logger = getLogger("cylindra")


@magicmenu(name="Template image")
class TemplateImage(ChildWidget):
    """Process/create template images for subtomogram alignment."""

    @set_design(text="Center by centroid")
    @do_not_record
    def center_by_centroid(
        self,
        path: Path.Read[FileFilter.IMAGE],
        save_path: Annotated[Optional[Path.Save[FileFilter.IMAGE]], {"text": "Overwrite"}] = None,
        view_in_canvas: bool = True,
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
        out.imsave(save_path)
        return self._plot_template(out, view_in_canvas)

    @set_design(text="PDB to image file")
    @do_not_record
    def convert_pdb_to_image(
        self,
        pdb_path: Path.Read[FileFilter.PDB],
        save_path: Path.Save[FileFilter.IMAGE],
        degrees: Annotated[list[tuple[Literal["z", "y", "x"], float]],{"layout": "vertical", "options": {"widget_type": SingleRotationEdit}}] = (),
        sigma: nm = 0.1,
        scale: nm = 0.2,
        view_in_canvas: bool = True,
    ):  # fmt: skip
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
        img = (
            ip.asarray(
                pipe.from_pdb(pdb_path, rotator).provide(scale),
                dtype=np.float32,
                axes="zyx",
            )
            .set_scale(zyx=scale, unit="nm")
            .gaussian_filter(sigma / scale)
        )
        img.imsave(save_path)
        return self._plot_template(img, view_in_canvas)

    @set_design(text="CSV to image file")
    @do_not_record
    def convert_csv_to_image(
        self,
        csv_path: Path.Read[FileFilter.CSV],
        save_path: Path.Save[FileFilter.IMAGE],
        sigma: nm = 0.1,
        scale: nm = 0.2,
        view_in_canvas: bool = True,
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
        img = (
            ip.asarray(img, dtype=np.float32, axes="zyx")
            .set_scale(zyx=scale, unit="nm")
            .gaussian_filter(sigma / scale)
        )
        img.imsave(save_path)
        return self._plot_template(img, view_in_canvas)

    def _get_template_path(self, *_):
        return self._get_main().sta._template_param()

    @set_design(text="Simulate cylinder")
    @dask_thread_worker.with_progress(desc="Simulating cylinder template")
    def simulate_cylinder(
        self,
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
        save_path: Path.Save[FileFilter.IMAGE],
        length: Annotated[nm, {"min": 2.0, "max": 100.0}] = 10.0,
        edge_sigma: Annotated[nm, {"min": 0.0, "max": 10.0}] = 1.0,
        scale: Annotated[nm, {"min": 0.05, "max": 10.0}] = 0.25,
        spacing: Annotated[
            nm, {"min": 0.2, "max": 100.0, "step": 0.01, "label": "spacing (nm)"}
        ] = 4.1,
        twist: Annotated[
            float, {"min": -45.0, "max": 45.0, "label": "twist (deg)"}
        ] = 0.0,
        start: Annotated[int, {"min": -50, "max": 50, "label": "start"}] = 3,
        npf: Annotated[int, {"min": 1, "label": "number of PF"}] = 13,
        rise_sign: Literal[1, -1] = -1,
        radius: Annotated[
            nm, {"min": 0.5, "max": 50.0, "step": 0.5, "label": "radius (nm)"}
        ] = 10.5,
        view_in_canvas: bool = True,
    ):
        """
        Simulate a cylinder template image from monomeric template image.

        Parameters
        ----------
        template_path : path-like
            Path to the monomeric template image.
        save_path : path-like
            Path to save the output image.
        length : float, default 10.0
            Length of the cylinder., by default 10.0
        edge_sigma : float, default 1.0
            Sigma of the Gaussian blur applied to the edge in the longitudinal
            direction.
        scale : float, default 0.25
            Pixel size in nm/pixel of the output image.
        spacing : float, default 4.1
            Longitudinal spacing between the molecules in nm.
        twist : float, default 0.0
            Twist angle in degree.
        start : int, default 3
            The start number of the cylinder.
        npf : int, default 13
            Number of PF in the cylinder.
        rise_sign : 1 or -1, default -1
            Sign of the rise. This parameter is used to make the start number positive.
            For microtubule, set this to -1.
        radius : float, default 10.5
            Radius of the cylinder in nm.
        view_in_canvas : bool, default True
            If True, the output image will be shown in the napari viewer canvas.
        """
        # (lz, ly, lx) will be the shape of the output image in nm.
        template_image = pipe.from_file(template_path).provide(scale)
        lz = lx = radius * 2 + template_image.shape[1] * scale + 1
        ly = length + edge_sigma * 8  # four sigma on each side
        shape_px = (roundint(lz / scale), roundint(ly / scale), roundint(lx / scale))
        mole = self._prep_molecules(
            (lz, ly, lx), spacing, twist, start, npf, rise_sign, radius
        )
        simulator = TomogramSimulator(scale=scale)
        simulator.add_molecules(mole, template_image)
        simulated_image = ip.asarray(simulator.simulate(shape_px), axes="zyx")

        # apply mask to soften the edge
        mask = ip.zeros(shape_px, dtype=np.bool_, axes="zyx")
        len_px = length / scale
        slice_y = slice(
            int((shape_px[1] - len_px) / 2), ceilint((shape_px[1] + len_px) / 2)
        )
        mask[:, slice_y, :] = True
        mask_soft = mask.smooth_mask(sigma=edge_sigma / scale, dilate_radius=0)
        img = (simulated_image * mask_soft).set_scale(zyx=scale, unit="nm")
        img.imsave(save_path)
        return dask_thread_worker.callback(self._plot_template).with_args(
            img, view_in_canvas
        )

    def _prep_molecules(
        self,
        shape: tuple[nm, nm, nm],
        spacing: nm,
        twist: float,
        start: int,
        npf: int,
        rise_sign: int,
        radius: nm,
    ):
        lz, ly, lx = shape
        extra_length = (start / 2 + 0.5) * spacing
        spl_start = (lz / 2, -extra_length, lx / 2)
        spl_end = (lz / 2, ly + extra_length, lx / 2)
        spl = CylSpline.line(spl_start, spl_end).with_config({"rise_sign": rise_sign})
        kwargs = {
            H.spacing: spacing,
            H.twist: twist,
            H.start: start,
            H.npf: npf,
            H.radius: radius,
        }
        model = spl.cylinder_model(**kwargs)
        return model.to_molecules(spl)

    def _plot_template(self, img: ip.ImgArray, view_in_canvas: bool = True):
        import matplotlib.pyplot as plt

        _Logger.print("Created image:")
        with _Logger.set_plt():
            _, axes = plt.subplots(ncols=3, figsize=(12, 4))
            plt.sca(axes[0])
            plt.imshow(img.mean(axis="z"), cmap="gray")
            plt.axis("off")
            plt.sca(axes[1])
            plt.imshow(img.mean(axis="y")[::-1], cmap="gray")
            plt.axis("off")
            plt.sca(axes[2])
            plt.imshow(img.mean(axis="x"), cmap="gray")
            plt.axis("off")
            plt.show()

        if view_in_canvas:
            self._get_main().sta._show_rec(img, name="Created template", store=False)
        return None


@impl_preview(TemplateImage.simulate_cylinder)
def _preview_simulate_cylinder(self: TemplateImage):
    canvas = Vispy3DCanvas()
    fgui = get_function_gui(self.simulate_cylinder)
    ACTIVE_WIDGETS.add(canvas)
    dummy_pos = np.array([[-10, -10, -10], [0, 0, 0], [10, 10, 10]])
    points = canvas.add_points(dummy_pos, face_color="lime", size=2.0)

    @fgui.changed.connect
    def _gui_changed():
        _p = fgui.asdict()
        lz = lx = _p["radius"] * 2 + _p["spacing"] + 1
        ly = 15.0
        mole = self._prep_molecules(
            (lz, ly, lx), spacing=_p["spacing"], twist=_p["twist"], start=_p["start"],
            npf=_p["npf"], rise_sign=_p["rise_sign"], radius=_p["radius"]
        )  # fmt: skip
        points.data = mole.pos
        canvas.camera.center = np.mean(mole.pos, axis=0)

    canvas.native.setParent(self._get_main().native, canvas.native.windowFlags())
    canvas.show()
    _gui_changed()
    canvas.width = 400
    canvas.height = 400

    is_active = yield
    if not is_active:
        fgui.changed.disconnect(_gui_changed)
        canvas.close()
    return None
