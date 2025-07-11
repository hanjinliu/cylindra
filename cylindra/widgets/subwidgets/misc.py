import glob
from pathlib import Path

import impy as ip
from acryo.tilt import NoWedge, SingleAxis
from magicclass import (
    MagicTemplate,
    abstractapi,
    field,
    magicclass,
    set_design,
    vfield,
)
from magicclass.widgets import ConsoleTextEdit
from magicgui.widgets import TextEdit
from qtpy import QtGui
from qtpy.QtCore import Qt

from cylindra._previews import view_image
from cylindra.components import CylTomogram
from cylindra.const import FileFilter, ImageFilter
from cylindra.project import CylindraProject
from cylindra.utils import ceilint, find_tilt_angles


@magicclass(widget_type="groupbox")
class TiltModelEdit(MagicTemplate):
    """Parameters for the tilt model.

    Attributes
    ----------
    axis : str
        Tilt axis.
    xrange : tuple of float
        Range of the tilt around x-axis in degree.
    yrange : tuple of float
        Range of the tilt around y-axis in degree.
    """

    axis = vfield("y").with_choices(["none", "x", "y", "dual"])
    xrange = field(tuple[float, float]).with_options(
        visible=False, options={"min": -90, "max": 90, "step": 1}, value=(-60, 60)
    )
    yrange = field(tuple[float, float]).with_options(
        visible=True, options={"min": -90, "max": 90, "step": 1}, value=(-60, 60)
    )

    @axis.connect
    def _on_axis_change(self, axis: str):
        self.xrange.visible = axis in ("x", "dual")
        self.yrange.visible = axis in ("y", "dual")

    @property
    def value(self):
        match self.axis:
            case "none":
                out = None
            case "x":
                out = {"kind": self.axis, "range": self.xrange.value}
            case "y":
                out = {"kind": self.axis, "range": self.yrange.value}
            case "dual":
                out = {
                    "kind": "dual",
                    "xrange": self.xrange.value,
                    "yrange": self.yrange.value,
                }
            case axis:
                raise ValueError(f"Unknown axis {axis!r}.")
        return out

    @value.setter
    def value(self, val):
        if val is None:
            self.axis = "none"
        elif isinstance(val, dict):
            self.axis = val["kind"]
            if self.axis == "dual":
                self.xrange.value = val["xrange"]
                self.yrange.value = val["yrange"]
            elif rng := val.get("range"):
                self.xrange.value = rng
                self.yrange.value = rng
            else:
                pass
        else:
            low, high = val
            self.axis = "y"
            self.xrange.value = self.yrange.value = (low, high)


@magicclass(name="_Open image", record=False)
class ImageLoader(MagicTemplate):
    """Load an image file and process it before sending it to the viewer.

    Attributes
    ----------
    path : Path
        Path to the tomogram. Must be a 3-D image.
    use_reference : bool
        Use a user-supplied reference image. The reference image is usually a binned
        or denoised tomogram that will help to visualize the tomogram in the viewer.
    reference_path : Path
        Path to the reference image. Must be a 3-D image.
    autofill_pattern : str
        Pattern to autofill the reference image path. The pattern should contain a
        placeholder `{}` that will be replaced with the name of the tomogram file. For
        example, if you want to autofill like "TS_01.mrc" -> "TS_01_bin4.mrc", you can
        set the pattern to "{}_bin4.mrc", or "{}_*.mrc" for more flexibility.
    bin_size : int or list of int, default [1]
        Initial bin size of image. Binned image will be used for visualization in the
        viewer. You can use both binned and non-binned image for analysis.
    filter : ImageFilter
        Choose filter for the reference image (does not affect image data itself).
    invert : bool
        Invert intensity of the image. If tomogram is light-background, check this.
    eager : bool
        Load the entire image into memory to avoid disk access.
    cache_image : bool
        Cache image on SSD for faster access. Cached image will be deleted when new
        tomogram is loaded or the application is closed.
    fix_reference_scale : bool
        Fix the pixel size of the reference image if the scale of the raw tomogram was
        overridden. For example, if the header of the tomogram says 0.3 nm/pixel but is
        overridden to 0.27 nm/pixel, and the reference image has a scale of 0.9 nm/pixel
        (possibly binned by 3), the reference image will be rescaled to
        0.9 / 0.3 * 0.27 = 0.81 nm/pixel.
    """

    path = vfield(Path).with_options(filter=FileFilter.IMAGE)
    use_reference = vfield(False, label="Use user-supplied reference image")
    reference_path = vfield(Path).with_options(filter=FileFilter.IMAGE)
    autofill_pattern = vfield("{}_*.mrc")

    @use_reference.connect
    def _on_use_reference_change(self, use_ref: bool):
        """Show or hide the reference path field based on the use_reference value."""
        self["reference_path"].visible = use_ref
        self["autofill_pattern"].visible = use_ref
        self["fix_reference_scale"].visible = use_ref
        self["eager"].visible = not use_ref

        self["open_image"].visible = not use_ref
        self["open_image_with_reference"].visible = use_ref

    @path.connect
    def _on_path_change(self, path: Path):
        """Autofill the reference path if use_reference is True."""
        path = Path(path)
        if path.exists() and self.autofill_pattern:
            ref_path_ptn = str(path.parent / self.autofill_pattern.format(path.stem))
            ref_path = next(iter(glob.glob(ref_path_ptn)), None)
            if ref_path is not None:
                self.reference_path = Path(ref_path)

    def __post_init__(self):
        self._on_use_reference_change(self.use_reference)

    @magicclass(layout="horizontal", labels=False)
    class scale(MagicTemplate):
        """Scale of the image.

        Attributes
        ----------
        scale_value : float
            Scale of the image in nm/pixel.
        """

        scale_label = vfield("scale (nm)", widget_type="Label")
        scale_value = vfield(1.0).with_options(min=1e-3, step=1e-4, max=10.0)
        scan_header = abstractapi()

    tilt_model = field(TiltModelEdit)
    bin_size = vfield([1]).with_options(options={"min": 1, "max": 32})
    filter = vfield(ImageFilter | None).with_options(value=ImageFilter.Lowpass)
    invert = vfield(False, label="Invert intensity")
    eager = vfield(False, label="Load the entire image into memory")
    cache_image = vfield(False, label="Cache image on SSD")
    fix_reference_scale = vfield(True)

    @set_design(text="Scan header", max_width=90, location=scale)
    def scan_header(self):
        """Scan scale from image header and set the optimal bin size."""

        path = Path(self.path)
        if not path.exists() or not path.is_file():
            return
        header = ip.read_header(path)
        scale = header.scale["x"]
        self.scale.scale_value = f"{scale:.4f}"
        if len(self.bin_size) < 2:
            self.bin_size = [ceilint(0.96 / scale)]
        # look for mdoc file
        if (tilt_angle := find_tilt_angles(path.parent)) is not None:
            tilt_min = round(tilt_angle.min(), 1)
            tilt_max = round(tilt_angle.max(), 1)
            self.tilt_model.xrange.value = tilt_min, tilt_max
            self.tilt_model.yrange.value = tilt_min, tilt_max
        return None

    @set_design(text="Preview")
    def preview_image(self):
        """Preview image at the path.

        The preview will NOT consider the `invert` and `filter` settings.
        """
        return view_image(self.path, parent=self)

    open_image = abstractapi()
    open_image_with_reference = abstractapi()


@magicclass(record=False, widget_type="collapsible", labels=False)
class GeneralInfo(MagicTemplate):
    """
    General information of the current project.

    Attributes
    ----------
    image_info : str
        Information of the image.
    project_desc : str
        User editable description of the project.
    """

    def __post_init__(self):
        self.image_info.read_only = True
        self.image_info.native.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.image_info.native.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)

    def _refer_tomogram(self, tomo: CylTomogram):
        img = tomo.image
        fpath = tomo._orig_or_read_path() or "Unknown"
        scale = tomo.scale
        shape_px = ", ".join(f"{s} px" for s in img.shape)
        shape_nm = ", ".join(f"{s*scale:.2f} nm" for s in img.shape)
        if isinstance(tomo.tilt_model, NoWedge):
            tilt_range = "No missing wedge"
        elif isinstance(tomo.tilt_model, SingleAxis):
            deg0, deg1 = tomo.tilt_model.tilt_range
            axis = type(tomo.tilt_model).__name__[-1].lower()
            tilt_range = f"{deg0:.1f}° — {deg1:.1f}° (axis: {axis})"
        else:
            tilt_range = repr(tomo.tilt_model)
        value = (
            f"File: {fpath}\n"
            f"Scale: {scale:.4f} nm/pixel\n"
            f"ZYX-Shape: ({shape_px})\n"
            f"ZYX-Shape (nm): ({shape_nm})\n"
            f"Data type: {img.dtype}\n"
            f"Tilt range: {tilt_range}"
        )
        self.image_info.value = value

    def _refer_project(self, project: CylindraProject):
        self.project_desc.value = project.project_description

    label0 = field("\n<b>Image information</b>", widget_type="Label")
    image_info = field(widget_type=ConsoleTextEdit)
    label1 = field("\n<b>Project description</b>", widget_type="Label")
    project_desc = field(widget_type=TextEdit)
