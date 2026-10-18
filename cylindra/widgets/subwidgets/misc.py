import glob
from pathlib import Path

from acryo.tilt import NoWedge, SingleAxis
from magicclass import (
    MagicTemplate,
    abstractapi,
    field,
    magicclass,
    set_design,
    vfield,
)
from magicclass.widgets import ConsoleTextEdit, ToggleButtons
from magicgui.widgets import TextEdit
from qtpy import QtGui
from qtpy.QtCore import Qt

from cylindra import _io
from cylindra._previews import view_image
from cylindra.components import CylTomogram
from cylindra.const import FileFilter, ImageFilter
from cylindra.project import CylindraProject, TomogramDefaults
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

    def __init__(self, value=None, **kwargs):
        super().__init__()
        if name := kwargs.get("name"):
            self.name = name
        if label := kwargs.get("label"):
            self.label = label
        if value is not None:
            self.value = value

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
    bin_size : int or list of int, default [1]
        Initial bin size of image. Binned image will be used for visualization in the
        viewer. You can use both binned and non-binned image for analysis.
    filter : ImageFilter
        Choose filter for the reference image (does not affect image data itself).
    invert : bool
        Invert intensity of the image. If tomogram is light-background, check this.
    invert_reference : bool
        Invert intensity of the reference image if the user-supplied reference image
        is used.
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

    @use_reference.connect
    def _on_use_reference_change(self, use_ref: bool):
        """Show or hide the reference path field based on the use_reference value."""
        self["reference_path"].visible = use_ref
        self["fix_reference_scale"].visible = use_ref
        self["eager"].visible = not use_ref
        self["invert_reference"].visible = use_ref

        self["open_image"].visible = not use_ref
        self["open_image_with_reference"].visible = use_ref

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
        scan_header_or_defaults = abstractapi()

    tilt_model = field(TiltModelEdit)
    bin_size = vfield([1]).with_options(options={"min": 1, "max": 32})
    filter = vfield(ImageFilter | None, widget_type=ToggleButtons).with_options(
        value=ImageFilter.Lowpass
    )

    @magicclass(layout="horizontal", labels=False)
    class invert_section(MagicTemplate):
        invert = abstractapi()
        invert_reference = abstractapi()

    invert = vfield(False, label="Invert intensity", location=invert_section)
    invert_reference = vfield(
        False, label="Invert reference intensity", location=invert_section
    )
    eager = vfield(False, label="Load the entire image into memory")
    fix_reference_scale = vfield(True)
    cache_image = vfield(False, label="Cache image on SSD")

    @set_design(text="Scan", max_width=60, location=scale)
    def scan_header_or_defaults(self):
        """Scan header or the default setting to update this GUI."""
        path = _resolve_path_str(self.path)
        if not path.exists() or not path.is_file():
            return
        # try to read header
        header = _io.read_header(path)
        scale = header.scale["x"]
        self.scale.scale_value = f"{scale:.4f}"
        bin_size_new = None
        if len(self.bin_size) < 2:
            bin_size_new = [ceilint(0.96 / scale)]
        # look for mdoc file
        if (tilt_angle := find_tilt_angles(path.parent)) is not None:
            tilt_min = round(tilt_angle.min(), 1)
            tilt_max = round(tilt_angle.max(), 1)
            self.tilt_model.xrange.value = tilt_min, tilt_max
            self.tilt_model.yrange.value = tilt_min, tilt_max
        if default_setting := TomogramDefaults.from_dir(path.parent):
            # apply default settings
            if (_scale := default_setting.scale) is not None:
                self.scale.scale_value = f"{_scale:.4f}"
            if (_inv := default_setting.invert) is not None:
                self.invert = _inv
            if (_bin_size := default_setting.bin_size) is not None:
                bin_size_new = _bin_size
            if (_mw := default_setting.missing_wedge) is not None:
                self.tilt_model.value = _mw.as_param()
            if (_filter := default_setting.filter) is not None:
                self.filter = _filter
            if _img_ref := default_setting.resolve_reference_path(path):
                self.use_reference = True
                self.reference_path = _img_ref
                if (_inv_ref := default_setting.invert_reference) is not None:
                    self.invert_reference = _inv_ref
        if bin_size_new is not None:
            # NOTE: Qt list widget cannot be updated twice; otherwise the child widgets
            # float out.
            self.bin_size = bin_size_new

    @set_design(text="Preview")
    def preview_image(self):
        """Preview image at the path.

        The preview will NOT consider the `invert` and `filter` settings.
        """
        return view_image(self.path, parent=self)

    open_image = abstractapi()
    open_image_with_reference = abstractapi()


@magicclass(name="_Open tilt series", record=False)
class TiltSeriesLoader(MagicTemplate):
    """Widget for opening a tilt series.

    Attributes
    ----------
    path : Path
        Path to the file that describes the tilt series.
    create_reference : bool
        Whether to create a coarse tomogram from the tilt series.
    reference_bin_size : int
        Bin size of the coarse tomogram for visualization.
    reference_path : Path
        Path to the coarse tomogram reconstructed from the tilt series, if available.
    tomo_hand : int
        Handedness of the tomogram.
    tomo_height : int
        Height of the tomogram in pixels for coarse tomogram reconstruction (binned).
    save_reference : bool
        Whether to save the coarse tomogram after creation.
    bin_size : list of int
        Bin sizes available for analysis.
    filter : ImageFilter | None
        Filter to apply to the tilt series images for visualization.
    invert_reference : bool
        Whether to invert the intensity of the reference tomogram.
    cache_image : bool
        Whether to cache the image on the SSD.
    """

    path = vfield(Path).with_options(filter=FileFilter.TILT_SERIES)

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class create_ref(MagicTemplate):
        """Create reference image now."""

        create_reference = abstractapi()
        reference_bin_size = abstractapi()

    create_reference = vfield(True, label="Create coarse tomogram", location=create_ref)
    reference_bin_size = vfield(
        6, label="... with bin size", location=create_ref
    ).with_options(min=2, max=32)
    reference_path = vfield(Path).with_options(filter=FileFilter.IMAGE)

    @magicclass(layout="horizontal", labels=False)
    class scale(MagicTemplate):
        """Scale of the tilt series.

        Attributes
        ----------
        scale_value : float
            Scale of the tilt series in nm/pixel.
        """

        scale_label = vfield("scale (nm)", widget_type="Label")
        scale_value = vfield(1.0).with_options(min=1e-3, step=1e-4, max=10.0)
        scan_header_or_defaults = abstractapi()

    tomo_hand = vfield(
        int, label="Tomogram handedness", widget_type=ToggleButtons
    ).with_choices([-1, 1])
    tomo_height = vfield(200, label="Tomogram height (pixels)")
    save_reference = vfield(True, label="Save coarse tomogram")
    bin_size = vfield([2, 4]).with_options(options={"min": 1, "max": 32})
    filter = vfield(ImageFilter | None, widget_type=ToggleButtons).with_options(
        value=ImageFilter.Lowpass
    )

    invert_reference = vfield(True, label="Invert reference intensity")
    cache_image = vfield(False, label="Cache image on SSD")

    @create_reference.connect
    def _on_create_reference_change(self, create_ref: bool):
        """Show or hide the reference path field based on the create_reference value."""
        self["reference_path"].visible = not create_ref
        self["reference_bin_size"].visible = create_ref
        self["tomo_height"].visible = create_ref
        self["invert_reference"].visible = not create_ref
        self["open_tilt_series"].visible = create_ref
        self["open_tilt_series_with_reference"].visible = not create_ref
        self["save_reference"].visible = create_ref

    @set_design(text="Scan", max_width=60, location=scale)
    def scan_header_or_defaults(self):
        """Scan header or the default setting to update this GUI."""
        path = _resolve_path_str(self.path)
        if not path.exists() or not path.is_file():
            return
        if scale_ := _infer_scale_for_tilt_series(path):
            self.scale.scale_value = f"{scale_:.4f}"
        ref_found = False
        if default_setting := TomogramDefaults.from_dir(path.parent):
            if (_tomo_hand := default_setting.tilt_series.tomo_hand) is not None:
                self.tomo_hand = _tomo_hand
            if (_tomo_height := default_setting.tilt_series.tomo_height) is not None:
                self.tomo_height = _tomo_height
            if (_scale := default_setting.scale) is not None:
                self.scale.scale_value = f"{_scale:.4f}"
            if (
                _bin_size := default_setting.tilt_series.reference_bin_size
            ) is not None:
                self.reference_bin_size = _bin_size
            if (_filter := default_setting.filter) is not None:
                self.filter = _filter
            if _img_ref := default_setting.resolve_reference_path(path):
                self.create_reference = False
                self.reference_path = _img_ref
                ref_found = True
                if (_inv_ref := default_setting.invert_reference) is not None:
                    self.invert_reference = _inv_ref
        if not ref_found:
            self.create_reference = True

    def __post_init__(self):
        self._on_create_reference_change(self.create_reference)

    @set_design(text="Preview")
    def preview_image(self):
        """Preview image at the path."""
        return view_image(
            _get_tilt_series_image_path(self.path),
            parent=self,
            default_bin_size=4,
        )

    open_tilt_series = abstractapi()
    open_tilt_series_with_reference = abstractapi()


def _get_tilt_series_image_path(path) -> str | list[str] | None:
    """Get the tilt series image path(s) from the given tilt series info path."""
    path = Path(path)
    if path.suffix == ".edf":
        return path.with_name(f"{path.stem}.mrc").as_posix()
    elif path.suffix == ".star":
        import starfile_rs

        df = starfile_rs.read_star(path).first().to_polars()
        # TODO: to absolute path
        return df.sort("rlnTomoNominalStageTiltAngle")["rlnMicrographName"].to_list()
    return None


def _infer_scale_for_tilt_series(path) -> float | None:
    """Infer the scale from the given tilt series info path."""
    path = _get_tilt_series_image_path(path)
    if path is None:
        return None
    if isinstance(path, list):
        path = path[0]
    path = Path(path)

    header = _io.read_header(path)
    scale = header.scale["x"]
    return scale


def _resolve_path_str(path_input):
    path_str = str(path_input)
    if ";" in path_str:
        path = Path(path_str.split(";")[0])  # only use the first path
    elif "*" in path_str or "?" in path_str or "[" in path_str:
        # use glob to find the first file
        path = Path(sorted(glob.glob(path_str, recursive=True))[0])
    else:
        path = Path(path_str)
    return path


def _format_nbytes(nbytes: int) -> str:
    if nbytes < 1024**3:
        return f"{nbytes / 1024**2:.2f} MB"
    return f"{nbytes / 1024**3:.2f} GB"


@magicclass(record=False, widget_type="collapsible", labels=False)
class GeneralInfo(MagicTemplate):
    """General information of the current project.

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
        fpath = (tomo._orig_or_read_path() or "Unknown").replace(";", " + ")
        scale = tomo.scale
        shape_px = ", ".join(f"{s} px" for s in tomo.shape)
        shape_nm = ", ".join(f"{s * scale:.2f} nm" for s in tomo.shape)
        orig_nm = ", ".join(f"{o:.2f} nm" for o in tomo.origin)
        if isinstance(tomo.tilt_model, NoWedge):
            tilt_range = "No missing wedge"
        elif isinstance(tomo.tilt_model, SingleAxis):
            deg0, deg1 = tomo.tilt_model.tilt_range
            axis = type(tomo.tilt_model).__name__[-1].lower()
            tilt_range = f"{deg0:.1f}° — {deg1:.1f}° (axis: {axis})"
        else:
            tilt_range = repr(tomo.tilt_model)
        if (ts := tomo.tilt_series) is not None:
            # the bin-1 image is not in memory nor on disk
            if (nbytes := ts.nbytes) is None:
                size = "Unknown (tilt series)"
            else:
                size = f"{_format_nbytes(nbytes)} (tilt series)"
            ref_path = tomo.metadata.get("display_reference", None) or (
                tomo.metadata.get("reference", None)
            )
            if ref_path:
                fpath += f'<br><span style="color: gray;">Reference:</span> {Path(ref_path).as_posix()}'  # fmt: skip
        else:
            size = _format_nbytes(img.value.nbytes)

        self.image_info.value = (
            f'<span style="color: gray;">File:</span> {fpath}<br>'
            f'<span style="color: gray;">Scale:</span> {scale:.4f} nm/pixel<br>'
            f'<span style="color: gray;">ZYX shape:</span> ({shape_px})<br>'
            f'<span style="color: gray;">ZYX shape (nm):</span> ({shape_nm})<br>'
            f'<span style="color: gray;">ZYX origin (nm):</span> ({orig_nm})<br>'
            f'<span style="color: gray;">Tilt range:</span> {tilt_range}<br>'
            f'<span style="color: gray;">Data type:</span> {img.dtype}<br>'
            f'<span style="color: gray;">Data size:</span> {size}'
        )

    def _refer_project(self, project: CylindraProject):
        self.project_desc.value = project.project_description

    label0 = field("\n<b>Image information</b>", widget_type="Label")
    image_info = field(widget_type=ConsoleTextEdit)
    label1 = field("\n<b>Project description</b>", widget_type="Label")
    project_desc = field(widget_type=TextEdit)
