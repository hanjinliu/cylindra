import os
from magicclass import (
    magicclass,
    magicmenu,
    magictoolbar,
    field,
    vfield,
    MagicTemplate,
    set_design,
    abstractapi,
)
from magicclass.widgets import Separator, ConsoleTextEdit
from magicclass.types import OneOf, SomeOf, Optional
from pathlib import Path
import impy as ip

from .widget_utils import FileFilter
from ._previews import view_image

from cylindra.utils import ceilint
from cylindra.ext.etomo import PEET
from cylindra.components import CylTomogram
from cylindra.const import GlobalVariables as GVar
from cylindra.widgets.global_variables import GlobalVariables

ICON_DIR = Path(__file__).parent / "icons"

# Menus


@magicmenu
class File(MagicTemplate):
    """File I/O."""

    open_image_loader = abstractapi()
    load_project = abstractapi()
    load_splines = abstractapi()
    load_molecules = abstractapi()
    sep0 = field(Separator)
    save_project = abstractapi()
    save_spline = abstractapi()
    save_molecules = abstractapi()
    sep1 = field(Separator)
    process_images = abstractapi()
    view_project = abstractapi()
    PEET = PEET


@magicmenu
class Image(MagicTemplate):
    """Image processing and visualization"""

    filter_reference_image = abstractapi()
    add_multiscale = abstractapi()
    set_multiscale = abstractapi()
    sep0 = field(Separator)
    open_sweeper = abstractapi()
    open_simulator = abstractapi()
    sep1 = field(Separator)
    sample_subtomograms = abstractapi()
    paint_cylinders = abstractapi()
    set_colormap = abstractapi()
    show_colorbar = abstractapi()


@magicmenu
class Splines(MagicTemplate):
    """Operations on splines"""

    show_splines = abstractapi()
    add_anchors = abstractapi()
    sep0 = field(Separator)

    @magicmenu
    class Orientation(MagicTemplate):
        """Adjust spline orientation."""

        invert_spline = abstractapi()
        align_to_polarity = abstractapi()
        auto_align_to_polarity = abstractapi()

    clip_spline = abstractapi()
    open_spline_clipper = abstractapi()
    delete_spline = abstractapi()
    sep1 = field(Separator)
    fit_splines = abstractapi()
    fit_splines_manually = abstractapi()
    refine_splines = abstractapi()
    molecules_to_spline = abstractapi()


@magicmenu
class Molecules_(MagicTemplate):
    """Operations on molecules"""

    @magicmenu
    class Mapping(MagicTemplate):
        """Map monomers along splines in several ways."""

        map_monomers = abstractapi()
        map_centers = abstractapi()
        map_along_pf = abstractapi()

    show_orientation = abstractapi()
    extend_molecules = abstractapi()
    translate_molecules = abstractapi()

    @magicmenu(name="Combine")
    class Combine(MagicTemplate):
        """Combine existing molecules."""

        concatenate_molecules = abstractapi()
        merge_molecule_info = abstractapi()

    @magicmenu(name="Features")
    class MoleculeFeatures(MagicTemplate):
        """Analysis based on molecule features."""

        show_molecule_features = abstractapi()
        filter_molecules = abstractapi()
        split_molecules = abstractapi()
        paint_molecules = abstractapi()
        sep0 = field(Separator)
        calculate_molecule_features = abstractapi()
        calculate_intervals = abstractapi()
        calculate_skews = abstractapi()
        sep1 = field(Separator)
        seam_search_by_feature = abstractapi()


@magicmenu
class Analysis(MagicTemplate):
    """Analysis of tomograms."""

    set_radius = abstractapi()
    local_ft_analysis = abstractapi()
    global_ft_analysis = abstractapi()
    sep0 = field(Separator)
    reanalyze_image = abstractapi()
    load_project_for_reanalysis = abstractapi()
    sep1 = field(Separator)
    open_spectra_measurer = abstractapi()
    open_subtomogram_analyzer = abstractapi()
    open_project_batch_analyzer = abstractapi()


@magicmenu
class Others(MagicTemplate):
    """Other menus."""

    @magicmenu
    class Macro:
        show_macro = abstractapi()
        show_full_macro = abstractapi()
        show_native_macro = abstractapi()
        sep0 = field(Separator)
        load_macro_file = abstractapi()
        run_file = abstractapi()

    open_command_palette = abstractapi()
    Global_variables = GlobalVariables
    open_logger = abstractapi()
    clear_cache = abstractapi()

    @magicmenu
    class Help(MagicTemplate):
        open_help = abstractapi()
        cylindra_info = abstractapi()
        report_issues = abstractapi()


# Toolbar


@magictoolbar(labels=False)
class toolbar(MagicTemplate):
    """Frequently used operations."""

    register_path = abstractapi()
    open_runner = abstractapi()
    sep0 = field(Separator)
    pick_next = abstractapi()
    auto_center = abstractapi()

    @magicmenu(icon=ICON_DIR / "adjust_intervals.png")
    class Adjust(MagicTemplate):
        """
        Adjust auto picker parameters.

        Attributes
        ----------
        stride : nm
            Stride length (nm) of auto picker.
        angle_deviation : float
            Angle deviation (degree) of auto picker.
        angle_precision : float
            Angle precision (degree) of auto picker.
        max_shifts : nm
            Maximum shift (nm) in auto centering.
        """

        stride = vfield(50.0, widget_type="FloatSlider", record=False).with_options(
            min=10, max=100
        )
        angle_deviation = vfield(
            12.0, widget_type="FloatSlider", record=False
        ).with_options(min=1.0, max=40.0, step=0.5)
        angle_precision = vfield(
            1.0, widget_type="FloatSlider", record=False
        ).with_options(min=0.5, max=5.0, step=0.1)
        max_shifts = vfield(20.0, record=False).with_options(
            min=1.0, max=50.0, step=0.5
        )

    sep1 = field(Separator)
    clear_current = abstractapi()
    clear_all = abstractapi()


# Runner


@magicclass(widget_type="groupbox", name="Fitting parameters", record=False)
class runner_params1:
    """
    Parameters used in spline fitting.

    Attributes
    ----------
    edge_sigma : nm
        Sharpness of dense-mode mask at the edges. Useful if cylindric structures are
        densely packed. Initial spline position must be 'almost' fitted in dense-mode.
    max_shift : nm
        Maximum shift in nm of manually selected spline to the true center.
    """

    edge_sigma = vfield(Optional[float], label="Edge sigma").with_options(
        value=2.0,
        options=dict(step=0.1, min=0.0, max=50.0),
        text="Don't use dense mode",
    )
    max_shift = vfield(5.0, label="Maximum shift (nm)").with_options(max=50.0, step=0.5)


@magicclass(widget_type="groupbox", name="Local-CFT parameters", record=False)
class runner_params2:
    """
    Parameters used in calculation of local properties.

    Attributes
    ----------
    interval : nm
        Interval of sampling points of cylinder fragments.
    ft_size: nm
        Longitudinal length of local discrete Fourier transformation used
        for structural analysis.
    paint : bool
        Check if paint the tomogram with the local properties.
    """

    interval = vfield(32.0, label="Interval (nm)").with_options(min=1.0, max=200.0)
    ft_size = vfield(32.0, label="Local-CFT window size (nm)").with_options(
        min=1.0, max=200.0
    )
    paint = vfield(True)


@magicclass(name="_Run cylindrical fitting", record=False)
class Runner(MagicTemplate):
    """
    Attributes
    ----------
    all_splines : bool
        Uncheck to select along which spline algorithms will be executed.
    splines : list of int
        Splines that will be analyzed
    bin_size : int
        Set to >1 to use binned image for fitting.
    n_refine : int
        Iteration number of spline refinement.
    local_props : bool
        Check if calculate local properties.
    global_props : bool
        Check if calculate global properties.
    """

    def _get_splines(self, _=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        from .main import CylindraMainWidget

        try:
            tomo = self.find_ancestor(CylindraMainWidget).tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_available_binsize(self, _=None) -> list[int]:
        from .main import CylindraMainWidget

        try:
            parent = self.find_ancestor(CylindraMainWidget)
        except Exception:
            return [1]
        if parent.tomogram is None:
            return [1]
        out = [x[0] for x in parent.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return sorted(out)

    all_splines = vfield(True).with_options(text="Run for all the splines.")
    splines = vfield(SomeOf[_get_splines]).with_options(visible=False)
    bin_size = vfield(OneOf[_get_available_binsize])

    fit = vfield(True, label="Fit splines")
    params1 = runner_params1
    n_refine = vfield(1, label="Refinement iteration").with_options(max=10)
    local_props = vfield(True, label="Calculate local properties")
    params2 = runner_params2
    global_props = vfield(True, label="Calculate global properties")

    @all_splines.connect
    def _toggle_spline_list(self, val: bool):
        self["splines"].visible = not val

    @fit.connect
    def _toggle_fit_params(self, visible: bool):
        self.params1.visible = visible

    @local_props.connect
    def _toggle_localprops_params(self, visible: bool):
        self.params2.visible = visible

    def _get_splines_to_run(self, w=None) -> list[int]:
        if self.all_splines:
            n_choices = len(self["splines"].choices)
            return list(range(n_choices))
        else:
            return self.splines

    def _get_max_shift(self, w=None):
        return self.params1.max_shift

    cylindrical_fit = abstractapi()


@magicclass(name="_Open image", record=False)
class ImageLoader(MagicTemplate):
    """
    Load an image file and process it before sending it to the viewer.

    Attributes
    ----------
    path : Path
        Path to the tomogram. Must be 3-D image.
    bin_size : int or list of int, default is [1]
        Initial bin size of image. Binned image will be used for visualization in the viewer.
        You can use both binned and non-binned image for analysis.
    filter_reference_image : bool, default is True
        Apply low-pass filter on the reference image (does not affect image data itself).
    """

    path = vfield(Path).with_options(filter=FileFilter.IMAGE)

    @magicclass(layout="horizontal", labels=False)
    class scale(MagicTemplate):
        """
        Scale of the image.

        Attributes
        ----------
        scale_value : float
            Scale of the image in nm/pixel.
        """

        scale_label = vfield("scale (nm)", widget_type="Label")
        scale_value = vfield(1.0, label="scale (nm)").with_options(
            min=0.001, step=0.0001, max=10.0
        )
        read_header = abstractapi()

    bin_size = vfield([1]).with_options(options={"min": 1, "max": 8})
    filter_reference_image = vfield(True)

    @scale.wraps
    def read_header(self):
        """Read scale from image header."""
        path = self.path
        if not os.path.exists(path) or not os.path.isfile(path):
            return
        img = ip.lazy_imread(path, chunks=GVar.daskChunk)
        scale = img.scale.x
        self.scale.scale_value = f"{scale:.4f}"
        if len(self.bin_size) < 2:
            self.bin_size = [ceilint(0.96 / scale)]

    open_image = abstractapi()

    @set_design(text="Preview")
    def preview_image(self):
        """Preview image at the path."""
        return view_image(self.path, parent=self)


@magicclass(name="Image info", record=False, widget_type="collapsible", labels=False)
class ImageInfo(MagicTemplate):
    """
    Attributes
    ----------
    source : str
        Source file of the image.
    shape : tuple of int
        Shape of the image.
    scale : float
        Scale of the image.
    """

    def __post_init__(self):
        self.text_edit.read_only = True

    def _from_tomogram(self, tomo: CylTomogram):
        img = tomo.image
        source = tomo.metadata.get("source", "Unknown")
        scale = tomo.scale
        shape_px = ", ".join(f"{s} px" for s in img.shape)
        shape_nm = ", ".join(f"{s*scale:.2f} nm" for s in img.shape)
        value = (
            f"File: {source}\n"
            f"Scale: {scale:.4f} nm/pixel\n"
            f"ZYX-Shape: ({shape_px}), ({shape_nm})"
        )
        self.text_edit.value = value

    text_edit = field(widget_type=ConsoleTextEdit)
