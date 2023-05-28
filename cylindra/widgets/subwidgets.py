import os
from typing import Annotated, Sequence, Union
from magicgui.widgets import TextEdit
from magicclass import (
    do_not_record,
    magicclass,
    magicmenu,
    magictoolbar,
    field,
    vfield,
    MagicTemplate,
    set_design,
    bind_key,
    abstractapi,
)
from magicclass.widgets import Separator, ConsoleTextEdit
from magicclass.types import SomeOf, Optional
from magicclass.logging import getLogger
from pathlib import Path
import impy as ip

from .widget_utils import FileFilter
from ._previews import view_image

from cylindra.utils import ceilint
from cylindra.ext.etomo import PEET
from cylindra.components import CylTomogram, AutoCorrelationPicker
from cylindra.const import GlobalVariables as GVar, nm, ImageFilter
from cylindra.project import CylindraProject
from cylindra.widgets.global_variables import GlobalVariablesMenu

ICON_DIR = Path(__file__).parent / "icons"
_Logger = getLogger("cylindra")

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


@magicmenu(name="Image")
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
    backpaint_molecule_density = abstractapi()
    set_colormap = abstractapi()
    show_colorbar = abstractapi()


@magicmenu
class Splines(MagicTemplate):
    """Operations on splines"""

    show_splines = abstractapi()
    show_splines_as_meshes = abstractapi()
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
    sep2 = field(Separator)
    set_spline_props = abstractapi()
    molecules_to_spline = abstractapi()


@magicmenu(name="Molecules")
class MoleculesMenu(MagicTemplate):
    """Operations on molecules"""

    @magicmenu
    class Mapping(MagicTemplate):
        """Map monomers along splines in several ways."""

        map_monomers = abstractapi()
        map_monomers_with_extensions = abstractapi()
        map_centers = abstractapi()
        map_along_pf = abstractapi()

    show_orientation = abstractapi()
    translate_molecules = abstractapi()

    @magicmenu(name="Combine")
    class Combine(MagicTemplate):
        """Combine existing molecules."""

        concatenate_molecules = abstractapi()
        merge_molecule_info = abstractapi()

    @magicmenu(name="Features")
    class MoleculeFeatures(MagicTemplate):
        """Analysis based on molecule features."""

        filter_molecules = abstractapi()
        split_molecules = abstractapi()
        sep0 = field(Separator)
        calculate_molecule_features = abstractapi()
        calculate_intervals = abstractapi()
        calculate_skews = abstractapi()
        calculate_radii = abstractapi()
        calculate_lateral_angles = abstractapi()
        sep1 = field(Separator)
        seam_search_by_feature = abstractapi()

    @magicmenu(name="Visualize")
    class Visualize(MagicTemplate):
        """Visualize molecules analysis results."""

        show_molecule_features = abstractapi()
        paint_molecules = abstractapi()
        plot_molecule_feature = abstractapi()
        show_molecules_colorbar = abstractapi()


@magicmenu
class Analysis(MagicTemplate):
    """Analysis of tomograms."""

    measure_radius = abstractapi()
    local_ft_analysis = abstractapi()
    global_ft_analysis = abstractapi()
    sep0 = field(Separator)
    reanalyze_image = abstractapi()
    load_project_for_reanalysis = abstractapi()
    sep1 = field(Separator)
    open_spectra_measurer = abstractapi()
    open_subtomogram_analyzer = abstractapi()
    open_project_batch_analyzer = abstractapi()
    sep2 = field(Separator)
    repeat_command = abstractapi()


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
    GlobalVariables = GlobalVariablesMenu
    open_logger = abstractapi()

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

    @magicmenu(icon=ICON_DIR / "adjust_intervals.svg", record=False)
    class Adjust(MagicTemplate):
        """
        Adjust auto picker parameters.

        Attributes
        ----------
        interval : nm
            Interval (nm) of auto picking.
        max_angle : float
            Maximum angle (degree) to search in auto picking.
        angle_step : float
            Step of angle (degree) to search in auto picking.
        max_shifts : nm
            Maximum shift (nm) to search in auto picking.
        """

        interval = vfield(50.0, widget_type="FloatSlider").with_options(min=10, max=100)  # fmt: skip
        max_angle = vfield(12.0, widget_type="FloatSlider").with_options(min=1.0, max=40.0, step=0.5)  # fmt: skip
        angle_step = vfield(1.0, widget_type="FloatSlider").with_options(min=0.5, max=5.0, step=0.1)  # fmt: skip
        max_shifts = vfield(20.0).with_options(min=1.0, max=50.0, step=0.5)

        def _get_picker(self) -> AutoCorrelationPicker:
            """Make a picker with current parameters."""
            return AutoCorrelationPicker(self.interval, self.max_angle, self.angle_step, self.max_shifts)  # fmt: skip

    sep1 = field(Separator)

    clear_current = abstractapi()
    clear_all = abstractapi()

    sep2 = field(Separator)

    @do_not_record
    @set_design(icon=ICON_DIR / "undo.svg")
    @bind_key("Ctrl+Z")
    def undo(self):
        """Undo last action."""
        if len(self.macro.undo_stack["undo"]) == 0:
            raise RuntimeError("Undo stack is empty.")
        expr = self.macro[-1]
        self.macro.undo()
        return _Logger.print_html(f"Undo: <code>{expr}</code>")

    @do_not_record
    @set_design(icon=ICON_DIR / "redo.svg")
    @bind_key("Ctrl+Y")
    def redo(self):
        """Redo last undo action."""
        if len(self.macro.undo_stack["redo"]) == 0:
            raise RuntimeError("Redo stack is empty.")
        self.macro.redo()
        expr = self.macro[-1]
        return _Logger.print_html(f"Redo: <code>{expr}</code>")


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
    infer_polarity : bool
        Check if infer spline polarity after run.
    """

    def _get_parent(self):
        from .main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget)

    def _get_splines(self, _=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        try:
            tomo = self._get_parent().tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_available_binsize(self, _=None) -> list[int]:
        try:
            parent = self._get_parent()
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
    bin_size = vfield(int).with_choices(choices=_get_available_binsize)

    fit = vfield(True, label="Fit splines")
    params1 = runner_params1
    n_refine = vfield(1, label="Refinement iteration").with_options(max=10)
    local_props = vfield(True, label="Calculate local properties")
    params2 = runner_params2
    global_props = vfield(True, label="Calculate global properties")
    infer_polarity = vfield(True, label="Infer polarity")
    map_monomers = vfield(True, label="Map monomers")

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
        if self.fit:
            return self.params1.max_shift
        else:
            return -1.0

    @set_design(text="Run")
    @do_not_record(recursive=False)
    def run_workflow(
        self,
        splines: Annotated[Sequence[int], {"bind": _get_splines_to_run}] = (),
        bin_size: Annotated[int, {"bind": bin_size}] = 1,
        max_shift: Annotated[nm, {"bind": _get_max_shift}] = 5.0,
        edge_sigma: Annotated[nm, {"bind": params1.edge_sigma}] = 2.0,
        n_refine: Annotated[int, {"bind": n_refine}] = 1,
        local_props: Annotated[bool, {"bind": local_props}] = True,
        interval: Annotated[nm, {"bind": params2.interval}] = 32.0,
        ft_size: Annotated[nm, {"bind": params2.ft_size}] = 32.0,
        global_props: Annotated[bool, {"bind": global_props}] = True,
        paint: Annotated[bool, {"bind": params2.paint}] = True,
        infer_polarity: Annotated[bool, {"bind": infer_polarity}] = True,
        map_monomers: Annotated[bool, {"bind": map_monomers}] = False,
    ):
        """Run workflow."""
        parent = self._get_parent()
        if parent.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if parent.tomogram.n_splines == 0:
            raise ValueError("No spline is added to the viewer canvas.")
        elif len(splines) == 0:
            splines = list(range(parent.tomogram.n_splines))
        parent._runner.close()

        if max_shift > 0.0:
            parent.fit_splines(
                splines=splines,
                bin_size=bin_size,
                edge_sigma=edge_sigma,
                max_shift=max_shift,
            )
        for _ in range(n_refine):
            parent.refine_splines(
                splines=splines,
                max_interval=max(interval, 30),
                bin_size=bin_size,
            )
        parent.measure_radius(splines=splines, bin_size=bin_size)
        if local_props:
            parent.local_ft_analysis(
                splines=splines, interval=interval, ft_size=ft_size, bin_size=bin_size
            )
        if infer_polarity:
            parent.auto_align_to_polarity()
        if global_props:
            parent.global_ft_analysis(splines=splines, bin_size=bin_size)
        if local_props and paint:
            parent.paint_cylinders()
        if map_monomers:
            parent.map_monomers(orientation=GVar.clockwise)
        parent._current_ft_size = ft_size
        return None


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
    filter_reference_image : ImageFilter
        Choose filter for the reference image (does not affect image data itself).
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
        scale_value = vfield(1.0).with_options(min=1e-3, step=1e-4, max=10.0)
        read_header = abstractapi()

    @magicclass(layout="horizontal", labels=False)
    class tilt_range(MagicTemplate):
        """Tilt range of the tomogram."""

        tilt_range_label = vfield("tilt range (deg)", widget_type="Label")
        range = vfield(Optional[tuple[float, float]]).with_options(
            text="No missing wedge",
            options=dict(options=dict(min=-90, max=90, step=1)),
            value=(-60, 60),
        )

    bin_size = vfield([1]).with_options(options={"min": 1, "max": 32})
    filter = vfield(Union[ImageFilter, None]).with_options(
        value=ImageFilter.Lowpass, label="filter"
    )

    @scale.wraps
    @set_design(max_width=90)
    def read_header(self):
        """Read scale from image header."""
        path = self.path
        if not os.path.exists(path) or not os.path.isfile(path):
            return
        img = ip.lazy_imread(path, chunks=GVar.dask_chunk)
        scale = img.scale.x
        self.scale.scale_value = f"{scale:.4f}"
        if len(self.bin_size) < 2:
            self.bin_size = [ceilint(0.96 / scale)]

    @set_design(text="Preview")
    def preview_image(self):
        """Preview image at the path."""
        return view_image(self.path, parent=self)

    open_image = abstractapi()


@magicclass(name="General info", record=False, widget_type="collapsible", labels=False)
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

    def _refer_tomogram(self, tomo: CylTomogram):
        img = tomo.image
        source = tomo.metadata.get("source", "Unknown")
        scale = tomo.scale
        shape_px = ", ".join(f"{s} px" for s in img.shape)
        shape_nm = ", ".join(f"{s*scale:.2f} nm" for s in img.shape)
        if tomo.tilt_range is not None:
            deg0, deg1 = tomo.tilt_range
            tilt_range = f"{deg0:.1f}° — {deg1:.1f}°"
        else:
            tilt_range = "No missing wedge"
        value = (
            f"File: {source}\n"
            f"Scale: {scale:.4f} nm/pixel\n"
            f"ZYX-Shape: ({shape_px}), ({shape_nm})\n"
            f"Tilt range: {tilt_range}"
        )
        self.image_info.value = value

    def _refer_project(self, project: CylindraProject):
        self.project_desc.value = project.project_description

    label0 = field("\n<b>Image information</b>", widget_type="Label")
    image_info = field(widget_type=ConsoleTextEdit)
    label1 = field("\n<b>Project description</b>", widget_type="Label")
    project_desc = field(widget_type=TextEdit)
