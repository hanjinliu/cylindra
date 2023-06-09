import os
from typing import Annotated, Sequence, Union
from magicgui.widgets import TextEdit
from magicclass import (
    do_not_record,
    get_function_gui,
    magicclass,
    magicmenu,
    field,
    vfield,
    MagicTemplate,
    set_design,
    bind_key,
    abstractapi,
)
from magicclass.widgets import Separator, ConsoleTextEdit
from magicclass.types import Optional, Path
from magicclass.logging import getLogger
from magicclass.ext.polars import DataFrameView
import impy as ip

from cylindra.widgets.widget_utils import FileFilter
from cylindra.widgets._previews import view_image
from cylindra.widgets._widget_ext import CheckBoxes

from cylindra._custom_layers import MoleculesLayer
from cylindra.utils import ceilint, roundint
from cylindra.types import get_monomer_layers
from cylindra.ext.etomo import PEET
from cylindra.components import CylTomogram
from cylindra.const import GlobalVariables as GVar, nm, ImageFilter, get_versions
from cylindra.project import CylindraProject
from .global_variables import GlobalVariablesMenu

_Logger = getLogger("cylindra")


class ChildWidget(MagicTemplate):
    def _get_main(self):
        from cylindra.widgets import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget)


@magicmenu
class File(ChildWidget):
    """File input and output."""

    @set_design(text="Open image")
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+O")
    def open_image_loader(self):
        """Load an image file and process it before sending it to the viewer."""
        return self._get_main()._image_loader.show()

    load_project = abstractapi()
    load_splines = abstractapi()
    load_molecules = abstractapi()
    sep0 = field(Separator)
    save_project = abstractapi()
    overwrite_project = abstractapi()
    save_spline = abstractapi()
    save_molecules = abstractapi()
    sep1 = field(Separator)

    @set_design(text="Process images")
    @do_not_record
    def open_image_processor(self):
        """Open image processor."""
        return self._get_main().image_processor.show()

    @set_design(text="View project")
    @do_not_record
    def view_project(self, path: Path.Read[FileFilter.JSON]):
        main = self._get_main()
        pviewer = CylindraProject.from_json(path).make_project_viewer()
        pviewer.native.setParent(main.native, pviewer.native.windowFlags())
        main._active_widgets.add(pviewer)
        return pviewer.show()

    PEET = PEET


@magicmenu(name="Image")
class Image(ChildWidget):
    """Image processing and visualization"""

    filter_reference_image = abstractapi()
    add_multiscale = abstractapi()
    set_multiscale = abstractapi()
    sep0 = field(Separator)

    @do_not_record
    @set_design(text="Open spline slicer")
    def open_slicer(self):
        """Open spline slicer widget"""
        main = self._get_main()
        main.spline_slicer.show()
        return main.spline_slicer.refresh_widget_state()

    @set_design(text="Simulate cylindric structure")
    @do_not_record
    @bind_key("Ctrl+K, I")
    def open_simulator(self):
        """Open the simulator widget."""
        return self._get_main().cylinder_simulator.show()

    sep1 = field(Separator)
    sample_subtomograms = abstractapi()
    paint_cylinders = abstractapi()
    backpaint_molecule_density = abstractapi()
    show_colorbar = abstractapi()


@magicmenu
class Splines(ChildWidget):
    """Operations on splines"""

    show_splines = abstractapi()
    show_splines_as_meshes = abstractapi()

    @set_design(text="Show local properties")
    @do_not_record
    def show_localprops(self):
        """Show spline local properties in a table widget."""
        from magicgui.widgets import Container, ComboBox

        main = self._get_main()
        cbox = ComboBox(choices=main._get_splines)
        table = DataFrameView(value={})

        @cbox.changed.connect
        def _update_table(i: int):
            if i is not None:
                spl = main.tomogram.splines[i]
                table.value = spl.localprops

        container = Container(widgets=[cbox, table], labels=False)
        self.parent_viewer.window.add_dock_widget(
            container, area="left", name="Molecule Features"
        ).setFloating(True)
        cbox.changed.emit(cbox.value)
        return None

    add_anchors = abstractapi()
    sep0 = field(Separator)

    @magicmenu
    class Orientation(MagicTemplate):
        """Adjust spline orientation."""

        invert_spline = abstractapi()
        align_to_polarity = abstractapi()
        auto_align_to_polarity = abstractapi()

    clip_spline = abstractapi()

    @set_design(text="Open spline clipper")
    @do_not_record
    def open_spline_clipper(self):
        """Open the spline clipper widget to precisely clip spines."""
        main = self._get_main()
        main.spline_clipper.show()
        if main.tomogram.n_splines > 0:
            main.spline_clipper.load_spline(main.SplineControl.num)
        return None

    delete_spline = abstractapi()
    sep1 = field(Separator)
    fit_splines = abstractapi()

    @set_design(text="Fit splines manually")
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+/")
    def fit_splines_manually(
        self, max_interval: Annotated[nm, {"label": "Max interval (nm)"}] = 50.0
    ):
        """
        Open a spline fitter window and fit cylinder with spline manually.

        Parameters
        ----------
        max_interval : nm, default is 50.0
            Maximum interval (nm) between spline anchors that will be used to
            sample subtomogram projections.
        """
        main = self._get_main()
        main.spline_fitter._load_parent_state(max_interval=max_interval)
        return main.spline_fitter.show()

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

        @set_design(text="Show molecule features")
        @do_not_record
        def show_molecule_features(self):
            """Show molecules features in a table widget."""
            from magicgui.widgets import Container, ComboBox

            cbox = ComboBox(choices=get_monomer_layers)
            table = DataFrameView(value={})

            @cbox.changed.connect
            def _update_table(layer: MoleculesLayer):
                if layer is not None:
                    table.value = layer.features

            container = Container(widgets=[cbox, table], labels=False)
            self.parent_viewer.window.add_dock_widget(
                container, area="left", name="Molecule Features"
            ).setFloating(True)
            cbox.changed.emit(cbox.value)
            return None

        paint_molecules = abstractapi()
        plot_molecule_feature = abstractapi()


@magicmenu
class Analysis(ChildWidget):
    """Analysis of tomograms."""

    measure_radius = abstractapi()
    measure_local_radius = abstractapi()
    measure_radius_by_molecules = abstractapi()
    sep0 = field(Separator)
    local_ft_analysis = abstractapi()
    global_ft_analysis = abstractapi()
    sep1 = field(Separator)
    reanalyze_image = abstractapi()
    load_project_for_reanalysis = abstractapi()
    sep2 = field(Separator)

    @set_design(text="Open spectra measurer")
    @do_not_record
    def open_spectra_measurer(self):
        """Open the spectra measurer widget to determine cylindric parameters."""
        main = self._get_main()
        if main.tomogram is not None and main.tomogram.n_splines > 0:
            binsize = roundint(main._layer_image.scale[0] / main.tomogram.scale)
            main.spectra_measurer.load_spline(main.SplineControl.num, binsize)
        return main.spectra_measurer.show()

    @set_design(text="Open subtomogram analyzer")
    @do_not_record
    @bind_key("Ctrl+K, S")
    def open_subtomogram_analyzer(self):
        """Open the subtomogram analyzer dock widget."""
        return self._get_main().sta.show()

    @set_design(text="Open batch analyzer")
    @do_not_record
    @bind_key("Ctrl+K, B")
    def open_project_batch_analyzer(self):
        """Open the batch analyzer widget."""
        from cylindra.widgets.batch import CylindraBatchWidget

        main = self._get_main()
        uibatch = CylindraBatchWidget()
        uibatch.native.setParent(main.native, uibatch.native.windowFlags())
        main._batch = uibatch
        uibatch.show()
        main._active_widgets.add(uibatch)
        return uibatch

    sep2 = field(Separator)

    @set_design(text="Repeat command")
    @do_not_record(recursive=False)
    @bind_key("Ctrl+Shift+R")
    def repeat_command(self):
        """Repeat the last command."""
        return self.macro.repeat_method(same_args=False, raise_parse_error=False)


@magicmenu
class Others(ChildWidget):
    """Other menus."""

    @magicmenu
    class Macro(ChildWidget):
        @set_design(text="Show macro")
        @do_not_record
        @bind_key("Ctrl+Shift+M")
        def show_macro(self):
            """Create Python executable script of the current project."""
            main = self._get_main()
            new = main.macro.widget.new_window()
            new.textedit.value = str(main._format_macro()[main._macro_offset :])
            new.show()
            main._active_widgets.add(new)
            return None

        @set_design(text="Show full macro")
        @do_not_record
        def show_full_macro(self):
            """Create Python executable script since the startup this time."""
            main = self._get_main()
            new = main.macro.widget.new_window()
            new.textedit.value = str(main._format_macro())
            new.show()
            main._active_widgets.add(new)
            return None

        @set_design(text="Show native macro")
        @do_not_record
        def show_native_macro(self):
            """
            Show the native macro widget of magic-class, which is always synchronized but
            is not editable.
            """
            main = self._get_main()
            main.macro.widget.show()
            main._active_widgets.add(main.macro.widget)
            return None

        sep0 = field(Separator)
        load_macro_file = abstractapi()

    run_workflow = abstractapi()
    define_workflow = abstractapi()

    sep0 = field(Separator)

    @set_design(text="Open command palette")
    @do_not_record
    @bind_key("Ctrl+P")
    def open_command_palette(self):
        from magicclass.command_palette import exec_command_palette

        return exec_command_palette(self._get_main(), alignment="screen")

    GlobalVariables = GlobalVariablesMenu

    @set_design(text="Open logger")
    @do_not_record
    def open_logger(self):
        """Open logger window."""
        wdt = _Logger.widget
        name = "Log"
        if name in self.parent_viewer.window._dock_widgets:
            self.parent_viewer.window._dock_widgets[name].show()
        else:
            self.parent_viewer.window.add_dock_widget(wdt, name=name)

    @magicmenu
    class Help(MagicTemplate):
        open_help = abstractapi()
        cylindra_info = abstractapi()
        report_issues = abstractapi()

    @Help.wraps
    @set_design(text="Open help")
    @do_not_record
    def open_help(self):
        """Open a help window."""
        from magicclass import build_help

        help = build_help(self._get_main())
        return help.show()

    @Help.wraps
    @set_design(text="Info")
    @do_not_record
    def cylindra_info(self):
        """Show information of dependencies."""

        main = self._get_main()
        versions = get_versions()
        value = "\n".join(f"{k}: {v}" for k, v in versions.items())
        w = ConsoleTextEdit(value=value)
        w.read_only = True
        w.native.setParent(main.native, w.native.windowFlags())
        w.show()
        main._active_widgets.add(w)
        return None

    @Help.wraps
    @set_design(text="Report issues")
    @do_not_record
    def report_issues(self):
        """Report issues on GitHub."""
        from magicclass.utils import open_url

        return open_url("https://github.com/hanjinliu/cylindra/issues/new")


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
    max_shift = vfield(5.0, label="Max shift (nm)").with_options(max=50.0, step=0.5)


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
    ft_size = vfield(32.0, label="FT window size (nm)").with_options(min=1.0, max=200.0)
    paint = vfield(True)


@magicclass(name="_Run cylindrical fitting", record=False)
class Runner(MagicTemplate):
    """
    Attributes
    ----------
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
        from cylindra.widgets.main import CylindraMainWidget

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

    splines = vfield(widget_type=CheckBoxes).with_choices(_get_splines)
    bin_size = vfield(int).with_choices(choices=_get_available_binsize)

    fit = vfield(True, label="Fit splines")
    params1 = runner_params1
    n_refine = vfield(1, label="Refinement iteration").with_options(max=10)
    local_props = vfield(True, label="Calculate local properties")
    params2 = runner_params2
    global_props = vfield(True, label="Calculate global properties")
    infer_polarity = vfield(True, label="Infer polarity")
    map_monomers = vfield(True, label="Map monomers")

    @fit.connect
    def _toggle_fit_params(self, visible: bool):
        self.params1.visible = visible

    @local_props.connect
    def _toggle_localprops_params(self, visible: bool):
        self.params2.visible = visible

    def _get_max_shift(self, w=None):
        if self.fit:
            return self.params1.max_shift
        else:
            return -1.0

    @set_design(text="Run")
    @do_not_record(recursive=False)
    def run(
        self,
        splines: Annotated[Sequence[int], {"bind": splines}] = (),
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
        if parent._layer_work.data.size > 0:
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
                splines=splines, interval=interval, depth=ft_size, bin_size=bin_size
            )
        if infer_polarity:
            parent.auto_align_to_polarity(bin_size=bin_size)
        if global_props:
            parent.global_ft_analysis(splines=splines, bin_size=bin_size)
        if local_props and paint:
            limits = get_function_gui(parent.paint_cylinders).limits.value
            parent.paint_cylinders(limits=limits)
        if map_monomers:
            parent.map_monomers(orientation=GVar.clockwise)
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
    filter : ImageFilter
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
    filter = vfield(Union[ImageFilter, None]).with_options(value=ImageFilter.DoG)
    eager = vfield(False, label="Load the entire image into memory")

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
