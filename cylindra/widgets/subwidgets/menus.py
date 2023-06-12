from typing import Annotated, TYPE_CHECKING

from macrokit import Head, parse, Symbol
from macrokit.utils import check_call_args, check_attributes

from magicclass import (
    do_not_record,
    magicmenu,
    field,
    nogui,
    setup_function_gui,
    MagicTemplate,
    set_options,
    set_design,
    bind_key,
    abstractapi,
)
from magicclass.widgets import Separator, ConsoleTextEdit, CodeEdit
from magicclass.types import Path
from magicclass.logging import getLogger
from magicclass.ext.polars import DataFrameView

from cylindra.widgets.widget_utils import FileFilter

from cylindra._custom_layers import MoleculesLayer
from cylindra.utils import roundint
from cylindra.types import get_monomer_layers
from cylindra.ext.etomo import PEET
from cylindra.const import nm, get_versions
from cylindra.project import CylindraProject
from cylindra import _config
from .global_variables import GlobalVariablesMenu

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget
    from magicgui.widgets import FunctionGui

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
    set_source_spline = abstractapi()
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

    @magicmenu(record=False)
    class Macro(ChildWidget):
        @set_design(text="Show macro")
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
        def show_full_macro(self):
            """Create Python executable script since the startup this time."""
            main = self._get_main()
            new = main.macro.widget.new_window()
            new.textedit.value = str(main._format_macro())
            new.show()
            main._active_widgets.add(new)
            return None

        @set_design(text="Show native macro")
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

        @set_design(text="Load Python file")
        @do_not_record
        def load_macro_file(self, path: Path.Read[FileFilter.PY]):
            """Load a Python script file to a new macro window."""
            main = self._get_main()
            return main.load_macro_file(path)

    @magicmenu(record=False)
    class Workflows(ChildWidget):
        def _get_workflow_names(self, *_) -> list[str]:
            return [file.stem for file in _config.WORKFLOWS_DIR.glob("*.py")]

        @set_design(text="Run workflow")
        @bind_key("Ctrl+K, Ctrl+Shift+R")
        @set_options(labels=False)
        def run_workflow(
            self,
            filename: Annotated[str, {"choices": _get_workflow_names}],
        ):
            main = self._get_main()
            return main.run_workflow(filename)

        @nogui
        def append_workflow(self, path: Path):
            main = self._get_main()

            @set_design(text=f"Run `{path.stem}`")
            @do_not_record
            def run():
                return main.run_workflow(path)

            return self.append(run)

        @set_design(text="Define workflow")
        @bind_key("Ctrl+K, Ctrl+Shift+D")
        def define_workflow(
            self,
            filename: str,
            workflow: Annotated[str, {"widget_type": CodeEdit}],
        ):
            """Define a workflow script for the daily analysis."""
            code = normalize_workflow(workflow, self._get_main())
            path = _config.workflow_path(filename)
            path.write_text(code)
            _Logger.print("Workflow saved: " + path.as_posix())
            return None

        @set_design(text="View/Edit workflow")
        @set_options(call_button="Overwrite", labels=False)
        def edit_workflow(
            self,
            filename: Annotated[str, {"choices": _get_workflow_names}],
            workflow: Annotated[str, {"widget_type": CodeEdit}],
        ):
            """View or edit a workflow script."""
            return self.define_workflow(filename, workflow)

        sep0 = field(Separator)

    sep0 = field(Separator)

    @set_design(text="Command palette")
    @do_not_record
    @bind_key("Ctrl+P")
    def open_command_palette(self):
        """Open the command palette widget."""
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


def normalize_workflow(workflow: str, ui: "CylindraMainWidget") -> str:
    expr = parse(workflow)
    # TODO: fix macro-kit to check this
    # check_call_args(expr, {"ui": ui})
    check_attributes(expr, {"ui": ui})
    _main_function_found = False
    for line in expr.args:
        if isinstance(line, Symbol):
            continue
        if line.head is Head.function and line.args[0].args[0].name == "main":
            _main_function_found = True
            break

    if not _main_function_found:
        raise ValueError("No main function found in workflow script.")
    return workflow


@setup_function_gui(Others.Workflows.run_workflow)
def _(self: Others.Workflows, gui: "FunctionGui"):
    txt = CodeEdit()
    txt.syntax_highlight("python")
    txt.read_only = True
    gui.insert(1, txt)

    @gui.filename.changed.connect
    def _on_name_change(filename: str):
        txt.value = _config.workflow_path(filename).read_text()

    _on_name_change(gui.filename.value)


@setup_function_gui(Others.Workflows.define_workflow)
def _(self: Others.Workflows, gui: "FunctionGui"):
    gui.workflow.syntax_highlight("python")
    gui.workflow.value = "\n".join(
        [
            "import numpy as np",
            "import impy as ip",
            "import polars as pl",
            "from pathlib import Path",
            "from cylindra.widgets import CylindraMainWidget",
            "",
            "def main(ui: 'CylindraMainWidget'):",
            "    # Write your workflow here",
            "",
        ]
    )
    gui.called.connect(self.reset_choices)


@setup_function_gui(Others.Workflows.edit_workflow)
def _(self: Others.Workflows, gui: "FunctionGui"):
    gui.workflow.syntax_highlight("python")

    @gui.filename.changed.connect
    def _on_name_change(filename: str):
        gui.workflow.value = _config.workflow_path(filename).read_text()

    _on_name_change(gui.filename.value)
