from functools import partial
import inspect
from typing import Annotated, TYPE_CHECKING, Literal
from datetime import datetime
import shutil

import numpy as np
from macrokit import Head, parse, Symbol
from macrokit.utils import check_call_args, check_attributes
import matplotlib.pyplot as plt
from acryo import pipe

from magicclass import (
    do_not_record,
    get_function_gui,
    magicmenu,
    field,
    nogui,
    setup_function_gui,
    MagicTemplate,
    set_options,
    set_design,
    bind_key,
    confirm,
    abstractapi,
)
from magicclass.widgets import Separator, ConsoleTextEdit, CodeEdit, OneLineRunner
from magicclass.types import Path, Color, Optional
from magicclass.logging import getLogger
from magicclass.utils import thread_worker
from magicclass.ext.polars import DataFrameView

from cylindra._custom_layers import MoleculesLayer
from cylindra.utils import roundint
from cylindra.types import get_monomer_layers, ColoredLayer
from cylindra.ext.etomo import PEET
from cylindra.const import nm, get_versions, ImageFilter
from cylindra.project import CylindraProject, extract
from cylindra.widgets.widget_utils import FileFilter, get_code_theme
from cylindra.widgets._widget_ext import CheckBoxes
from cylindra.components.spline import SplineConfig
from cylindra import _config

from ._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.widgets import CylindraMainWidget
    from magicgui.widgets import FunctionGui
    from magicclass._gui._macro import MacroEdit

_Logger = getLogger("cylindra")


@magicmenu
class File(ChildWidget):
    """File input and output."""

    @set_design(text="Open image")
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+O")
    def open_image_loader(self):
        """Load an image file and process it before sending it to the viewer."""
        loader = self._get_main()._image_loader
        loader.show()
        return loader

    load_project = abstractapi()
    load_splines = abstractapi()
    load_molecules = abstractapi()
    sep0 = field(Separator)
    save_project = abstractapi()
    overwrite_project = abstractapi()
    save_spline = abstractapi()
    save_molecules = abstractapi()
    sep1 = field(Separator)

    @magicmenu(record=False)
    class Stash(ChildWidget):
        """Stashing projects for later use."""

        def _get_stashed_names(self, w=None) -> list[str]:
            return _config.get_stash_list()

        def _need_save(self):
            return self._get_main()._need_save

        @set_design(text="Stash current project")
        def stash_project(self):
            """
            Stash current project in the user directory.

            This method simply saves the current project in the user directory. Stashing project
            is useful when you want to temporarily save the current project for later use, without
            thinking of the name, where to put it etc. Stashed projects can be easily loaded or
            cleared using other methods in this menu.
            """
            root = _config.get_stash_dir()
            path = root / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            return self._get_main().save_project(path)

        @set_design(text="Load stashed project")
        @confirm(text="You may have unsaved data. Open a new project?", condition=_need_save)  # fmt: skip
        @thread_worker.with_progress(text="Loading stashed project...")
        def load_stash_project(
            self,
            name: Annotated[str, {"choices": _get_stashed_names}],
            filter: ImageFilter | None = ImageFilter.LoG,
        ):
            """
            Load a stashed project.

            Parameters
            ----------
            name : str
                Name of the stashed project.
            filter : ImageFilter, default is ImageFilter.DoG
                Image filter to apply to the loaded images.
            """
            yield from self._get_main().load_project.arun(
                _config.get_stash_dir() / name, filter=filter
            )

        @set_design(text="Pop stashed project")
        @confirm(text="You may have unsaved data. Open a new project?", condition=_need_save)  # fmt: skip
        def pop_stash_project(
            self,
            name: Annotated[str, {"choices": _get_stashed_names}],
            filter: ImageFilter | None = ImageFilter.DoG,
        ):
            """
            Load a stashed project and delete it from the stash list.

            Parameters
            ----------
            name : str
                Name of the stashed project.
            filter : ImageFilter, default is ImageFilter.DoG
                Image filter to apply to the loaded images.
            """
            self.load_stash_project(name, filter=filter)
            return self.delete_stash_project(name)

        @set_design(text="Delete stashed project")
        def delete_stash_project(
            self, name: Annotated[str, {"choices": _get_stashed_names}]
        ):
            """
            Delete a stashed project.

            Parameters
            ----------
            name : str
                Name of the stashed project to be deleted.
            """
            path = _config.get_stash_dir() / name
            shutil.rmtree(path)
            self.reset_choices()
            return None

        @set_design(text="Clear stashed projects")
        def clear_stash_projects(self):
            """Clear all the stashed projects."""
            for name in self._get_stashed_names():
                self.delete_stash_project(name)
            return None

    @set_design(text="Open file iterator")
    @do_not_record
    def open_file_iterator(self):
        """Open image processor widget."""
        return self._get_main()._file_iterator.show()

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
    @bind_key("Ctrl+K, /")
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

    @set_design(text="Show colorbar")
    @do_not_record
    def show_colorbar(
        self,
        layer: ColoredLayer,
        length: Annotated[int, {"min": 16}] = 256,
        orientation: Literal["vertical", "horizontal"] = "horizontal",
    ):
        """
        Show the colorbar of the molecules or painted cylinder in the logger.

        Parameters
        ----------
        layer : Layer
            The layer to show the colorbar of.
        length : int, default is 256
            Length of the colorbar.
        orientation : 'vertical' or 'horizontal', default is 'horizontal'
            Orientation of the colorbar.
        """
        info = layer.colormap_info
        colors = info.cmap.map(np.linspace(0, 1, length))
        cmap_arr = np.stack([colors] * (length // 12), axis=0)
        xmin, xmax = info.clim
        with _Logger.set_plt():
            if orientation == "vertical":
                plt.imshow(np.swapaxes(cmap_arr, 0, 1)[::-1])
                plt.xticks([], [])
                plt.yticks([0, length - 1], [f"{xmax:.2f}", f"{xmin:.2f}"])
            else:
                plt.imshow(cmap_arr)
                plt.xticks([0, length - 1], [f"{xmin:.2f}", f"{xmax:.2f}"])
                plt.yticks([], [])
            plt.tight_layout()
            plt.show()
        return None


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
        if len(main.tomogram.splines) > 0:
            main.spline_clipper.load_spline(main.SplineControl.num)
        return None

    delete_spline = abstractapi()
    copy_spline = abstractapi()
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

    sep3 = field(Separator)

    @magicmenu(record=False)
    class Config(ChildWidget):
        def _get_saved_config_files(self, w=None) -> list[str]:
            return [path.stem for path in _config.get_config().list_config_paths()]

        def _get_splines(self, w=None) -> list[tuple[str, int]]:
            return self._get_main()._get_splines()

        @set_design(text="Update default config")
        @bind_key("Ctrl+K, Ctrl+[")
        def update_default_config(
            self,
            npf_range: Annotated[tuple[int, int], {"options": {"min": 2, "max": 100}}] = (11, 17),
            spacing_range: Annotated[tuple[nm, nm], {"options": {"step": 0.05}}] = (3.9, 4.3),
            skew_range: Annotated[tuple[float, float], {"options": {"min": -45.0, "max": 45.0, "step": 0.05}}] = (-1.0, 1.0),
            rise_range: Annotated[tuple[float, float], {"options": {"min": -45.0, "max": 45.0, "step": 0.1}}] = (0.0, 45.0),
            rise_sign: Literal[-1, 1] = -1,
            clockwise: Literal["PlusToMinus", "MinusToPlus"] = "MinusToPlus",
            thickness_inner: Annotated[nm, {"min": 0.0, "step": 0.1}] = 2.0,
            thickness_outer: Annotated[nm, {"min": 0.0, "step": 0.1}] = 3.0,
            fit_depth: Annotated[nm, {"min": 4.0, "step": 1}] = 48.0,
            fit_width: Annotated[nm, {"min": 4.0, "step": 1}] = 44.0,
        ):  # fmt: skip
            """
            Update the default spline config.

            Parameters
            ----------
            npf_range : (int, int), default is (11, 17)
                Range of protofilament number.
            spacing_range : (float, float), default is (3.9, 4.3)
                Range of longitudinal lattice spacing.
            skew_range : (float, float), default is (-1.0, 1.0)
                Range of skew angle in degree.
            rise_range : (float, float), default is (0.0, 45.0)
                Range of rise angle in degree.
            rise_sign : -1 or 1, default is -1
                Sign of the rise angle.
            clockwise : "PlusToMinus" or "MinusToPlus", default is "MinusToPlus"
                Closewise rotation of the y-project corresponds to which orientation.
            thickness_inner : float, default is 2.0
                Cylinder thickness inside the radial peak.
            thickness_outer : float, default is 3.0
                Cylinder thickness outside the radial peak.
            fit_depth : float, default is 48.0
                Depth in nm used during spline fitting.
            fit_width : float, default is 44.0
                Width in nm used during spline fitting.
            """
            loc = locals()
            del loc["self"]
            self._get_main().default_config = SplineConfig(**loc)
            return None

        sep0 = field(Separator)

        @set_design(text="Load default config")
        def load_default_config(
            self, name: Annotated[str, {"choices": _get_saved_config_files}]
        ):
            """Load a saved config file as the default config."""
            path = _config.get_config().spline_config_path(name)
            self._get_main().default_config = SplineConfig.from_file(path)
            return None

        @set_design(text="Save default config")
        def save_default_config(self, name: str):
            """Save current default config."""
            path = _config.get_config().spline_config_path(name)
            if path.exists():
                raise FileExistsError(f"Config file {path} already exists.")
            return self._get_main().default_config.to_file(path)


@magicmenu(name="Molecules")
class MoleculesMenu(ChildWidget):
    """Operations on molecules"""

    @magicmenu
    class Mapping(MagicTemplate):
        """Map monomers along splines in several ways."""

        map_monomers = abstractapi()
        map_monomers_with_extensions = abstractapi()
        map_centers = abstractapi()
        map_along_pf = abstractapi()

    set_source_spline = abstractapi()
    translate_molecules = abstractapi()
    rotate_molecules = abstractapi()

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
        interpolate_spline_properties = abstractapi()
        calculate_lattice_structure = abstractapi()
        sep2 = field(Separator)
        convolve_feature = abstractapi()
        binarize_feature = abstractapi()
        label_feature_clusters = abstractapi()
        regionprops_features = abstractapi()

    @magicmenu(name="View")
    class View(ChildWidget):
        """Visualize molecule features."""

        @set_design(text="Show orientation")
        @do_not_record
        def show_orientation(
            self,
            layer: MoleculesLayer,
            x_color: Color = "orange",
            y_color: Color = "cyan",
            z_color: Color = "crimson",
        ):
            """
            Show molecule orientations with a vectors layer.

            Parameters
            ----------
            layer : MolecularLayer
                The layer to show the orientation of.
            x_color : Color, default is "crimson"
                Vector color of the x direction.
            y_color : Color, default is "cyan"
                Vector color of the y direction.
            z_color : Color, default is "orange"
                Vector color of the z direction.
            """
            main = self._get_main()
            mol = layer.molecules
            nmol = len(mol)
            name = f"Axes of {layer.name}"

            zvec = np.stack([mol.pos, mol.z], axis=1)
            yvec = np.stack([mol.pos, mol.y], axis=1)
            xvec = np.stack([mol.pos, mol.x], axis=1)

            vector_data = np.concatenate([zvec, yvec, xvec], axis=0)

            layer = main.parent_viewer.add_vectors(
                vector_data,
                edge_width=0.3,
                edge_color="direction",
                edge_color_cycle=[z_color, y_color, x_color],
                features={"direction": ["z"] * nmol + ["y"] * nmol + ["x"] * nmol},
                length=_config.get_config().point_size * 0.8,
                name=name,
                vector_style="arrow",
            )
            return main._undo_callback_for_layer(layer)

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

        @set_design(text="Render molecules")
        @do_not_record
        def render_molecules(
            self,
            layer: MoleculesLayer,
            template_path: Path.Read[FileFilter.IMAGE],
            scale: Annotated[nm, {"min": 0.1, "max": 10.0}] = 1.5,
        ):
            """
            Render molecules using the template image.

            This method is only for visualization purpose. Iso-surface will be calculated
            using the input template image and mapped to every molecule position. Surfaces
            will be colored as the input molecules layer.

            Parameters
            ----------
            layer : MoleculesLayer
                The layer used to render.
            template_path : Path
                Path to the template image.
            scale : nm, default is 1.5
                Scale to resize the template image.
            """
            from skimage.measure import marching_cubes
            from skimage.filters import threshold_yen

            template = pipe.from_file(template_path).provide(scale)
            mole = layer.molecules
            nmol = len(mole)

            # create surface
            verts, faces, _, _ = marching_cubes(
                template,
                level=threshold_yen(template),
                step_size=1,
                spacing=(scale,) * 3,
            )

            nverts = verts.shape[0]
            all_verts = np.empty((nmol * nverts, 3), dtype=np.float32)
            all_faces = np.concatenate(
                [faces + i * nverts for i in range(nmol)], axis=0
            )
            center = np.array(template.shape) / 2 + 0.5

            for i, v in enumerate(verts):
                v_transformed = mole.rotator.apply(v - center * scale) + mole.pos
                all_verts[i::nverts] = v_transformed

            if layer.colormap_info is None:
                colormap = None
                contrast_limits = None
                data = (all_verts, all_faces)
            else:
                color_by = layer.colormap_info.name
                all_values = np.stack(
                    [layer.features[color_by]] * nverts, axis=1
                ).ravel()
                colormap = layer.colormap_info.cmap
                contrast_limits = layer.colormap_info.clim
                data = (all_verts, all_faces, all_values)

            self.parent_viewer.add_surface(
                data=data,
                colormap=colormap,
                contrast_limits=contrast_limits,
                shading="smooth",
                name=f"Rendered {layer.name}",
            )
            return None


@magicmenu
class Analysis(ChildWidget):
    """Analysis of tomograms."""

    measure_radius = abstractapi()
    set_radius = abstractapi()
    measure_local_radius = abstractapi()
    measure_radius_by_molecules = abstractapi()
    sep0 = field(Separator)
    local_ft_analysis = abstractapi()
    global_ft_analysis = abstractapi()
    sep1 = field(Separator)
    reanalyze_image = abstractapi()
    load_project_for_reanalysis = abstractapi()
    sep2 = field(Separator)

    @set_design(text="Open spectra inspector")
    @do_not_record
    def open_spectra_inspector(self):
        """Open the spectra measurer widget to determine cylindric parameters."""
        main = self._get_main()
        if len(main.tomogram.splines) > 0:
            binsize = roundint(main._reserved_layers.scale / main.tomogram.scale)
            main.spectra_inspector.load_spline(main.SplineControl.num, binsize)
        return main.spectra_inspector.show()

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
        def __init__(self):
            self._macro_window: "MacroEdit | None" = None

        def _get_macro_window(
            self, text: str = "", tabname: "str | None" = None
        ) -> "MacroEdit":
            if self._macro_window is None or not self._macro_window.visible:
                main = self._get_main()
                new = main.macro.widget.new_window(tabname=tabname)
                self._macro_window = new
                main._active_widgets.add(new)
                new.textedit.value = text
            else:
                new = self._macro_window
                new.new_tab(name=tabname, text=text)
            new.textedit.syntax_highlight(lang="python", theme=get_code_theme(self))
            return new

        @set_design(text="Show macro")
        @bind_key("Ctrl+Shift+M")
        def show_macro(self):
            """Create Python executable script of the current project."""
            main = self._get_main()
            text = str(main._format_macro()[main._macro_offset :])
            return self._get_macro_window(text).show()

        @set_design(text="Show full macro")
        def show_full_macro(self):
            """Create Python executable script since the startup this time."""
            main = self._get_main()
            text = str(main._format_macro())
            return self._get_macro_window(text, tabname="Full macro").show()

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

            edit = main.macro.widget.new_window(path.name)
            edit.textedit.value = str(extract(Path(path).read_text()))
            main._active_widgets.add(edit)
            return None

    @magicmenu(record=False)
    class Workflows(ChildWidget):
        def _get_workflow_names(self, *_) -> list[str]:
            return [file.stem for file in _config.WORKFLOWS_DIR.glob("*.py")]

        def _make_method_name(self, path: Path) -> str:
            abs_path = _config.workflow_path(path)
            return f"Run_{hex(hash(abs_path))}"

        @set_design(text="Run workflow")
        @bind_key("Ctrl+K, Ctrl+Shift+R")
        @set_options(labels=False)
        def run_workflow(
            self,
            filename: Annotated[str, {"choices": _get_workflow_names}],
        ):
            # close this magicgui before running whole workflow
            get_function_gui(self.run_workflow).close()
            fname = self._make_method_name(filename)
            self[fname].changed()

        @nogui
        def append_workflow(self, path: Path):
            """Append workflow as a widget to the menu."""
            main = self._get_main()
            main_func = _config.get_main_function(path)
            partial_func = partial(main_func, main)
            prms = list(inspect.signature(main_func).parameters.values())[1:]
            partial_func.__signature__ = inspect.Signature(prms)

            fn = set_design(text=f"Run `{path.stem}`")(do_not_record(partial_func))
            fn.__name__ = self._make_method_name(path)
            # Old menu should be removed
            try:
                del self[fn.__name__]
            except (IndexError, KeyError):
                pass
            return self.append(fn)

        @set_options(call_button="Save workflow")
        @set_design(text="Define workflow")
        @bind_key("Ctrl+K, Ctrl+Shift+D")
        def define_workflow(
            self,
            filename: str,
            workflow: Annotated[str, {"widget_type": CodeEdit}],
        ):
            """Define a workflow script for the daily analysis."""
            if filename == "":
                raise ValueError("Filename must be specified.")
            code = normalize_workflow(workflow, self._get_main())
            path = _config.workflow_path(filename)
            if path.exists():
                old_text: str | None = path.read_text()
            else:
                old_text = None
            path.write_text(code, encoding="utf-8")
            try:
                self.append_workflow(path)
            except Exception as e:
                if old_text:
                    path.write_text(old_text, encoding="utf-8")
                else:
                    path.unlink(missing_ok=True)
                raise e
            _Logger.print("Workflow saved: " + path.as_posix())
            self.reset_choices()
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

        @set_design(text="Delete workflow")
        @set_options(call_button="Delete", labels=False)
        def delete_workflow(
            self,
            filenames: Annotated[
                list[str],
                {
                    "choices": _get_workflow_names,
                    "widget_type": CheckBoxes,
                    "value": (),
                },
            ],
        ):
            """Delete an existing workflow file."""
            for filename in filenames:
                path = _config.workflow_path(filename)
                if path.exists():
                    assert path.suffix == ".py"
                    path.unlink()
                else:
                    raise FileNotFoundError(
                        f"Workflow file not found: {path.as_posix()}"
                    )
                name = self._make_method_name(path)
                for i, action in enumerate(self):
                    if action.name == name:
                        del self[i]
                        break
            self.reset_choices()
            return None

        sep0 = field(Separator)

    sep0 = field(Separator)

    @set_design(text="Command palette")
    @do_not_record
    @bind_key("Ctrl+P")
    def open_command_palette(self):
        """Open the command palette widget."""
        from magicclass.command_palette import exec_command_palette

        def _title_fmt(parent, widget):
            qn = type(parent).__qualname__
            if qn.startswith("CylindraMainWidget."):
                return qn[len("CylindraMainWidget.") :]
            return qn

        def _filter(parent, widget):
            qn = type(parent).__qualname__
            if qn.startswith("_") or "._" in qn:
                return False
            return True

        return exec_command_palette(
            self._get_main(),
            alignment="screen",
            title=_title_fmt,
            filter=_filter,
        )

    @set_design(text="Open one-line runner")
    @do_not_record
    def open_one_line_runner(self):
        """Open the one-line runner widget."""
        from IPython import get_ipython

        ipy = get_ipython()

        def _injector():
            return ipy.kernel.shell.user_ns

        wdt = OneLineRunner(injector=_injector)
        self.parent_viewer.window.add_dock_widget(
            wdt, name="One-line runner", area="right"
        )
        return None

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
        cylindra_info = abstractapi()
        report_issues = abstractapi()

    @do_not_record
    @set_design(text="Configure dask")
    def configure_dask(
        self,
        num_workers: Optional[Annotated[int, {"min": 1}]] = None,
        scheduler: Literal["single-threaded", "threads", "synchronous", "processes"] = "threads",
    ):  # fmt: skip
        """
        Configure dask parallel computation.

        Parameters
        ----------
        num_workers : int, optional
            Number of workers to use. If not specified, the maximum number of workers
            will be used.
        scheduler : str, default is 'threads'
            The scheduler to use.
        """
        import dask.config

        return dask.config.set(num_workers=num_workers, scheduler=scheduler)

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
    workflow = workflow.replace("\t", "    ")
    expr = parse(workflow)
    if errors := check_call_args(expr, {"ui": ui}):
        msg = "".join(map(lambda s: f"\n - {s}", errors))
        raise ValueError(f"Method errors found in workflow script: {msg}")
    if errors := check_attributes(expr, {"ui": ui}):
        msg = "".join(map(lambda s: f"\n - {s}", errors))
        raise ValueError(f"Attribute errors found in workflow script: {msg}")
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
    txt.syntax_highlight("python", theme=get_code_theme(self))
    txt.read_only = True
    gui.insert(1, txt)

    @gui.filename.changed.connect
    def _on_name_change(filename: str | None):
        if filename is None:
            return
        txt.value = _config.workflow_path(filename).read_text()

    _on_name_change(gui.filename.value)


@setup_function_gui(Others.Workflows.define_workflow)
def _(self: Others.Workflows, gui: "FunctionGui"):
    gui.workflow.syntax_highlight("python", theme=get_code_theme(self))
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
    gui.workflow.syntax_highlight("python", theme=get_code_theme(self))

    @gui.filename.changed.connect
    def _on_name_change(filename: str | None):
        if filename is None:
            return
        gui.workflow.value = _config.workflow_path(filename).read_text()

    _on_name_change(gui.filename.value)
