import os
from typing import Annotated, TYPE_CHECKING, Literal, Sequence
import warnings
from weakref import WeakSet

import impy as ip
import macrokit as mk
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from acryo import Molecules, SubtomogramLoader

from magicgui.widgets import Widget
from magicclass import (
    MagicTemplate,
    bind_key,
    build_help,
    confirm,
    do_not_record,
    field,
    get_function_gui,
    magicclass,
    nogui,
    set_design,
)
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.ext.polars import DataFrameView
from magicclass.types import (
    Color,
    Colormap as ColormapType,
    Optional,
    Path,
    ExprStr,
    Bound,
)
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.widgets import ConsoleTextEdit
from magicclass.undo import undo_callback

from napari.layers import Image, Layer, Points
from napari.utils.colormaps import Colormap

from cylindra import utils
from cylindra.components import CylSpline, CylTomogram
from cylindra.const import (
    SELECTION_LAYER_NAME,
    WORKING_LAYER_NAME,
    GlobalVariables as GVar,
    IDName,
    PropertyNames as H,
    MoleculesHeader as Mole,
    Ori,
    nm,
    get_versions,
)
from cylindra._custom_layers import MoleculesLayer, CylinderLabels
from cylindra.types import ColoredLayer, get_monomer_layers
from cylindra.project import CylindraProject, get_project_json

# widgets
from cylindra.widgets import _shared_doc, subwidgets
from cylindra.widgets import widget_utils
from cylindra.widgets.image_processor import ImageProcessor
from cylindra.widgets.properties import GlobalPropertiesWidget, LocalPropertiesWidget
from cylindra.widgets.spline_control import SplineControl
from cylindra.widgets.spline_clipper import SplineClipper
from cylindra.widgets.spline_fitter import SplineFitter
from cylindra.widgets.sta import SubtomogramAveraging
from cylindra.widgets.sweeper import SplineSweeper
from cylindra.widgets.simulator import CylinderSimulator
from cylindra.widgets.measure import SpectraMeasurer

from cylindra.widgets.widget_utils import (
    FileFilter,
    add_molecules,
    change_viewer_focus,
    POLARS_NAMESPACE,
)

if TYPE_CHECKING:
    from .batch import CylindraBatchWidget
    from cylindra.components._base import BaseComponent

ICON_DIR = Path(__file__).parent / "icons"
SPLINE_ID = "spline-id"
SELF = mk.Mock("self")
DEFAULT_COLORMAP = {0.0: "#0B0000", 0.58: "#FF0000", 1.0: "#FFFF00"}
_Logger = getLogger("cylindra")

# stylesheet
_STYLE = (Path(__file__).parent / "style.qss").read_text()

############################################################################################
#   The Main Widget of cylindra
############################################################################################


@magicclass(widget_type="scrollable", stylesheet=_STYLE, name="cylindra")
@_shared_doc.update_cls
class CylindraMainWidget(MagicTemplate):
    # Main GUI class.

    # Weak references of active widgets. Useful for test session teardown.
    _active_widgets = WeakSet[Widget]()

    # Widget for manual spline fitting
    spline_fitter = field(SplineFitter, name="_Spline fitter")
    # Widget for manual spline clipping
    spline_clipper = field(SplineClipper, name="_Spline clipper")
    # Widget for sweeping along splines
    spline_sweeper = field(SplineSweeper, name="_Spline sweeper")
    # Widget for pre-filtering/pre-processing
    image_processor = field(ImageProcessor, name="_Image Processor")
    # Widget for tomogram simulator
    cylinder_simulator = field(CylinderSimulator, name="_Cylinder Simulator")
    # Widget for measuring FFT parameters from a 2D power spectra
    spectra_measurer = field(SpectraMeasurer, name="_FFT Measurer")
    # Widget for subtomogram analysis
    sta = field(SubtomogramAveraging, name="_Subtomogram averaging")

    @property
    def batch(self) -> "CylindraBatchWidget":
        """Return the batch analyzer."""
        if self._batch is None:
            self.open_project_batch_analyzer()
        return self._batch

    @property
    def project_directory(self) -> "Path | None":
        """The current project directory."""
        if source := self.tomogram.source:
            return source.parent
        return None

    # Menu bar
    File = subwidgets.File
    ImageMenu = subwidgets.Image
    Splines = subwidgets.Splines
    MoleculesMenu = subwidgets.MoleculesMenu
    Analysis = subwidgets.Analysis
    Others = subwidgets.Others

    # Menu for global variables
    @property
    def global_variables(self):
        """Return the global variable widget."""
        return self.Others.GlobalVariables

    # Toolbar
    toolbar = subwidgets.toolbar

    # Child widgets
    ImageInfo = subwidgets.ImageInfo
    SplineControl = SplineControl  # Widget for controling splines
    # Widget for summary of local properties
    LocalProperties = field(LocalPropertiesWidget, name="Local Properties")
    # Widget for summary of glocal properties
    GlobalProperties = field(GlobalPropertiesWidget, name="Global Properties")
    # Widget for 2D overview of splines
    overview = field(QtImageCanvas, name="Overview").with_options(tooltip="Overview of splines")  # fmt: skip

    ### methods ###

    def __init__(self):
        self.tomogram: CylTomogram = None
        self._current_ft_size: nm = 50.0
        self._tilt_range: "tuple[float, float] | None" = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: CylinderLabels = None
        self._macro_offset: int = 1
        self._need_save: bool = False
        self._batch = None
        self.objectName()  # load napari types

        GVar.events.connect(self._global_variable_updated)

    def __post_init__(self):
        self.min_width = 400
        self.LocalProperties.collapsed = False
        self.GlobalProperties.collapsed = False
        self.overview.min_height = 300
        self.global_variables.load_default()
        return None

    def _get_splines(self, widget=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        tomo = self.tomogram
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_spline_coordinates(self, widget=None) -> np.ndarray:
        """Get coordinates of the manually picked spline."""
        coords = self.layer_work.data
        return np.round(coords, 3)

    def _get_available_binsize(self, _=None) -> list[int]:
        if self.tomogram is None:
            return [1]
        out = [x[0] for x in self.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return out

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "add_spline.svg")
    @bind_key("F1")
    def register_path(self, coords: Bound[_get_spline_coordinates] = None):
        """Register current selected points as a spline path."""
        if coords is None:
            _coords = self.layer_work.data
        else:
            _coords = np.asarray(coords)

        if _coords.size == 0:
            warnings.warn("No points are in the viewer.", UserWarning)
            return None

        tomo = self.tomogram
        tomo.add_spline(_coords)
        spl = tomo.splines[-1]

        # draw path
        self._add_spline_to_images(spl, tomo.n_splines - 1)
        self.layer_work.data = []
        self.layer_prof.selected_data = set()
        self.reset_choices()
        self.SplineControl.num = tomo.n_splines - 1

        return undo_callback(self.delete_spline).with_args(-1)

    _runner = field(subwidgets.Runner)
    _image_loader = subwidgets.ImageLoader

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "run_all.svg")
    @bind_key("F2")
    @do_not_record
    def open_runner(self):
        """Run cylindrical fitting algorithm with various settings."""
        return self._runner.show(run=False)

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "clear_last.svg")
    @confirm(
        text="Spline has properties. Are you sure to delete it?",
        condition="self.tomogram.splines[self.SplineControl.num].has_props()",
    )
    @do_not_record
    def clear_current(self):
        """Clear current selection."""
        if self.layer_work.data.size > 0:
            self.layer_work.data = []
        else:
            self.delete_spline(self.SplineControl.num)

        return None

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "clear_all.svg")
    @confirm(text="Are you sure to clear all?\nYou cannot undo this.")
    def clear_all(self):
        """Clear all the splines and results."""
        self._init_widget_state()
        self._init_layers()
        self.overview.layers.clear()
        self.tomogram.clear_cache()
        self.tomogram.splines.clear()
        self._need_save = False
        self.reset_choices()
        return None

    @Others.Help.wraps
    @set_design(text="Open help")
    @do_not_record
    def open_help(self):
        """Open a help window."""
        help = build_help(self)
        help.show()
        return None

    def _format_macro(self, macro: mk.Macro = None):
        if macro is None:
            macro = self.macro
        v = mk.Expr("getattr", [mk.symbol(self), "parent_viewer"])
        return macro.format([(mk.symbol(self.parent_viewer), v)])

    def _load_macro_file(self, path: Path):
        with open(path) as f:
            txt = f.read()
        macro = mk.parse(txt)
        return self._format_macro(macro)

    @Others.Macro.wraps
    @set_design(text="Load file")
    @do_not_record
    def load_macro_file(self, path: Path.Read[FileFilter.PY]):
        """Load a Python script file to a new macro window."""
        macro = self._load_macro_file(path)
        edit = self.macro.widget.new_window(path.name)
        edit.textedit.value = str(macro)
        return None

    @Others.Macro.wraps
    @set_design(text="Run file")
    @do_not_record
    def run_file(self, path: Path.Read[FileFilter.PY]):
        """Run a Python script file."""
        macro = self._load_macro_file(path)
        _ui = str(mk.symbol(self))
        with self.macro.blocked():
            macro.eval({}, {_ui: self})
        self.macro.extend(macro.args)
        return None

    @Others.Macro.wraps
    @set_design(text="Show macro")
    @do_not_record
    @bind_key("Ctrl-Shift-M")
    def show_macro(self):
        """Create Python executable script of the current project."""
        new = self.macro.widget.new_window()
        new.textedit.value = str(self._format_macro()[self._macro_offset :])
        new.show()
        return None

    @Others.Macro.wraps
    @set_design(text="Show full macro")
    @do_not_record
    def show_full_macro(self):
        """Create Python executable script since the startup this time."""
        new = self.macro.widget.new_window()
        new.textedit.value = str(self._format_macro())
        new.show()
        return None

    @Others.Macro.wraps
    @set_design(text="Show native macro")
    @do_not_record
    def show_native_macro(self):
        """
        Show the native macro widget of magic-class, which is always synchronized but
        is not editable.
        """
        self.macro.widget.show()
        return None

    @Others.wraps
    @set_design(text="Open command palette")
    @do_not_record
    @bind_key("Ctrl-P")
    def open_command_palette(self):
        from magicclass.command_palette import exec_command_palette

        return exec_command_palette(self, alignment="screen")

    @Others.wraps
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

    @Others.Help.wraps
    @set_design(text="Info")
    @do_not_record
    def cylindra_info(self):
        """Show information of dependencies."""
        versions = get_versions()
        value = "\n".join(f"{k}: {v}" for k, v in versions.items())
        w = ConsoleTextEdit(value=value)
        w.read_only = True
        w.native.setParent(self.native, w.native.windowFlags())
        w.show()
        return None

    @Others.Help.wraps
    @set_design(text="Report issues")
    @do_not_record
    def report_issues(self):
        """Report issues on GitHub."""
        from magicclass.utils import open_url

        open_url("https://github.com/hanjinliu/cylindra/issues/new")
        return None

    @File.wraps
    @set_design(text="Open image")
    @do_not_record
    def open_image_loader(self):
        """Load an image file and process it before sending it to the viewer."""
        return self._image_loader.show()

    @_image_loader.wraps
    @set_design(text="Run")
    @dask_thread_worker.with_progress(desc="Reading image")
    @confirm(
        text="You may have unsaved data. Open a new tomogram?",
        condition=SELF._need_save,
    )
    def open_image(
        self,
        path: Bound[_image_loader.path],
        scale: Bound[_image_loader.scale.scale_value] = 1.0,
        tilt_range: Bound[_image_loader.tilt_range.range] = None,
        bin_size: Bound[_image_loader.bin_size] = [1],
        filter: Bound[_image_loader.filter_reference_image] = True,
    ):  # fmt: skip
        """
        Load an image file and process it before sending it to the viewer.

        Parameters
        ----------
        path : Path
            Path to the tomogram. Must be 3-D image.
        scale : float, default is 1.0
            Pixel size in nm/pixel unit.
        bin_size : int or list of int, default is [1]
            Initial bin size of image. Binned image will be used for visualization in the viewer.
            You can use both binned and non-binned image for analysis.
        filter : bool, default is True
            Apply low-pass filter on the reference image (does not affect image data itself).
        """
        img = ip.lazy_imread(path, chunks=GVar.dask_chunk)
        if scale is not None:
            scale = float(scale)
            img.scale.x = img.scale.y = img.scale.z = scale
        else:
            scale = img.scale.x
        if isinstance(bin_size, int):
            bin_size = [bin_size]
        elif len(bin_size) == 0:
            raise ValueError("You must specify at least one bin size.")
        else:
            bin_size = list(bin_size)
        bin_size = list(set(bin_size))  # delete duplication
        tomo = CylTomogram.imread(
            path=path,
            scale=scale,
            tilt_range=tilt_range,
            binsize=bin_size,
        )

        self._macro_offset = len(self.macro)
        self.tomogram = tomo
        return thread_worker.to_callback(self._send_tomogram_to_viewer, filter)

    @open_image.started.connect
    def _open_image_on_start(self):
        return self._image_loader.close()

    @File.wraps
    @set_design(text="Load project")
    @dask_thread_worker.with_progress(desc="Reading project")
    @confirm(
        text="You may have unsaved data. Open a new project?",
        condition=SELF._need_save,
    )
    @do_not_record
    def load_project(self, path: Path.Read[FileFilter.PROJECT], filter: bool = True):
        """Load a project json file."""
        project_path = get_project_json(path)
        project = CylindraProject.from_json(project_path)
        _Logger.print(f"Project loaded: {project_path.as_posix()}")
        return thread_worker.to_callback(project.to_gui(self, filter=filter))

    @File.wraps
    @set_design(text="Save project")
    def save_project(self, save_dir: Path.Save):
        """
        Save current project state and the results in a directory.

        The json file contains paths of images and results, parameters of splines,
        scales and version. Local and global properties, molecule coordinates and
        features will be exported as csv files. If results are saved at the default
        directory, they will be written as relative paths in the project json file
        so that moving root directory does not affect loading behavior.

        Parameters
        ----------
        save_dir : Path
            Path of json file.
        """
        save_dir = Path(save_dir)
        CylindraProject.save_gui(self, save_dir / "project.json", save_dir)
        _Logger.print(f"Project saved: {save_dir.as_posix()}")
        self._need_save = False
        return

    @File.wraps
    @set_design(text="Load splines")
    def load_splines(self, paths: Path.Multiple[FileFilter.JSON]):
        """
        Load splines from a list of json paths.

        Parameters
        ----------
        paths : list of path-like objects
            Paths to json files that describe spline parameters in the correct format.
        """
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        splines = [CylSpline.from_json(path) for path in paths]
        self.tomogram.splines.extend(splines)
        self._update_splines_in_images()
        self.reset_choices()
        return None

    @File.wraps
    @set_design(text="Load molecules")
    def load_molecules(self, paths: Path.Multiple[FileFilter.CSV]):
        """Load molecules from a csv file."""
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        for path in paths:
            mole = Molecules.from_csv(path)
            name = Path(path).stem
            add_molecules(self.parent_viewer, mole, name)
        return None

    @File.wraps
    @set_design(text="Save spline")
    def save_spline(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        save_path: Path.Save[FileFilter.JSON],
    ):
        """Save splines as a json file."""
        spl = self.tomogram.splines[spline]
        spl.to_json(save_path)
        return None

    @File.wraps
    @set_design(text="Save molecules")
    def save_molecules(
        self, layer: MoleculesLayer, save_path: Path.Save[FileFilter.CSV]
    ):
        """
        Save monomer coordinates, orientation and features as a csv file.

        Parameters
        ----------
        {layer}
        save_path : Path
            Where to save the molecules.
        """
        mole = layer.molecules
        mole.to_csv(save_path)
        return None

    @File.wraps
    @set_design(text="Process images")
    @do_not_record
    def process_images(self):
        """Open image processor."""
        return self.image_processor.show()

    @File.wraps
    @set_design(text="View project")
    @do_not_record
    def view_project(self, path: Path.Read[FileFilter.JSON]):
        pviewer = CylindraProject.from_json(path).make_project_viewer()
        pviewer.native.setParent(self.native, pviewer.native.windowFlags())
        return pviewer.show()

    @ImageMenu.wraps
    @set_design(text="Filter reference image")
    @dask_thread_worker.with_progress(desc="Low-pass filtering")
    @do_not_record
    def filter_reference_image(self):
        """Apply low-pass filter to enhance contrast of the reference image."""
        cutoff = 0.2
        with utils.set_gpu():
            img: ip.ImgArray = self.layer_image.data
            overlap = [min(s, 32) for s in img.shape]
            self.layer_image.data = img.tiled_lowpass_filter(
                cutoff,
                chunks=(96, 96, 96),
                overlap=overlap,
            )
        contrast_limits = np.percentile(self.layer_image.data, [1, 99.9])

        @thread_worker.to_callback
        def _filter_reference_image_on_return():
            self.layer_image.contrast_limits = contrast_limits
            proj = self.layer_image.data.proj("z")
            self.overview.image = proj
            self.overview.contrast_limits = contrast_limits

        return _filter_reference_image_on_return

    @ImageMenu.wraps
    @set_design(text="Add multi-scale")
    @dask_thread_worker.with_progress(
        desc=lambda bin_size: f"Adding multiscale (bin = {bin_size})"
    )
    def add_multiscale(
        self,
        bin_size: Annotated[int, {"choices": list(range(2, 17))}] = 4,
    ):
        """
        Add a new multi-scale image of current tomogram.

        Parameters
        ----------
        bin_size : int, default is 4
            Bin size of the new image
        """
        tomo = self.tomogram
        tomo.get_multiscale(binsize=bin_size, add=True)
        self._need_save = True
        return thread_worker.to_callback(self.set_multiscale, bin_size)

    @ImageMenu.wraps
    @set_design(text="Set multi-scale")
    def set_multiscale(
        self, bin_size: Annotated[int, {"choices": _get_available_binsize}]
    ):
        """
        Set multiscale used for image display.

        Parameters
        ----------
        bin_size: int
            Bin size of multiscaled image.
        """
        tomo = self.tomogram
        _old_bin_size = self.layer_image.metadata["current_binsize"]
        imgb = tomo.get_multiscale(bin_size)
        factor = self.layer_image.scale[0] / imgb.scale.x
        current_z = self.parent_viewer.dims.current_step[0]
        self.layer_image.data = imgb
        self.layer_image.scale = imgb.scale
        self.layer_image.name = f"{imgb.name} (bin {bin_size})"
        self.layer_image.translate = [tomo.multiscale_translation(bin_size)] * 3
        self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]
        self.parent_viewer.dims.set_current_step(axis=0, value=current_z * factor)

        if self.layer_paint is not None:
            self.layer_paint.scale = self.layer_image.scale
            self.layer_paint.translate = self.layer_image.translate

        # update overview
        self.overview.image = imgb.proj("z")
        self.overview.xlim = [x * factor for x in self.overview.xlim]
        self.overview.ylim = [y * factor for y in self.overview.ylim]
        self.layer_image.metadata["current_binsize"] = bin_size
        self.reset_choices()
        return undo_callback(self.set_multiscale).with_args(_old_bin_size)

    @ImageMenu.wraps
    @do_not_record
    @set_design(text="Open spline sweeper")
    def open_sweeper(self):
        """Open spline sweeper widget"""
        self.spline_sweeper.show()
        return self.spline_sweeper.refresh_widget_state()

    @ImageMenu.wraps
    @set_design(text="Sample subtomograms")
    def sample_subtomograms(self):
        """Sample subtomograms at the anchor points on splines"""
        self.spline_fitter.close()

        # initialize GUI
        if len(self.tomogram.splines) == 0:
            raise ValueError("No spline found.")
        spl = self.tomogram.splines[0]
        if spl._anchors is not None:
            self.SplineControl["pos"].max = spl.anchors.size - 1
        self.SplineControl._num_changed()
        self.layer_work.mode = "pan_zoom"

        self._update_local_properties_in_widget()
        self._update_global_properties_in_widget()
        self._highlight_spline()

        # reset contrast limits
        self.SplineControl._reset_contrast_limits()
        return None

    @Splines.wraps
    @set_design(text="Show splines as curves")
    def show_splines(self):
        """Show 3D spline paths of cylinder central axes as a layer."""
        paths = [r.partition(100) for r in self.tomogram.splines]

        layer = self.parent_viewer.add_shapes(
            paths,
            shape_type="path",
            name="Spline Curves",
            edge_color="lime",
            edge_width=1,
        )
        return (
            undo_callback(self._try_removing_layer)
            .with_args(layer)
            .with_redo(self._add_layers_future(layer))
        )

    @Splines.wraps
    @set_design(text="Show splines as meshes")
    def show_splines_as_meshes(self):
        """Show 3D spline cylinder as a surface layer."""
        nodes = []
        vertices = []
        n_nodes = 0
        for i, spl in enumerate(self.tomogram.splines):
            n, v = spl.cylinder_model().to_mesh(spl)
            nodes.append(n)
            vertices.append(v + i * n_nodes)
            n_nodes += n.shape[0]
        nodes = np.concatenate(nodes, axis=0)
        vertices = np.concatenate(vertices, axis=0)
        layer = self.parent_viewer.add_surface([nodes, vertices], shading="smooth")
        # NOTE: re-adding surface layer is not redoable, since viewer.add_layer seems
        # broken for the surface layer.
        return undo_callback(self._try_removing_layer, redo=False).with_args(layer)

    @Splines.Orientation.wraps
    @set_design(text="Invert spline")
    def invert_spline(self, spline: Annotated[int, {"bind": SplineControl.num}] = None):
        """
        Invert current displayed spline **in place**.

        Parameters
        ----------
        spline : int, optional
            ID of splines to be inverted.
        """
        if spline is None:
            return
        spl = self.tomogram.splines[spline]
        self.tomogram.splines[spline] = spl.invert()
        self._update_splines_in_images()
        self.reset_choices()

        need_resample = self.SplineControl.need_resample
        self._init_widget_state()
        if need_resample:
            self.sample_subtomograms()
        self._set_orientation_marker(spline)
        self._need_save = True
        return undo_callback(self.invert_spline).with_args(spline)

    @Splines.Orientation.wraps
    @set_design(text="Align to polarity")
    def align_to_polarity(
        self, orientation: Literal["MinusToPlus", "PlusToMinus"] = "MinusToPlus"
    ):
        """
        Align all the splines in the direction parallel to the cylinder polarity.

        Parameters
        ----------
        orientation : Ori, default is Ori.MinusToPlus
            To which direction splines will be aligned.
        """
        need_resample = self.SplineControl.need_resample
        _old_orientations = [spl.orientation for spl in self.tomogram.splines]
        self.tomogram.align_to_polarity(orientation=orientation)
        self._update_splines_in_images()
        self._init_widget_state()
        self.reset_choices()
        if need_resample:
            self.sample_subtomograms()
        for i in range(len(self.tomogram.splines)):
            self._set_orientation_marker(i)
        _new_orientations = [spl.orientation for spl in self.tomogram.splines]
        self._need_save = True

        return (
            undo_callback(self._set_orientations)
            .with_args(_old_orientations, need_resample)
            .with_redo(lambda: self._set_orientations(_new_orientations))
        )

    @Splines.Orientation.wraps
    @set_design(text="Auto-align to polarity")
    @thread_worker.with_progress(
        desc="Auto-detecting polarities...", total="len(self.tomogram.splines)"
    )
    def auto_align_to_polarity(
        self,
        align_to: Annotated[
            Optional[Literal["MinusToPlus", "PlusToMinus"]], {"text": "Do not align"}
        ] = None,
        depth: Annotated[nm, {"min": 5.0, "max": 500.0, "step": 5.0}] = 40,
    ):
        """
        Automatically detect the polarities and align if necessary.

        This function uses Fourier vorticity to detect the polarities of the splines.
        The subtomogram at the center of the spline will be sampled in the cylindric
        coordinate and the power spectra in (radius, angle) space will be calculated.
        The peak position of the `angle = nPF` line scan will be used to detect the
        polarity of the spline.

        Parameters
        ----------
        align_to : Ori, optional
            To which direction splines will be aligned. If not given, splines will
            not be inverted even if the orientation is not aligned.
        depth : nm, default is 40 nm
            Depth (Y-length) of the subtomogram to be sampled.
        """
        binsize: int = self.layer_image.metadata["current_binsize"]
        tomo = self.tomogram
        _old_orientations = [spl.orientation for spl in self.tomogram.splines]
        _new_orientations = tomo.infer_polarity(binsize=binsize, depth=depth)
        for i in range(len(tomo.splines)):
            spl = tomo.splines[i]
            spl.orientation = _new_orientations[i]

        if align_to is not None:
            return thread_worker.to_callback(self.align_to_polarity, align_to)
        else:

            @thread_worker.to_callback
            def _on_return():
                self._update_splines_in_images()
                for i in range(len(tomo.splines)):
                    self._set_orientation_marker(i)

                self.SplineControl._update_canvas()
                return (
                    undo_callback(self._set_orientations)
                    .with_args(_old_orientations)
                    .with_redo(lambda: self._set_orientations(_new_orientations))
                )

            return _on_return

    def _set_orientations(self, orientations: list[Ori], resample: bool = True):
        for spl, ori in zip(self.tomogram.splines, orientations):
            spl.orientation = ori
        self._update_splines_in_images()
        self._init_widget_state()
        self.reset_choices()
        for i in range(len(self.tomogram.splines)):
            self._set_orientation_marker(i)
        if resample:
            self.sample_subtomograms()
        return None

    @Splines.wraps
    @set_design(text="Clip splines")
    def clip_spline(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        clip_lengths: Annotated[tuple[nm, nm], {"options": {"min": 0.0, "max": 1000.0, "step": 0.1, "label": "clip length (nm)"}}] = (0.0, 0.0),
    ):  # fmt: skip
        """
        Clip selected spline at its edges by given lengths.

        Parameters
        ----------
        spline : int
           The ID of spline to be clipped.
        clip_lengths : tuple of float, default is (0., 0.)
            The length in nm to be clipped at the start and end of the spline.
        """
        if spline is None:
            return
        spl = self.tomogram.splines[spline]
        _old_lims = spl.lims
        length = spl.length()
        start, stop = np.array(clip_lengths) / length
        self.tomogram.splines[spline] = spl.clip(start, 1 - stop)
        self._update_splines_in_images()
        self._need_save = True
        # current layer will be removed. Select another layer.
        self.parent_viewer.layers.selection = {self.layer_work}
        # initialize clipping values
        fgui = get_function_gui(self, "clip_spline")
        fgui.clip_lengths.value = (0.0, 0.0)

        @undo_callback
        def out():
            self.tomogram.splines[spline] = spl.restore().clip(*_old_lims)
            self._update_splines_in_images()

        return out

    def _confirm_delete(self):
        i = self.SplineControl.num
        return self.tomogram.splines[i].has_props()

    @Splines.wraps
    @set_design(text="Delete spline")
    @confirm(
        text="Spline has properties. Are you sure to delete it?",
        condition=_confirm_delete,
    )
    def delete_spline(self, i: Annotated[int, {"bind": SplineControl.num}]):
        """Delete currently selected spline."""
        if i < 0:
            i = len(self.tomogram.splines) - 1
        spl = self.tomogram.splines.pop(i)
        self.reset_choices()

        # update layer
        features = self.layer_prof.features
        spline_id = features[SPLINE_ID]
        spec = spline_id != i
        old_data = self.layer_prof.data
        self.layer_prof.data = old_data[spec]
        new_features = features[spec].copy()
        spline_id = np.asarray(new_features[SPLINE_ID])
        spline_id[spline_id >= i] -= 1
        new_features[SPLINE_ID] = spline_id
        self._update_splines_in_images()
        self.layer_prof.features = new_features
        self.layer_prof.feature_defaults[SPLINE_ID] = len(self.tomogram.splines)
        if self.SplineControl.need_resample and len(self.tomogram.splines) > 0:
            self.sample_subtomograms()
        self._need_save = True

        @undo_callback
        def out():
            self.tomogram.splines.insert(i, spl)
            self.layer_prof.data = old_data
            self.layer_prof.features = features
            self._add_spline_to_images(spl, i)
            self._update_splines_in_images()
            self.reset_choices()

        return out

    @Splines.wraps
    @set_design(text="Fit splines")
    @thread_worker.with_progress(desc="Spline Fitting", total="len(splines)")
    def fit_splines(
        self,
        splines: Annotated[
            list[int], {"choices": _get_splines, "widget_type": "Select"}
        ] = (),
        max_interval: Annotated[nm, {"label": "Max interval (nm)"}] = 30,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        degree_precision: float = 0.5,
        edge_sigma: Annotated[Optional[nm], {"text": "Do not mask image"}] = 2.0,
        max_shift: nm = 5.0,
    ):
        """
        Fit cylinder with spline curve, using manually selected points.

        Parameters
        ----------
        {splines}{max_interval}{bin_size}
        degree_precision : float, default is 0.5
            Precision of xy-tilt degree in angular correlation.
        edge_sigma : bool, default is False
            Check if cylindric structures are densely packed. Initial spline position must
            be "almost" fitted in dense mode.
        max_shift : nm, default is 5.0
            Maximum shift to be applied to each point of splines.
        """
        tomo = self.tomogram
        if len(splines) == 0:
            splines = list(range(tomo.n_splines))
        old_splines = {i: tomo.splines[i].copy() for i in splines}
        for i in splines:
            tomo.fit(
                i,
                max_interval=max_interval,
                binsize=bin_size,
                degree_precision=degree_precision,
                edge_sigma=edge_sigma,
                max_shift=max_shift,
            )
            yield thread_worker.to_callback(self._update_splines_in_images)
        new_splines = {i: tomo.splines[i].copy() for i in splines}
        self._need_save = True

        @undo_callback
        def _undo():
            for i, spl in old_splines.items():
                tomo.splines[i].copy_from(spl)
            self._update_splines_in_images()

        @_undo.with_redo
        def _undo():
            for i, spl in new_splines.items():
                tomo.splines[i].copy_from(spl)
            self._init_widget_state()
            self._update_splines_in_images()

        @thread_worker.to_callback
        def out():
            self._init_widget_state()
            self._update_splines_in_images()
            return _undo

        return out

    @Splines.wraps
    @set_design(text="Fit splines manually")
    @do_not_record
    def fit_splines_manually(
        self, max_interval: Annotated[nm, {"label": "Max interval (nm)"}] = 50.0
    ):
        """
        Open a spline fitter window and fit cylinder with spline manually.

        Parameters
        ----------
        {max_interval}
        """
        self.spline_fitter._load_parent_state(max_interval=max_interval)
        self.spline_fitter.show()
        return None

    @Splines.wraps
    @set_design(text="Add anchors")
    def add_anchors(
        self,
        splines: Annotated[list[int], {"choices": _get_splines, "widget_type": "Select"}] = (),
        interval: Annotated[nm, {"label": "Interval between anchors (nm)", "min": 1.0}] = 25.0,
    ):  # fmt: skip
        """
        Add anchors to splines.

        Parameters
        ----------
        {splines}{interval}
        """
        tomo = self.tomogram
        if len(splines) == 0:
            splines = list(range(tomo.n_splines))
        if len(splines) == 0:
            raise ValueError("Cannot add anchors before adding splines.")
        tomo.make_anchors(splines, interval=interval)
        self._update_splines_in_images()
        self._need_save = True
        return None

    @Analysis.wraps
    @set_design(text="Measure radius")
    @thread_worker.with_progress(desc="Measuring Radius", total="len(splines)")
    def measure_radius(
        self,
        splines: Annotated[list[int], {"choices": _get_splines, "widget_type": "Select"}] = (),
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Measure cylinder radius for each spline curve.

        Parameters
        ----------
        {splines}{bin_size}
        """
        if len(splines) == 0:
            splines = list(range(self.tomogram.n_splines))
        old_radius = {i: self.tomogram.splines[i].radius for i in splines}
        new_radius = {}
        for i in splines:
            radius = self.tomogram.set_radius(i, binsize=bin_size)
            yield
            new_radius[i] = radius
        self._need_save = True

        def out(radius_dict: dict[int, nm]):
            def wrapper():
                for i, radius in radius_dict.items():
                    self.tomogram.splines[i].radius = radius

            return wrapper

        return undo_callback(out(new_radius)).with_redo(out(old_radius))

    @Splines.wraps
    @set_design(text="Refine splines")
    @thread_worker.with_progress(desc="Refining splines", total="len(splines)")
    def refine_splines(
        self,
        splines: Annotated[list[int], {"choices": _get_splines, "widget_type": "Select"}] = (),
        max_interval: Annotated[nm, {"label": "Maximum interval (nm)"}] = 30,
        corr_allowed: Annotated[float, {"label": "Correlation allowed", "max": 1.0, "step": 0.1}] = 0.9,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Refine splines using the global cylindric structural parameters.

        Parameters
        ----------
        {splines}{max_interval}
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        {bin_size}
        """
        tomo = self.tomogram
        if len(splines) == 0:
            splines = list(range(tomo.n_splines))

        old_splines = {i: tomo.splines[i].copy() for i in splines}
        for i in splines:
            tomo.refine(
                i,
                max_interval=max_interval,
                corr_allowed=corr_allowed,
                binsize=bin_size,
            )
            yield thread_worker.to_callback(self._update_splines_in_images)

        new_splines = {i: tomo.splines[i].copy() for i in splines}

        self._need_save = True

        @undo_callback
        def undo():
            for i, spl in old_splines.items():
                tomo.splines[i].copy_from(spl)
            self._update_splines_in_images()
            self._update_local_properties_in_widget()

        @undo.with_redo
        def undo():
            for i, spl in new_splines.items():
                tomo.splines[i].copy_from(spl)
            self._init_widget_state()
            self._update_splines_in_images()
            self._update_local_properties_in_widget()

        @thread_worker.to_callback
        def out():
            self._init_widget_state()
            self._update_splines_in_images()
            self._update_local_properties_in_widget()
            return undo

        return out

    @Splines.wraps
    @set_design(text="Set spline parameters")
    def set_spline_props(
        self,
        spline: Annotated[int, {"bind": SplineControl.num}],
        spacing: Annotated[Optional[nm], {"label": "spacing (nm)", "text": "Do not update"}] = None,
        skew: Annotated[Optional[float], {"label": "skew angle (deg)", "text": "Do not update"}] = None,
        rise: Annotated[Optional[nm], {"label": "rise angle (deg)", "text": "Do not update"}] = None,
        npf: Annotated[Optional[int], {"label": "number of PF", "text": "Do not update"}] = None,
        radius: Annotated[Optional[nm], {"label": "radius (nm)", "text": "Do not update"}] = None,
        orientation: Annotated[Optional[Literal["MinusToPlus", "PlusToMinus"]], {"text": "Do not update"}] = None,
    ):  # fmt: skip
        """
        Set spline global properties.

        This method will overwrite spline properties with the user input. You should
        not call this method unless there's a good reason to do so, e.g. the number
        of protofilaments is obviously wrong.

        Parameters
        ----------
        spline : int
            The index of spline to update.
        spacing : nm, optional
            If given, update the monomer spacing.
        skew : float, optional
            If given, update the skew angle.
        rise : float, optional
            If given, update the rise ange.
        npf : int , optional
            If given, update the number of protofilaments.
        radius : nm, optional
            If given, update the radius of the cylinder.
        orientation : str, optional
            If given, update the spline orientation.
        """
        spl = self.tomogram.splines[spline]
        old_spl = spl.copy()
        spl.update_props(
            spacing=spacing,
            skew=skew,
            rise=rise,
            npf=npf,
            radius=radius,
            orientation=orientation,
        )
        self.sample_subtomograms()
        self._update_splines_in_images()

        @undo_callback
        def out():
            spl.copy_from(old_spl)
            self.sample_subtomograms()
            self._update_splines_in_images()

        return out

    @Splines.wraps
    @set_design(text="Molecules to spline")
    def molecules_to_spline(
        self,
        layers: Annotated[list[MoleculesLayer], {"choices": get_monomer_layers, "widget_type": "Select"}],
        interval: Annotated[nm, {"label": "Interval (nm)", "min": 1.0}] = 24.5,
    ):  # fmt: skip
        """
        Create splines from molecules.

        This function is useful to refine splines using results of subtomogram
        alignment. If the molecules layer alreadly has a source spline, replace
        it with the new one.
        Note that this function only works with molecules that is correctly
        assembled by such as :func:`map_monomers`.

        Parameters
        ----------
        {layers}{interval}
        """
        tomo = self.tomogram
        if len(layers) == 0:
            raise ValueError("No layers are selected.")
        for layer in layers:
            layer: MoleculesLayer
            mole = layer.molecules
            spl = utils.molecules_to_spline(mole)
            if layer.source_component is not None:
                idx = tomo.splines.index(layer.source_component)
                tomo.splines[idx] = spl
            else:
                tomo.splines.append(spl)
            layer.source_component = spl
            spl.make_anchors(interval=interval)

        self.reset_choices()
        self.sample_subtomograms()
        self._update_splines_in_images()
        return None

    @Splines.wraps
    @set_design(text="Open spline clipper")
    @do_not_record
    def open_spline_clipper(self):
        """Open the spline clipper widget to precisely clip spines."""
        self.spline_clipper.show()
        if self.tomogram.n_splines > 0:
            self.spline_clipper.load_spline(self.SplineControl.num)

    @Analysis.wraps
    @set_design(text="Local FT analysis")
    @thread_worker.with_progress(desc="Local Fourier transform", total="len(splines)")
    def local_ft_analysis(
        self,
        splines: Annotated[list[int], {"choices": _get_splines, "widget_type": "Select"}] = (),
        interval: Annotated[nm, {"min": 1.0, "step": 0.5}] = 24.5,
        ft_size: Annotated[nm, {"min": 2.0, "step": 0.5}] = 24.5,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Determine cylindrical structural parameters by local Fourier transformation.

        Parameters
        ----------
        {splines}{interval}
        ft_size : nm, default is 32.0
            Longitudinal length of local discrete Fourier transformation used for
            structural analysis.
        {bin_size}
        """
        tomo = self.tomogram
        if len(splines) == 0:
            splines = list(range(tomo.n_splines))

        @thread_worker.to_callback
        def _local_ft_analysis_on_yield(i: int):
            if i == 0:
                self.sample_subtomograms()
            self._update_splines_in_images()
            self._update_local_properties_in_widget()

        def _set_props(props: dict[int, pl.DataFrame]):
            def wrapper():
                for i, df in props.items():
                    tomo.splines[i].localprops = df
                return None

            return wrapper

        old_props: dict[int, pl.DataFrame] = {}
        new_props: dict[int, pl.DataFrame] = {}
        for i in splines:
            spl = tomo.splines[i]
            old_props[i] = spl.localprops
            tomo.make_anchors(i=i, interval=interval)
            new_props[i] = tomo.local_ft_params(i=i, ft_size=ft_size, binsize=bin_size)
            yield _local_ft_analysis_on_yield(i)
        self._current_ft_size = ft_size
        self._need_save = True
        return undo_callback(_set_props(new_props)).with_redo(_set_props(old_props))

    @Analysis.wraps
    @set_design(text="Global FT analysis")
    @thread_worker.with_progress(desc="Global Fourier transform", total="len(splines)")
    def global_ft_analysis(
        self,
        splines: Annotated[list[int], {"choices": _get_splines, "widget_type": "Select"}] = (),
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Determine cylindrical global structural parameters by Fourier transformation.

        Parameters
        ----------
        {splines}{bin_size}
        """
        tomo = self.tomogram
        if len(splines) == 0:
            splines = list(range(tomo.n_splines))

        @thread_worker.to_callback
        def _global_ft_analysis_on_yield(i: int):
            if i == 0:
                self.sample_subtomograms()
            self._update_splines_in_images()
            self._update_local_properties_in_widget()

        def _set_props(props: dict[int, pl.DataFrame]):
            def wrapper():
                for i, df in props.items():
                    tomo.splines[i].globalprops = df
                return None

            return wrapper

        @thread_worker.to_callback
        def _global_ft_analysis_on_return():
            # show all in a table
            df = (
                self.tomogram.collect_globalprops()
                .drop(IDName.spline)
                .to_pandas()
                .transpose()
            )
            df.columns = [f"Spline-{i}" for i in range(len(df.columns))]
            _Logger.print_table(df, precision=3)
            self._update_global_properties_in_widget()

            return undo_callback(_set_props(old_props)).with_redo(_set_props(new_props))

        old_props: dict[int, pl.DataFrame] = {}
        new_props: dict[int, pl.DataFrame] = {}
        for i in splines:
            spl = tomo.splines[i]
            old_props[i] = spl.globalprops
            if spl.radius is None:
                tomo.set_radius(i=i)
            new_props[i] = tomo.global_ft_params(i=i, binsize=bin_size)
            yield _global_ft_analysis_on_yield(i)
        self._need_save = True
        return _global_ft_analysis_on_return

    def _get_reanalysis_macro(self, path: Path):
        """Get the macro expression for reanalysis in the given project path."""
        _ui_sym = mk.symbol(self)
        project = CylindraProject.from_json(get_project_json(path))
        macro_path = Path(project.macro)
        macro_expr = mk.parse(macro_path.read_text())
        return _filter_macro_for_reanalysis(macro_expr, _ui_sym)

    @Analysis.wraps
    @set_design(text="Re-analyze current tomogram")
    @do_not_record
    def reanalyze_image(self):
        """
        Reanalyze the current tomogram.

        This method will extract the first manual operations from current session.
        For better reproducibility, this method will reload the image.
        """
        _ui_sym = mk.symbol(self)
        macro = _filter_macro_for_reanalysis(
            self._format_macro()[self._macro_offset :], _ui_sym
        )
        return macro.eval({_ui_sym: self})

    @Analysis.wraps
    @set_design(text="Re-analyze project")
    @do_not_record
    def load_project_for_reanalysis(self, path: Path.Read[FileFilter.JSON]):
        """
        Load a project file to re-analyze the data.

        This method will extract the first manual operations from a project file and
        run them. This is useful when you want to re-analyze the data with a different
        parameter set, or when there were some improvements in cylindra.
        """
        macro = self._get_reanalysis_macro(path)
        return macro.eval({mk.symbol(self): self})

    @Analysis.wraps
    @set_design(text="Open spectra measurer")
    @do_not_record
    def open_spectra_measurer(self):
        """Open the spectra measurer widget to determine cylindric parameters."""
        if self.tomogram is not None and self.tomogram.n_splines > 0:
            binsize = utils.roundint(self.layer_image.scale[0] / self.tomogram.scale)
            self.spectra_measurer.load_spline(self.SplineControl.num, binsize)
        return self.spectra_measurer.show()

    @Analysis.wraps
    @set_design(text="Open subtomogram analyzer")
    @do_not_record
    def open_subtomogram_analyzer(self):
        """Open the subtomogram analyzer dock widget."""
        return self.sta.show()

    @Analysis.wraps
    @set_design(text="Open batch analyzer")
    @do_not_record
    def open_project_batch_analyzer(self):
        """Open the batch analyzer widget."""
        from .batch import CylindraBatchWidget

        uibatch = CylindraBatchWidget()
        uibatch.native.setParent(self.native, uibatch.native.windowFlags())
        self._batch = uibatch
        uibatch.show()
        return uibatch

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map monomers")
    @bind_key("M")
    def map_monomers(
        self,
        splines: Annotated[
            list[int], {"choices": _get_splines, "widget_type": "Select"}
        ] = (),
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
    ):
        """
        Map points to tubulin molecules using the results of global Fourier transformation.

        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        {orientation}
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        molecules = tomo.map_monomers(i=splines, orientation=orientation)

        _Logger.print_html("<code>map_monomers</code>")
        _added_layers = []
        for i, mol in enumerate(molecules):
            _name = f"Mono-{i}"
            layer = self.add_molecules(mol, _name, source=tomo.splines[splines[i]])
            _added_layers.append(layer)
            _Logger.print(f"{_name!r}: n = {len(mol)}")

        self._need_save = True
        return (
            undo_callback(self._try_removing_layers)
            .with_args(_added_layers)
            .with_redo(self._add_layers_future(_added_layers))
        )

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map centers")
    def map_centers(
        self,
        splines: Annotated[
            list[int], {"choices": _get_splines, "widget_type": "Select"}
        ] = (),
        interval: Annotated[Optional[nm], {"text": "Set to dimer length"}] = None,
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
    ):
        """
        Map molecules along splines. Each molecule is rotated by skew angle.

        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        {interval}
        {orientation}
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        mols = tomo.map_centers(i=splines, interval=interval, orientation=orientation)
        _Logger.print_html("<code>map_centers</code>")
        _added_layers = []
        for i, mol in enumerate(mols):
            _name = f"Center-{i}"
            layer = self.add_molecules(mol, _name, source=tomo.splines[splines[i]])
            _added_layers.append(layer)
            _Logger.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return (
            undo_callback(self._try_removing_layers)
            .with_args(_added_layers)
            .with_redo(self._add_layers_future(_added_layers))
        )

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map alogn PF")
    def map_along_pf(
        self,
        splines: Annotated[
            list[int], {"choices": _get_splines, "widget_type": "Select"}
        ],
        interval: Annotated[Optional[nm], {"text": "Set to dimer length"}] = None,
        angle_offset: Annotated[float, {"max": 360}] = 0.0,
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
    ):
        """
        Map molecules along splines. Each molecule is rotated by skew angle.

        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        {interval}
        {orientation}
        """
        tomo = self.tomogram
        mols = tomo.map_pf_line(
            i=splines,
            interval=interval,
            angle_offset=angle_offset,
            orientation=orientation,
        )
        _Logger.print_html("<code>map_along_PF</code>")
        _added_layers = []
        for i, mol in enumerate(mols):
            _name = f"PF line-{i}"
            layer = self.add_molecules(mol, _name, source=tomo.splines[splines[i]])
            _added_layers.append(layer)
            _Logger.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return (
            undo_callback(self._try_removing_layers)
            .with_args(_added_layers)
            .with_redo(self._add_layers_future(_added_layers))
        )

    @MoleculesMenu.wraps
    @set_design(text="Show orientation")
    def show_orientation(
        self,
        layer: MoleculesLayer,
        orientation: Literal["x", "y", "z"] = "z",
        color: Color = "crimson",
    ):
        """
        Show molecule orientations with a vectors layer.

        Parameters
        ----------
        {layer}
        orientation : "x", "y" or "z", default is "z"
            Which orientation will be shown. "z" is the spline-to-molecule direction,
            "y" is parallel to the spline and "x" is defined by right-handedness.
        color : Color, default is "crimson"
            Vector color shown in viewer.
        """
        mol = layer.molecules
        name = f"{layer.name} {orientation.upper()}-axis"

        vector_data = np.stack([mol.pos, getattr(mol, orientation)], axis=1)

        layer = self.parent_viewer.add_vectors(
            vector_data,
            edge_width=0.3,
            edge_color=[color] * len(mol),
            length=2.4,
            name=name,
        )
        return (
            undo_callback(self._try_removing_layer)
            .with_args(layer)
            .with_redo(self._add_layers_future(layer))
        )

    @MoleculesMenu.wraps
    @set_design(text="Extend molecules")
    def extend_molecules(
        self,
        layer: MoleculesLayer,
        counts: Annotated[
            dict[int, tuple[int, int]],
            {"label": "prepend/append", "widget_type": widget_utils.ProtofilamentEdit},
        ] = {},
    ):
        """
        Extend the existing molecules by linear outerpolation.

        Parameters
        ----------
        {layer}
        counts : list of (int, (int, int))
            List of (PF, (prepend, append)) pairs. For instance, (0, (2, 3)) means that
            two molecules will be prepended and three molecules will be appended to the
            protofilament labeled with 0.
        """
        out = widget_utils.extend_protofilament(layer.molecules, dict(counts))
        name = layer.name + "-extended"
        return self.add_molecules(out, name, source=layer.source_component)

    @MoleculesMenu.Combine.wraps
    @set_design(text="Concatenate molecules")
    def concatenate_molecules(
        self,
        layers: Annotated[
            list[MoleculesLayer],
            {"choices": get_monomer_layers, "widget_type": "Select"},
        ],
        delete_old: bool = True,
    ):
        """
        Concatenate selected molecules and create a new ones.

        Parameters
        ----------
        {layers}
        delete_old : bool, default is True
            Delete the selected source layers after concatenation.
        """
        if len(layers) == 0:
            raise ValueError("No layer selected.")
        all_molecules = Molecules.concat([layer.molecules for layer in layers])
        points = add_molecules(self.parent_viewer, all_molecules, name="Mono-concat")
        if delete_old:
            for layer in layers:
                self.parent_viewer.layers.remove(layer)

        # logging
        layer_names: list[str] = []
        for layer in layers:
            layer.visible = False
            layer_names.append(layer.name)

        _Logger.print_html("<code>concatenate_molecules</code>")
        _Logger.print("Concatenated:", ", ".join(layer_names))
        _Logger.print(f"{points.name!r}: n = {len(all_molecules)}")
        return None

    @MoleculesMenu.Combine.wraps
    @set_design(text="Merge molecule info")
    def merge_molecule_info(
        self, pos: MoleculesLayer, rotation: MoleculesLayer, features: MoleculesLayer
    ):
        """
        Merge molecule info from different molecules.

        Parameters
        ----------
        pos : MoleculesLayer
            Molecules whose positions are used.
        rotation : MoleculesLayer
            Molecules whose rotations are used.
        features : MoleculesLayer
            Molecules whose features are used.
        """
        _pos = pos.molecules
        _rot = rotation.molecules
        _feat = features.molecules
        mole = Molecules(_pos.pos, _rot.rotator, features=_feat.features)
        self.add_molecules(mole, name="Mono-merged", source=pos.source_component)
        return None

    def _get_selected_layer_choice(self, w: Widget) -> list[str]:
        """When the selected layer is changed, update the list of features."""
        try:
            parent = w.parent.parent()
            if parent is None:
                return []
            mgui = parent._magic_widget
            if mgui is None or mgui.layer.value is None:
                return []
            return mgui.layer.value.features.columns
        except Exception as e:
            return []

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Split molecules by feature")
    def split_molecules(
        self,
        layer: MoleculesLayer,
        by: Annotated[str, {"choices": _get_selected_layer_choice}],
        delete_old: bool = False,
    ):
        """Split molecules by a feature column."""
        n_unique = layer.molecules.features[by].n_unique()
        if n_unique > 48:
            raise ValueError(
                f"Too many groups ({n_unique}). Did you choose a float column?"
            )
        for _key, mole in layer.molecules.groupby(by):
            self.add_molecules(
                mole, name=f"{layer.name}_{_key}", source=layer.source_component
            )
        if delete_old:
            self.parent_viewer.layers.remove(layer)
        return None

    @MoleculesMenu.wraps
    @set_design(text="Translate molecules")
    def translate_molecules(
        self,
        layer: MoleculesLayer,
        translation: Annotated[
            tuple[nm, nm, nm],
            {
                "options": {"min": -1000, "max": 1000, "step": 0.1},
                "label": "translation Z, Y, X (nm)",
            },
        ],
        internal: bool = True,
    ):
        """
        Translate molecule coordinates without changing their rotations.

        Parameters
        ----------
        {layer}
        translation : tuple of float
            Translation (nm) of the molecules in (Z, Y, X) order.
        internal : bool, default is True
            If true, the translation is applied to the internal coordinates, i.e. molecules
            with different rotations are translated differently.
        """
        mole = layer.molecules
        if internal:
            out = mole.translate_internal(translation)
            if Mole.position in out.features.columns:
                # update spline position feature
                dy = translation[1]
                out = out.with_features([pl.col(Mole.position) + dy])
        else:
            out = mole.translate(translation)
            if Mole.position in out.features.columns:
                # spline position is not predictable.
                out = out.drop_features([Mole.position])
        name = f"{layer.name}-Shift"
        layer = self.add_molecules(out, name=name, source=layer.source_component)
        return mole

    @MoleculesMenu.Visualize.wraps
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

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Filter molecules")
    def filter_molecules(
        self, layer: MoleculesLayer, predicate: ExprStr.In[POLARS_NAMESPACE]
    ):
        """
        Filter molecules by their features.

        Parameters
        ----------
        {layer}
        predicate : ExprStr
            A polars-style filter predicate, such as `pl.col("pf-id") == 3`
        """
        mole = layer.molecules
        expr = ExprStr(predicate, POLARS_NAMESPACE).eval()
        out = mole.filter(expr)
        name = f"{layer.name}-Filt"
        layer = self.add_molecules(out, name=name, source=layer.source_component)

        return (
            undo_callback(self._try_removing_layer)
            .with_args(layer)
            .with_redo(self._add_layers_future(layer))
        )

    def _get_paint_molecules_choice(self, w=None) -> list[str]:
        # don't use get_function_gui. It causes RecursionError.
        gui = self["paint_molecules"].mgui
        if gui is None or gui.layer.value is None:
            return []
        return gui.layer.value.features.columns

    @MoleculesMenu.Visualize.wraps
    @set_design(text="Paint molecules by features")
    def paint_molecules(
        self,
        layer: MoleculesLayer,
        color_by: Annotated[str, {"choices": _get_paint_molecules_choice}],
        cmap: ColormapType = DEFAULT_COLORMAP,
        limits: Annotated[
            tuple[float, float],
            {"options": {"min": -20, "max": 20, "step": 0.01}, "label": "limits (nm)"},
        ] = (4.00, 4.24),
    ):
        """
        Paint molecules by a feature.

        Parameters
        ----------
        {layer}
        color_by : str
            Name of the feature to paint by.
        cmap : ColormapType, default is "hot"-like colormap
            Colormap to use for painting.
        limits : tuple of float
            Limits for the colormap.
        """
        layer.set_colormap(color_by, limits, cmap)
        info = layer.colormap_info
        return undo_callback(layer.set_colormap).with_args(
            name=info.name, clim=info.clim, cmap_input=info.cmap
        )

    @MoleculesMenu.Visualize.wraps
    @set_design(text="Plot molecule feature in 2D")
    def plot_molecule_feature(
        self,
        layer: MoleculesLayer,
        backend: Literal["inline", "qt"] = "inline",
    ):
        """
        Plot current molecule feature coloring in 2D figure.

        For data visualization, plotting in 2D is better than in 3D. Current
        colormap in the 3D canvas is directly used for 2D plotting.

        Parameters
        ----------
        {layer}
        backend : "inline" or "qt", optional
            Plotting backend. "inline" means the plot is shown in the console.
        """
        from matplotlib.patches import Circle
        from matplotlib.axes import Axes

        mole = layer.molecules
        nth = mole.features[Mole.nth].to_numpy()
        pf = mole.features[Mole.pf].to_numpy()
        npf = int(pf.max() + 1)
        if isinstance(spl := layer.source_component, CylSpline):
            props = spl.globalprops
            spacing = props[H.spacing][0]
            rise = np.deg2rad(props[H.rise][0])
            tan = np.tan(rise) / spacing * (2 * np.pi * spl.radius / npf)
        else:
            _, _, nrise = utils.infer_geometry_from_molecules(mole)
            tan = nrise / npf
        y = nth + tan * pf

        face_color = layer.face_color
        if backend == "inline":
            plt.figure()
            ax: Axes = plt.gca()
        elif backend == "qt":
            from magicclass.widgets import Figure

            fig = Figure()
            ax = fig.ax
            fig.show()
            self._active_widgets.add(fig)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        for i in range(mole.count()):
            center = (pf[i], y[i])
            circ = Circle(center, 0.5, fc=face_color[i], ec="black", lw=0.1)
            ax.add_patch(circ)
        ax.set_xlim(pf.min() - 0.6, pf.max() + 0.6)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.set_aspect("equal")
        return undo_callback(lambda: _Logger.print("Undoing plotting does nothing"))

    @MoleculesMenu.Visualize.wraps
    @set_design(text="Show colorbar")
    @do_not_record
    def show_molecules_colorbar(
        self,
        layer: MoleculesLayer,
        length: Annotated[int, {"min": 16}] = 256,
        orientation: Literal["vertical", "horizontal"] = "horizontal",
    ):
        """
        Show the colorbar of the molecules layer in the logger.

        Parameters
        ----------
        {layer}
        length : int, default is 256
            Length of the colorbar.
        orientation : 'vertical' or 'horizontal', default is 'horizontal'
            Orientation of the colorbar.
        """
        info = layer.colormap_info
        colors = info.cmap.map(np.linspace(0, 1, length))
        cmap_arr = np.stack([colors] * (length // 12), axis=0)
        xmin, xmax = info.clim
        with _Logger.set_plt(rc_context={"font.size": 15}):
            if orientation == "vertical":
                plt.imshow(np.swapaxis(cmap_arr, 0, 1))
                plt.xticks([], [])
                plt.yticks([0, length - 1], [f"{xmin:.2f}", f"{xmax:.2f}"])
            else:
                plt.imshow(cmap_arr)
                plt.xticks([0, length - 1], [f"{xmin:.2f}", f"{xmax:.2f}"])
                plt.yticks([], [])

            plt.show()
        return undo_callback(
            lambda: _Logger.print("Undoing `show_molecules_colorbar` does nothing")
        )

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Calculate molecule features")
    def calculate_molecule_features(
        self,
        layer: MoleculesLayer,
        column_name: str,
        expression: ExprStr.In[POLARS_NAMESPACE],
    ):
        """
        Calculate a new feature from the existing features.

        This method is identical to running ``with_columns`` on the features dataframe
        as a ``polars.DataFrame``. For example,

        >>> ui.calculate_molecule_features(layer, "Y", "pl.col('X') + 1")

        is equivalent to

        >>> layer.features = layer.features.with_columns([(pl.col("X") + 1).alias("Y")])

        Parameters
        ----------
        {layer}
        column_name : str
            Name of the new column.
        expression : pl.Expr or str
            polars expression to calculate the new column.
        """
        feat = layer.molecules.features
        if column_name in feat.columns:
            raise ValueError(f"Column {column_name} already exists.")
        pl_expr = eval(str(expression), POLARS_NAMESPACE, {})
        if isinstance(pl_expr, pl.Expr):
            new_feat = feat.with_columns(pl_expr.alias(column_name))
        else:
            new_feat = feat.with_columns(pl.Series(column_name, pl_expr))
        layer.features = new_feat
        self.reset_choices()  # choices regarding of features need update
        return undo_callback(_set_layer_feature_future(layer, feat))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Calculate intervals")
    def calculate_intervals(self, layer: MoleculesLayer):
        """
        Calculate projective intervals (in nm) between adjacent molecules.

        The "projective interval" is defined by the component of the vector of
        adjacent molecules parallel to the vector of the spline curve corresponding
        to the position of the molecule. Please note that this quantity does not
        consider the orientation of each molecule.

        Parameters
        ----------
        {layer}
        """
        if layer.source_component is None:
            raise ValueError(f"Cannot find the source spline of layer {layer.name!r}.")
        layer.features = utils.with_interval(layer.molecules, layer.source_component)
        self.reset_choices()  # choices regarding of features need update

        # Set colormap
        _clim = [GVar.spacing_min, GVar.spacing_max]
        layer.set_colormap(Mole.interval, _clim, DEFAULT_COLORMAP)
        self._need_save = True
        return None

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Calculate skews")
    def calculate_skews(self, layer: MoleculesLayer):
        """
        Calculate projective skew angles (in degree) between adjacent molecules.

        The "projective angle" is defined by the component of the vector of
        adjacent molecules perpendicular to the vector from the spline curve to
        the molecule, and also perpendicular to the vector of the spline curve
        corresponding to the position of the molecule. Please note that this
        quantity does not consider the orientation of each molecule.

        Parameters
        ----------
        {layer}
        """
        if layer.source_component is None:
            raise ValueError(f"Cannot find the source spline of layer {layer.name!r}.")
        layer.features = utils.with_skew(layer.molecules, layer.source_component)
        self.reset_choices()  # choices regarding of features need update
        extreme = np.max(np.abs(layer.features[Mole.skew]))

        # Set colormap
        _clim = [-extreme, extreme]
        layer.set_colormap(
            Mole.skew, _clim, Colormap(["#2659FF", "#FFDBFE", "#FF6C6C"])
        )
        self._need_save = True
        return None

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Seam search by feature")
    def seam_search_by_feature(
        self,
        layer: MoleculesLayer,
        feature_name: Annotated[str, {"choices": _get_selected_layer_choice}],
    ):
        """
        Search for seams by a feature.

        Parameters
        ----------
        {layer}
        feature_name : str
            Name of the feature that will be used for seam search.
        """
        feat = layer.features
        if feature_name not in feat.columns:
            raise ValueError(f"Column {feature_name} does not exist.")
        npf = utils.roundint(layer.molecules.features[Mole.pf].max() + 1)
        seam = utils.infer_seam_from_labels(feat[feature_name], npf=npf)
        _id = np.arange(len(feat))
        res = (_id - seam) // npf
        layer.features = layer.molecules.features.with_columns(
            pl.Series(Mole.isotype, res % 2)
        )
        return undo_callback(_set_layer_feature_future(layer, feat))

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "pick_next.svg")
    @bind_key("F3")
    @do_not_record
    def pick_next(self):
        """Automatically pick cylinder center using previous two points."""
        picker = self.toolbar.Adjust._get_picker()
        points = self.layer_work.data
        if len(points) < 2:
            raise IndexError("Auto picking needs at least two points.")
        imgb: ip.ImgArray = self.layer_image.data
        scale = imgb.scale.x
        next_point = picker.iter_pick(imgb, points[-1], points[-2]).next()
        self.layer_work.add(next_point)

        change_viewer_focus(self.parent_viewer, next_point / scale, scale)
        return None

    @ImageMenu.wraps
    @thread_worker.with_progress(desc="Paint cylinders ...")
    @set_design(text="Paint cylinders")
    def paint_cylinders(
        self,
        color_by: Annotated[str, {"choices": [H.spacing, H.skew, H.rise, H.nPF]}] = H.spacing,
        cmap: ColormapType = DEFAULT_COLORMAP,
        limits: Optional[tuple[float, float]] = (GVar.spacing_min, GVar.spacing_max),
    ):  # fmt: skip
        """
        Paint cylinder fragments by its local properties.

        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """
        if self._current_ft_size is None:
            raise ValueError(
                "Local structural parameters have not been determined yet."
            )

        color: dict[int, list[float]] = {0: [0, 0, 0, 0]}
        tomo = self.tomogram
        all_df = tomo.collect_localprops()
        if color_by not in all_df.columns:
            raise ValueError(f"Column {color_by} does not exist.")

        paint_device = widget_utils.PaintDevice(
            self.layer_image.data.shape, self.layer_image.scale[-1]
        )
        lbl = yield from paint_device.paint_cylinders(
            self._current_ft_size, self.tomogram
        )

        # Labels layer properties
        _id = "ID"
        _str = "structure"
        columns = [_id, H.rise, H.spacing, H.skew, _str]
        df = (
            all_df.select([IDName.spline, IDName.pos, H.rise, H.spacing, H.skew, H.nPF, H.start])
            .with_columns(
                pl.format("{}-{}", pl.col(IDName.spline), pl.col(IDName.pos)).alias(_id),
                pl.format("{}_{}", pl.col(H.nPF), pl.col(H.start).round(1)).alias(_str),
                pl.col(H.rise),
                pl.col(H.spacing),
                pl.col(H.skew),
            )
            .to_pandas()
        )  # fmt: skip
        back = pd.DataFrame({c: [np.nan] for c in columns})
        props = pd.concat([back, df[columns]], ignore_index=True)
        if limits is None:
            limits = float(all_df[color_by].min()), float(all_df[color_by].max())

        @thread_worker.to_callback
        def _on_return():
            # Add labels layer
            if self.layer_paint is None:
                layer_paint = CylinderLabels(
                    lbl,
                    color=color,
                    scale=self.layer_image.scale,
                    translate=self.layer_image.translate,
                    opacity=0.33,
                    name="Cylinder properties",
                    features=props,
                )
                self.layer_paint = self.parent_viewer.add_layer(layer_paint)
            else:
                self.layer_paint.data = lbl
                self.layer_paint.features = props
            self.layer_paint.set_colormap(color_by, limits, cmap)
            return undo_callback(lambda: None)  # TODO: undo paint

        return _on_return

    @ImageMenu.wraps
    @set_design(text="Back-paint molecules")
    @dask_thread_worker.with_progress(desc="Back-painting molecules ...")
    def backpaint_molecule_density(
        self,
        layers: Annotated[list[MoleculesLayer], {"choices": get_monomer_layers, "widget_type": "Select"}],
        template_path: Path.Read[FileFilter.IMAGE],
        target_layer: Annotated[Optional[Image], {"text": "Create a new layer"}] = None,
    ):  # fmt: skip
        """
        Simulate an image using selected molecules.

        Parameters
        ----------
        {layers}
        template_path : path-like
            Path to the template image.
        target_layer : Image, optional
            If given, this layer will be over-painted by the simulated image.
        """
        from acryo.pipe import from_file

        molecules = []
        for layer in layers:
            layer: MoleculesLayer
            molecules.append(layer.molecules)
        mole = Molecules.concat(molecules)
        data = self.layer_image.data if target_layer is None else target_layer.data

        device = widget_utils.PaintDevice(data.shape, self.layer_image.scale[-1])
        template = from_file(template_path)(device.scale)
        sim = device.paint_molecules(template, mole)

        @thread_worker.to_callback
        def _on_return():
            if target_layer is None:
                layer = self.parent_viewer.add_image(
                    sim,
                    scale=self.layer_image.scale,
                    translate=self.layer_image.translate,
                    name="Simulated",
                )
                return (
                    undo_callback(self._try_removing_layer)
                    .with_args(layer)
                    .with_redo(self._add_layers_future(layer))
                )

            else:
                target_layer.data = new_data = data + sim

                @undo_callback
                def out():
                    target_layer.data = data

                @out.with_redo
                def out():
                    target_layer.data = new_data

                return out

        return _on_return

    @ImageMenu.wraps
    @set_design(text="Set colormap")
    def set_colormap(
        self,
        color_by: Annotated[str, {"choices": [H.spacing, H.skew, H.nPF, H.rise]}] = H.spacing,
        cmap: ColormapType = DEFAULT_COLORMAP,
        limits: Annotated[tuple[float, float], {"options": {"min": -20, "max": 20, "step": 0.01}, "label": "limits (nm)"}] = (4.00, 4.24),
    ):  # fmt: skip
        """
        Set the color-map for painting cylinders.

        Parameters
        ----------
        color_by : str, default is "yPitch"
            Select what property image will be colored by.
        cmap : colormap type, default is "hot"-like colormap
            Linear colormap input.
        limits : tuple, default is (4.00, 4.24)
            Color limits (nm).
        """
        self.layer_paint.set_colormap(color_by, limits, cmap)
        return None

    @ImageMenu.wraps
    @set_design(text="Show colorbar")
    @do_not_record
    def show_colorbar(self, layer: ColoredLayer):
        """Create a colorbar from the current colormap."""
        info = layer.colormap_info
        arr = np.stack([info.cmap.map(np.linspace(0, 1, 256))] * 36, axis=0)
        xmin, xmax = info.clim
        with _Logger.set_plt(rc_context={"font.size": 15}):
            plt.imshow(arr)
            plt.xticks([0, arr.shape[1] - 1], [f"{xmin:.2f}", f"{xmax:.2f}"])
            plt.yticks([], [])
            plt.show()
        return None

    @ImageMenu.wraps
    @set_design(text="Simulate cylindric structure")
    @do_not_record
    def open_simulator(self):
        """Open the simulator widget."""
        return self.cylinder_simulator.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Non-GUI methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @nogui
    @do_not_record
    def get_molecules(self, name: "str | None" = None) -> Molecules:
        """
        Retrieve Molecules object from layer list.

        Parameters
        ----------
        name : str, optional
            Name of the molecules layer. If not given, the most recent molecules object
            will be returned.

        Returns
        -------
        Molecules
            The ``Molecules`` object.
        """
        if name is None:
            # Return the most recent molecules object
            for layer in reversed(self.parent_viewer.layers):
                if isinstance(layer, MoleculesLayer):
                    name = layer.name
                    break
            else:
                raise ValueError("No molecules found in the layer list.")
        layer: MoleculesLayer = self.parent_viewer.layers[name]
        return layer.molecules

    @nogui
    @do_not_record
    def add_molecules(
        self,
        molecules: Molecules,
        name: "str | None" = None,
        source: "BaseComponent | None" = None,
    ) -> MoleculesLayer:
        """Add molecules as a points layer to the viewer."""
        return add_molecules(self.parent_viewer, molecules, name, source=source)

    @nogui
    @do_not_record
    def get_loader(
        self,
        name: "str | None" = None,
        order: int = 1,
    ) -> SubtomogramLoader:
        """
        Create a subtomogram loader using current tomogram and a molecules layer.

        Parameters
        ----------
        name : str, optional
            Name of the molecules layer.
        order : int, default is 1
            Interpolation order of the subtomogram loader.
        """
        mole = self.get_molecules(name)
        shape = self.sta._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(mole, shape, order=order)
        return loader

    @nogui
    @do_not_record
    def get_spline(self, i: "int | None" = None) -> CylSpline:
        """Get the i-th spline object. Return current one by default."""
        tomo = self.tomogram
        if i is None:
            i = self.SplineControl.num
        return tomo.splines[i]

    def _init_widget_state(self, _=None):
        """Initialize widget state of spline control and local properties for new plot."""
        self.SplineControl.pos = 0
        self.SplineControl["pos"].max = 0
        self.LocalProperties._init_text()

        for i in range(3):
            del self.SplineControl.canvas[i].image
            self.SplineControl.canvas[i].layers.clear()
            self.SplineControl.canvas[i].text_overlay.text = ""
        for i in range(2):
            self.LocalProperties.plot[i].layers.clear()
        return None

    def _try_removing_layer(self, layer: Layer):
        try:
            self.parent_viewer.layers.remove(layer)
        except ValueError as e:
            _Logger.print(f"ValueError: {e}")
        return None

    def _try_removing_layers(self, layers: list[Layer]):
        for layer in layers:
            self._try_removing_layer(layer)

    def _add_layers_future(self, layers: "Layer | list[Layer]"):
        def future_func():
            nonlocal layers
            if isinstance(layers, Layer):
                layers = [layers]
            for layer in layers:
                self.parent_viewer.add_layer(layer)

        return future_func

    def _send_tomogram_to_viewer(self, filt: bool):
        viewer = self.parent_viewer
        tomo = self.tomogram
        bin_size = max(x[0] for x in tomo.multiscaled)
        imgb = tomo.get_multiscale(bin_size)
        tr = tomo.multiscale_translation(bin_size)
        name = f"{imgb.name} (bin {bin_size})"
        # update image layer
        if self.layer_image not in viewer.layers:
            self.layer_image = viewer.add_image(
                imgb,
                scale=imgb.scale,
                name=name,
                translate=[tr, tr, tr],
                contrast_limits=[np.min(imgb), np.max(imgb)],
            )
        else:
            self.layer_image.data = imgb
            self.layer_image.scale = imgb.scale
            self.layer_image.name = name
            self.layer_image.translate = [tr, tr, tr]
            self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]

        self.layer_image.metadata["current_binsize"] = bin_size
        self.ImageInfo._from_tomogram(tomo)

        # update viewer dimensions
        viewer.scale_bar.unit = imgb.scale_unit
        viewer.dims.axis_labels = ("z", "y", "x")
        change_viewer_focus(viewer, np.asarray(imgb.shape) / 2, imgb.scale.x)

        # update labels layer
        if self.layer_paint is not None:
            self.layer_paint.data = np.zeros(imgb.shape, dtype=np.uint8)
            self.layer_paint.scale = imgb.scale
            self.layer_paint.translate = [tr, tr, tr]

        # update overview
        proj = imgb.proj("z")
        self.overview.image = proj
        self.overview.ylim = (0, proj.shape[0])

        try:
            parts = tomo.source.parts
            if len(parts) > 2:
                _name = ".../" + Path(os.path.join(*parts[-2:])).as_posix()
            else:
                _name = tomo.source.as_posix()
        except Exception:
            _name = f"Tomogram<{hex(id(tomo))}>"
        _Logger.print_html(f"<h2>{_name}</h2>")
        self.clear_all()
        if filt:
            self.filter_reference_image()

    def _on_layer_removing(self, event):
        # NOTE: To make recorded macro completely reproducible, removing molecules
        # from the viewer layer list must always be monitored.
        layer: Layer = self.parent_viewer.layers[event.index]
        if isinstance(layer, MoleculesLayer) and self.macro.active:
            expr = mk.Mock(mk.symbol(self)).parent_viewer.layers[layer.name].expr
            undo = self._add_layers_future(layer)
            self.macro.append_with_undo(mk.Expr("del", [expr]), undo)
        return

    def _on_layer_removed(self, event):
        idx: int = event.index
        layer: Layer = event.value
        if layer in (
            self.layer_image,
            self.layer_prof,
            self.layer_work,
            self.layer_paint,
        ):
            self.parent_viewer.layers.insert(idx, layer)
            warnings.warn(f"Cannot remove layer {layer.name!r}", UserWarning)

    def _on_layer_inserted(self, event):
        layer: Layer = event.value
        layer.events.name.connect(self.reset_choices)

    def _disconnect_layerlist_events(self):
        viewer = self.parent_viewer
        viewer.layers.events.removing.disconnect(self._on_layer_removing)
        viewer.layers.events.removed.disconnect(self._on_layer_removed)
        viewer.layers.events.inserted.disconnect(self._on_layer_inserted)

    def _init_layers(self):
        viewer = self.parent_viewer
        self._disconnect_layerlist_events()

        # remove all the molecules layers
        _layers_to_remove: list[str] = []
        for layer in self.parent_viewer.layers:
            if isinstance(layer, MoleculesLayer):
                _layers_to_remove.append(layer.name)

        for name in _layers_to_remove:
            layer: Layer = self.parent_viewer.layers[name]
            self.parent_viewer.layers.remove(layer)

        common_properties = dict(ndim=3, out_of_slice_display=True, size=8)
        if self.layer_prof in self.parent_viewer.layers:
            viewer.layers.remove(self.layer_prof)

        self.layer_prof: Points = viewer.add_points(
            **common_properties,
            name=SELECTION_LAYER_NAME,
            features={SPLINE_ID: []},
            opacity=0.4,
            edge_color="black",
            face_color="blue",
            text={"color": "yellow"},
        )
        self.layer_prof.feature_defaults[SPLINE_ID] = 0
        self.layer_prof.editable = False

        if self.layer_work in viewer.layers:
            viewer.layers.remove(self.layer_work)

        self.layer_work = viewer.add_points(
            **common_properties,
            name=WORKING_LAYER_NAME,
            face_color="yellow",
            blending="translucent_no_depth",
        )

        self.layer_work.mode = "add"

        if self.layer_paint is not None:
            self.layer_paint.data = np.zeros_like(self.layer_paint.data)
            self.layer_paint.scale = self.layer_image.scale
        self.GlobalProperties._init_text()

        # Connect layer events.
        viewer.layers.events.removing.connect(self._on_layer_removing)
        viewer.layers.events.removed.connect(self._on_layer_removed)
        viewer.layers.events.inserted.connect(self._on_layer_inserted)
        return None

    @SplineControl.num.connect
    def _highlight_spline(self):
        i = self.SplineControl.num
        if i is None:
            return

        for layer in self.overview.layers:
            if f"spline-{i}" in layer.name:
                layer.color = "red"
            else:
                layer.color = "lime"

        spec = self.layer_prof.features[SPLINE_ID] == i
        self.layer_prof.face_color = "blue"
        self.layer_prof.face_color[spec] = [0.8, 0.0, 0.5, 1]
        self.layer_prof.refresh()
        return None

    @SplineControl.num.connect
    def _update_global_properties_in_widget(self, _=None):
        i = self.SplineControl.num
        if i is None:
            return
        spl = self.tomogram.splines[i]
        headers = [H.spacing, H.skew, H.nPF, H.start, H.radius, H.orientation]
        if spl.has_globalprops(headers):
            itv, skew, npf, start, rad, ori = spl.globalprops.select(headers).row(0)
            self.GlobalProperties._set_text(itv, skew, npf, start, rad, ori)
        else:
            self.GlobalProperties._init_text()

    @SplineControl.num.connect
    @SplineControl.pos.connect
    def _update_local_properties_in_widget(self):
        i = self.SplineControl.num
        tomo = self.tomogram
        if i is None or i >= len(tomo.splines):
            return
        j = self.SplineControl.pos
        spl = tomo.splines[i]
        if spl.has_localprops([H.spacing, H.skew, H.nPF, H.start]):
            pitch, skew, npf, start = spl.localprops.select(
                [H.spacing, H.skew, H.nPF, H.start]
            ).row(j)
            self.LocalProperties._set_text(pitch, skew, npf, start)
        else:
            self.LocalProperties._init_plot()
            self.LocalProperties._init_text()
        return None

    def _add_spline_to_images(self, spl: CylSpline, i: int):
        interval = 15
        length = spl.length()
        scale = self.layer_image.scale[0]

        n = max(int(length / interval) + 1, 2)
        fit = spl(np.linspace(0, 1, n))
        self.layer_prof.feature_defaults[SPLINE_ID] = i
        self.layer_prof.add(fit)
        self.overview.add_curve(
            fit[:, 2] / scale,
            fit[:, 1] / scale,
            color="lime",
            lw=2,
            name=f"spline-{i}",
        )
        self._set_orientation_marker(i)
        return None

    def _set_orientation_marker(self, idx: int):
        spline_id = self.layer_prof.features[SPLINE_ID]
        spec = spline_id == idx
        if self.layer_prof.text.string.encoding_type == "ConstantStringEncoding":
            # if text uses constant string encoding, update it to ManualStringEncoding
            string_arr = np.zeros(len(self.layer_prof.data), dtype="<U1")
        else:
            string_arr = np.asarray(self.layer_prof.text.string.array, dtype="<U1")

        spl = self.tomogram.splines[idx]
        str_of_interest = string_arr[spec]

        if spl.orientation == Ori.none:
            str_of_interest[:] = ""
        elif spl.orientation == Ori.MinusToPlus:
            str_of_interest[0], str_of_interest[-1] = "-", "+"
        elif spl.orientation == Ori.PlusToMinus:
            str_of_interest[0], str_of_interest[-1] = "+", "-"
        else:
            raise RuntimeError(spl.orientation)

        # update
        string_arr[spec] = str_of_interest
        self.layer_prof.text.string = list(string_arr)
        return self.layer_prof.refresh()

    def _update_splines_in_images(self, _=None):
        """Refresh splines in overview canvas and napari canvas."""
        self.overview.layers.clear()
        self.layer_prof.data = []
        scale = self.layer_image.scale[0]
        for i, spl in enumerate(self.tomogram.splines):
            self._add_spline_to_images(spl, i)
            if spl._anchors is None:
                continue
            coords = spl()
            self.overview.add_scatter(
                coords[:, 2] / scale,
                coords[:, 1] / scale,
                color="lime",
                symbol="x",
                lw=2,
                size=10,
                name=f"spline-{i}-anc",
            )
        self._highlight_spline()
        return None

    def _global_variable_updated(self):
        """Update GUI states that are related to global variables."""
        get_function_gui(self.global_variables.set_variables).update(GVar.dict())

        fgui = get_function_gui(self.set_spline_props)
        fgui.spacing.min, fgui.spacing.max = GVar.spacing_min, GVar.spacing_max
        fgui.spacing.value = (GVar.spacing_min + GVar.spacing_max) / 2
        fgui.skew.min, fgui.skew.max = GVar.skew_min, GVar.skew_max
        fgui.skew.value = (GVar.skew_min + GVar.skew_max) / 2
        fgui.npf.min, fgui.npf.max = GVar.npf_min, GVar.npf_max
        fgui.npf.value = (GVar.npf_min + GVar.npf_max) // 2

        fgui = get_function_gui(self.cylinder_simulator.update_model)
        fgui.spacing.min, fgui.spacing.max = GVar.spacing_min, GVar.spacing_max
        fgui.spacing.value = (GVar.spacing_min + GVar.spacing_max) / 2
        fgui.skew.min, fgui.skew.max = GVar.skew_min, GVar.skew_max
        fgui.skew.value = (GVar.skew_min + GVar.skew_max) / 2
        fgui.npf.min, fgui.npf.max = GVar.npf_min, GVar.npf_max
        fgui.npf.value = (GVar.npf_min + GVar.npf_max) // 2

        self.cylinder_simulator.parameters.update(
            spacing=fgui.spacing.value,
            skew=fgui.skew.value,
            npf=fgui.npf.value,
        )

        get_function_gui(self.map_monomers)["orientation"].value = GVar.clockwise
        get_function_gui(self.map_along_pf)["orientation"].value = GVar.clockwise
        get_function_gui(self.map_centers)["orientation"].value = GVar.clockwise


############################################################################################
#   Other helper functions
############################################################################################


def _filter_macro_for_reanalysis(macro_expr: mk.Expr, ui_sym: mk.Symbol):
    _manual_operations = {
        "open_image",
        "register_path",
        "load_splines",
        "load_molecules",
        "delete_spline",
        "add_multiscale",
        "set_multiscale",
        "spline_fitter.fit",
    }
    exprs: list[mk.Expr] = []
    breaked_line: "mk.Expr | None" = None
    for line in macro_expr.args:
        if line.head is not mk.Head.call:
            breaked_line = line
            break
        _fn, *_ = line.split_call()
        if _fn.head is not mk.Head.getattr:
            breaked_line = line
            break
        first, *attrs = _fn.split_getattr()
        if first != ui_sym:
            breaked_line = line
            break
        if ".".join(map(str, attrs)) not in _manual_operations:
            breaked_line = line
            break
        exprs.append(line)
    if breaked_line is not None:
        exprs.append(
            mk.Expr(mk.Head.comment, [str(breaked_line) + " ... breaked here."])
        )

    return mk.Expr(mk.Head.block, exprs)


def _set_layer_feature_future(layer: MoleculesLayer, features):
    def _wrapper():
        layer.features = features

    return _wrapper
