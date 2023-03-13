from contextlib import contextmanager
import os
from typing import Annotated, TYPE_CHECKING
import warnings

import impy as ip
import macrokit as mk
import matplotlib.pyplot as plt
import napari
import numpy as np
import polars as pl
import pandas as pd
from acryo import Molecules, SubtomogramLoader
from magicclass import (MagicTemplate, bind_key, build_help, confirm,
                        do_not_record, field, get_function_gui, magicclass, impl_preview, nogui,
                        set_design, set_options, setup_function_gui)
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.ext.polars import DataFrameView
from magicclass.types import Bound, Color, OneOf, Optional, SomeOf, Path, ExprStr
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.widgets import ConsoleTextEdit
from napari.layers import Image, Labels, Layer, Points
from napari.utils.colormaps import Colormap, label_colormap
from scipy import ndimage as ndi

from cylindra import utils
from cylindra.components import CylSpline, CylTomogram
from cylindra.const import (
    SELECTION_LAYER_NAME,
    WORKING_LAYER_NAME, GlobalVariables as GVar,
    IDName, Mode, PropertyNames as H, 
    MoleculesHeader as Mole, Ori, nm, get_versions
)
from cylindra.types import MoleculesLayer, get_monomer_layers
from cylindra.project import CylindraProject

# widgets
from cylindra.widgets import _previews, _shared_doc, subwidgets
from cylindra.widgets.image_processor import ImageProcessor
from cylindra.widgets.properties import GlobalPropertiesWidget, LocalPropertiesWidget
from cylindra.widgets.spline_control import SplineControl
from cylindra.widgets.spline_clipper import SplineClipper
from cylindra.widgets.spline_fitter import SplineFitter
from cylindra.widgets.sta import SubtomogramAveraging
from cylindra.widgets.sweeper import SplineSweeper
from cylindra.widgets.simulator import CylinderSimulator
from cylindra.widgets.measure import SpectraMeasurer
from cylindra.widgets.widget_utils import FileFilter, add_molecules, change_viewer_focus

if TYPE_CHECKING:
    from .batch import CylindraBatchWidget

ICON_DIR = Path(__file__).parent / "icons"
SPLINE_ID = "spline-id"
_Logger = getLogger("cylindra")

# namespace used in predicate
POLARS_NAMESPACE = {"pl": pl, "int": int, "float": float, "str": str, "np": np, "__builtins__": {}}

############################################################################################
#   The Main Widget of cylindra
############################################################################################

@magicclass(widget_type="scrollable", name="cylindra")
@_shared_doc.update_cls
class CylindraMainWidget(MagicTemplate):
    # Main GUI class.

    spline_fitter = field(SplineFitter, name="_Spline fitter")  # Widget for manual spline fitting
    spline_clipper = field(SplineClipper, name="_Spline clipper")  # Widget for manual spline clipping
    spline_sweeper = field(SplineSweeper, name="_Spline sweeper")  # Widget for sweeping along splines
    image_processor = field(ImageProcessor, name="_Image Processor")  # Widget for pre-filtering/pre-processing
    cylinder_simulator = field(CylinderSimulator, name="_Cylinder Simulator")  # Widget for tomogram simulator
    spectra_measurer = field(SpectraMeasurer, name="_FFT Measurer")  # Widget for measuring FFT parameters from a 2D power spectra
    sta = field(SubtomogramAveraging, name="_Subtomogram averaging")  # Widget for subtomogram analysis

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
    Image = subwidgets.Image
    Splines = subwidgets.Splines
    Molecules_ = subwidgets.Molecules_
    Analysis = subwidgets.Analysis
    Others = subwidgets.Others
    
    # Toolbar
    toolbar = subwidgets.toolbar
    
    # Child widgets
    ImageInfo = subwidgets.ImageInfo
    SplineControl = SplineControl  # Widget for controling splines
    LocalProperties = field(LocalPropertiesWidget, name="Local Properties")  # Widget for summary of local properties
    GlobalProperties = field(GlobalPropertiesWidget, name="Global Properties")  # Widget for summary of glocal properties
    overview = field(QtImageCanvas, name="Overview").with_options(tooltip="Overview of splines")  # Widget for 2D overview of splines
    
    ### methods ###
    
    def __init__(self):
        self.tomogram: CylTomogram = None
        self._current_ft_size: nm = 50.
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        self._macro_offset: int = 1
        self._need_save: bool = False
        self._batch = None
        self.objectName()  # load napari types
        
    def __post_init__(self):
        self.set_colormap()
        self.min_width = 400
        self.LocalProperties.collapsed = False
        self.GlobalProperties.collapsed = False
        self.overview.min_height = 300
        
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
    @set_design(icon=ICON_DIR/"add_spline.png")
    @bind_key("F1")
    def register_path(self, coords: Bound[_get_spline_coordinates] = None):
        """Register current selected points as a spline path."""        
        if coords is None:
            coords = self.layer_work.data
        else:
            coords = np.asarray(coords)
        
        if coords.size == 0:
            warnings.warn("No points are in viewer.", UserWarning)
            return None

        tomo = self.tomogram
        tomo.add_spline(coords)
        spl = tomo.splines[-1]
        
        # draw path
        self._add_spline_to_images(spl, tomo.n_splines - 1)
        self.layer_work.data = []
        self.layer_prof.selected_data = set()
        self.reset_choices()
        return None
    
    _runner = subwidgets.Runner
    _image_loader = subwidgets.ImageLoader
    
    @toolbar.wraps
    @set_design(icon=ICON_DIR/"run_all.png")
    @bind_key("F2")
    @do_not_record
    def open_runner(self):
        """Run cylindrical fitting algorithm with various settings."""
        return self._runner.show(run=False)
    
    @_runner.wraps
    @set_design(text="Run")
    @thread_worker.with_progress(
        desc="Running cylindrical fitting", 
        total="(1 + n_refine + int(local_props) + int(global_props)) * len(splines)",
    )
    def cylindrical_fit(
        self,
        splines: Bound[_runner._get_splines_to_run] = (),
        bin_size: Bound[_runner.bin_size] = 1,
        interval: Bound[_runner.params2.interval] = 32.0,
        ft_size: Bound[_runner.params2.ft_size] = 32.0,
        n_refine: Bound[_runner.n_refine] = 1,
        max_shift: Bound[_runner.params1.max_shift] = 5.0,
        edge_sigma: Bound[_runner._get_edge_sigma] = 2.0,
        local_props: Bound[_runner.local_props] = True,
        global_props: Bound[_runner.global_props] = True,
        paint: Bound[_runner.params2.paint] = True,
    ):
        """Run cylindrical fitting."""     
        if self.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if self.tomogram.n_splines == 0:
            raise ValueError("No spline found.")
        elif len(splines) == 0:
            splines = range(self.tomogram.n_splines)
        
        tomo = self.tomogram
        _on_yield = thread_worker.to_callback(self._update_splines_in_images)
        for i_spl in splines:
            tomo.fit(i=i_spl, edge_sigma=edge_sigma, max_shift=max_shift, binsize=bin_size)
            
            for _ in range(n_refine):
                yield _on_yield
                tomo.refine(i=i_spl, max_interval=max(interval, 30), binsize=bin_size)
            tomo.set_radius(i=i_spl, binsize=bin_size)
                
            tomo.make_anchors(i=i_spl, interval=interval)
            if local_props:
                yield _on_yield
                tomo.local_ft_params(i=i_spl, ft_size=ft_size, binsize=bin_size)
            if global_props:
                yield _on_yield
                tomo.global_ft_params(i=i_spl, binsize=bin_size)
            yield _on_yield

        self._current_ft_size = ft_size
        self._need_save = True
        
        @thread_worker.to_callback
        def _cylindrical_fit_on_return():
            if local_props or global_props:
                self.sample_subtomograms()
                if global_props:
                    df = self.tomogram.collect_globalprops(i=splines).to_pandas().transpose()
                    df.columns = [f"Spline-{i}" for i in splines]
                    _Logger.print_table(df, precision=3)
            if local_props and paint:
                self.paint_cylinders()
            if global_props:
                self._update_global_properties_in_widget()
            self._update_splines_in_images()
        
        return _cylindrical_fit_on_return
    
    @cylindrical_fit.started.connect
    def _cylindrical_fit_on_start(self):
        return self._runner.close()

    @toolbar.wraps
    @set_design(icon=ICON_DIR/"clear_last.png")
    @do_not_record
    def clear_current(self):
        """Clear current selection."""
        if self.layer_work.data.size > 0:
            self.layer_work.data = []
        else:
            self.delete_spline(-1)
        
        return None
    
    @toolbar.wraps
    @set_design(icon=ICON_DIR/"clear_all.png")
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
        with open(path, mode="r") as f:
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
        _ui = str(str(mk.symbol(self)))
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
        new.textedit.value = str(self._format_macro()[self._macro_offset:])
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
        
        return exec_command_palette(self)

    @Others.wraps
    @set_design(text="Open logger")
    @do_not_record
    def open_logger(self):
        """Open logger window."""
        return _Logger.widget.show()

    @Others.wraps
    @set_design(text="Clear cache")
    @confirm(text="Are you sure to clear cache?\nYou cannot undo this.")
    def clear_cache(self):
        """Clear cache stored on the current tomogram."""
        if self.tomogram is not None:
            self.tomogram.clear_cache()
        return None
    
    @Others.wraps
    @set_design(text="Send UI to console")
    @do_not_record
    def send_ui_to_console(self, identifier: str = "ui"):
        """Send this widget instance to napari console by any identifier."""
        self.parent_viewer.update_console({identifier: self})
        return None
    
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
    @confirm(text="You may have unsaved data. Open a new tomogram?", condition="self._need_save")
    def open_image(
        self, 
        path: Bound[_image_loader.path],
        scale: Bound[_image_loader.scale.scale_value] = 1.0,
        bin_size: Bound[_image_loader.bin_size] = [1],
        filter: Bound[_image_loader.filter_reference_image] = True,
    ):
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
        img = ip.lazy_imread(path, chunks=GVar.daskChunk)
        if scale is not None:
            scale = float(scale)
            img.scale.x = img.scale.y = img.scale.z = scale
        else:
            scale = img.scale.x
        if isinstance(bin_size, int):
            bin_size = [bin_size]
        elif len(bin_size) == 0:
            raise ValueError("You must specify at least one bin size.")
        bin_size = list(set(bin_size))  # delete duplication
        tomo = CylTomogram.imread(
            path=path,
            scale=scale,
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
    @confirm(text="You may have unsaved data. Open a new project?", condition="self._need_save")
    @do_not_record
    def load_project(self, path: Path.Read[FileFilter.JSON], filter: bool = True):
        """Load a project json file."""
        project = CylindraProject.from_json(path)
        return thread_worker.to_callback(project.to_gui(self, filter=filter))
    
    @File.wraps
    @set_design(text="Save project")
    def save_project(
        self,
        json_path: Path.Save[FileFilter.JSON],
        results_dir: Annotated[Optional[Path.Dir], {"text": "Save at the same directory"}] = None,
    ):
        """
        Save current project state as a json file and the results in a directory.
        
        The json file contains paths of images and results, parameters of splines,
        scales and version. Local and global properties, molecule coordinates and
        features will be exported as csv files. If results are saved at the default
        directory, they will be written as relative paths in the project json file
        so that moving root directory does not affect loading behavior.

        Parameters
        ----------
        json_path : Path
            Path of json file.
        results_dir : Path, optional
            Optionally you can specify the directory to save csv files.
        """
        CylindraProject.save_gui(self, Path(json_path), results_dir)
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
    def save_spline(self, spline: OneOf[_get_splines], save_path: Path.Save[FileFilter.JSON]):
        """Save splines as a json file."""
        spl = self.tomogram.splines[spline]
        spl.to_json(save_path)
        return None
        
    @File.wraps
    @set_design(text="Save molecules")
    def save_molecules(self, layer: MoleculesLayer, save_path: Path.Save[FileFilter.CSV]):
        """
        Save monomer coordinates, orientation and features as a csv file.

        Parameters
        ----------
        {layer}
        save_path : Path
            Where to save the molecules.
        save_features : bool, default is True
            Check if save molecule features.
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

    @Image.wraps
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
                cutoff, chunks=(96, 96, 96), overlap=overlap,
            )
        contrast_limits = np.percentile(self.layer_image.data, [1, 99.9])
        
        @thread_worker.to_callback
        def _filter_reference_image_on_return():
            self.layer_image.contrast_limits = contrast_limits
            proj = self.layer_image.data.proj("z")
            self.overview.image = proj
            self.overview.contrast_limits = contrast_limits
        
        return _filter_reference_image_on_return

    @Image.wraps
    @set_design(text="Add multi-scale")
    @dask_thread_worker.with_progress(desc=lambda bin_size: f"Adding multiscale (bin = {bin_size})")
    def add_multiscale(self, bin_size: OneOf[2:9] = 4):
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
    
    @Image.wraps
    @set_design(text="Set multi-scale")
    def set_multiscale(self, bin_size: OneOf[_get_available_binsize]):
        """
        Set multiscale used for image display.
        
        Parameters
        ----------
        bin_size: int
            Bin size of multiscaled image.
        """
        tomo = self.tomogram
        imgb = tomo.get_multiscale(bin_size)
        factor = self.layer_image.scale[0] / imgb.scale.x
        current_z = self.parent_viewer.dims.current_step[0]
        self.layer_image.data = imgb
        self.layer_image.scale = imgb.scale
        self.layer_image.name = f"{imgb.name} (bin {bin_size})"
        self.layer_image.translate = [tomo.multiscale_translation(bin_size)] * 3
        self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]
        self.parent_viewer.dims.set_current_step(axis=0, value=current_z*factor)
        
        if self.layer_paint is not None:
            self.layer_paint.scale = self.layer_image.scale
            self.layer_paint.translate = self.layer_image.translate
        
        # update overview
        self.overview.image = imgb.proj("z")
        self.overview.xlim = [x*factor for x in self.overview.xlim]
        self.overview.ylim = [y*factor for y in self.overview.ylim]
        self.layer_image.metadata["current_binsize"] = bin_size
        self.reset_choices()
        return None
    
    @Image.wraps
    @do_not_record
    @set_design(text="Open spline sweeper")
    def open_sweeper(self):
        """Open spline sweeper widget"""
        self.spline_sweeper.show()
        return self.spline_sweeper.refresh_widget_state()
        
    @Image.wraps
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
    @set_design(text="Show splines")
    def show_splines(self):
        """Show 3D spline paths of cylinder central axes as a layer."""        
        paths = [r.partition(100) for r in self.tomogram.splines]
        
        self.parent_viewer.add_shapes(
            paths, shape_type="path", edge_color="lime", edge_width=1,
        )
        return None

    @Splines.Orientation.wraps
    @set_design(text="Invert spline")
    def invert_spline(self, spline: Bound[SplineControl.num] = None):
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
        
        need_resample = self.SplineControl.canvas[0].image is not None
        self._init_widget_state()
        if need_resample:
            self.sample_subtomograms()
        self._set_orientation_marker(spline)
        self._need_save = True
        return None
    
    @Splines.Orientation.wraps
    @set_design(text="Align to polarity")
    def align_to_polarity(self, orientation: OneOf["MinusToPlus", "PlusToMinus"] = "MinusToPlus"):
        """
        Align all the splines in the direction parallel to the cylinder polarity.

        Parameters
        ----------
        orientation : Ori, default is Ori.MinusToPlus
            To which direction splines will be aligned.
        """
        need_resample = self.SplineControl.canvas[0].image is not None
        self.tomogram.align_to_polarity(orientation=orientation)
        self._update_splines_in_images()
        self._init_widget_state()
        self.reset_choices()
        if need_resample:
            self.sample_subtomograms()
        for i in range(len(self.tomogram.splines)):
            self._set_orientation_marker(i)
        self._need_save = True
        return None
    
    @Splines.Orientation.wraps
    @set_design(text="Auto-align to polarity")
    @thread_worker.with_progress(desc="Auto-detecting polarities...", total="len(self.tomogram.splines)")
    def auto_align_to_polarity(
        self,
        clockwise_is: OneOf["MinusToPlus", "PlusToMinus"] = "MinusToPlus",
        align_to: Annotated[Optional[OneOf["MinusToPlus", "PlusToMinus"]], {"text": "Do not align"}] = "MinusToPlus",
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
        clockwise_is : Ori, default is Ori.MinusToPlus
            Polarity corresponding to clockwise rotation of the projection image.
        align_to : Ori, default is Ori.MinusToPlus
            To which direction splines will be aligned.
        depth : nm, default is 40 nm
            Depth (Y-length) of the subtomogram to be sampled.
        """
        binsize = self.layer_image.metadata["current_binsize"]
        tomo = self.tomogram
        current_scale = tomo.scale * binsize
        imgb = tomo.get_multiscale(binsize)

        length_px = tomo.nm2pixel(depth, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)
        
        ori_clockwise = Ori(clockwise_is)
        ori_anticlockwise = Ori.invert(ori_clockwise, allow_none=False)
        for i, spl in enumerate(self.tomogram.splines):
            if spl.radius is None:
                r_range = 0.5, width_px / 2
            else:
                r_px = tomo.nm2pixel(spl.radius, binsize=binsize)
                r_range = (GVar.inner * r_px, GVar.outer * r_px)
            point = 0.5
            coords = spl.local_cylindrical(r_range, length_px, point, scale=current_scale)
            mapped = utils.map_coordinates(imgb, coords, order=1, mode=Mode.reflect)
            img_flat = ip.asarray(mapped, axes="rya").proj("y")
            npf = utils.roundint(spl.globalprops[H.nPF][0])
            pw_peak = img_flat.local_power_spectra(
                key=ip.slicer.a[npf-1:npf+2], upsample_factor=20, dims="ra",
            ).proj("a", method=np.max)
            r_argmax = np.argmax(pw_peak)
            clkwise = r_argmax - (pw_peak.size + 1) // 2 > 0
            spl.orientation = ori_clockwise if clkwise else ori_anticlockwise
            
            yield thread_worker.to_callback(_Logger.print, f"Spline {i} was {spl.orientation.name}.")
        
        if align_to is not None:
            return thread_worker.to_callback(self.align_to_polarity, align_to)
        else:
            pass
            # for i in range(len(tomo.splines)):  TODO: make this work
            #     self._set_orientation_marker(i)
        return None
    
    @Splines.wraps
    @set_options(clip_lengths={"options": {"min": 0.0, "max": 1000.0, "step": 0.1, "label": "clip length (nm)"}})
    @set_design(text="Clip splines")
    def clip_spline(self, spline: OneOf[_get_splines], clip_lengths: tuple[nm, nm] = (0., 0.)):
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
        length = spl.length()
        start, stop = np.array(clip_lengths) / length
        self.tomogram.splines[spline] = spl.clip(start, 1 - stop)
        self._update_splines_in_images()
        self._need_save = True
        # current layer will be removed. Select another layer.
        self.parent_viewer.layers.selection = {self.layer_work}
        # initialize clipping values
        fgui = get_function_gui(self, "clip_spline")
        fgui.clip_lengths.value = (0., 0.)
        return None
    
    @impl_preview(clip_spline, auto_call=True)
    def _during_clip_spline(self, spline: int, clip_lengths: tuple[nm, nm]):
        tomo = self.tomogram
        name = "Spline preview"
        spl = self.tomogram.splines[spline]
        length = spl.length()
        start, stop = np.array(clip_lengths) / length
        verts = tomo.splines[spline].clip(start, 1 - stop).partition(100)
        verts_2d = verts[:, 1:]
        viewer = self.parent_viewer
        if name in viewer.layers:
            layer: Layer = viewer.layers[name]
            layer.data = verts_2d
        else:
            layer = viewer.add_shapes(
                verts_2d, shape_type="path", edge_color="crimson", edge_width=3, 
                name=name
            )
        try:
            is_active = yield
        finally:
            if not is_active and layer in viewer.layers:
                viewer.layers.remove(layer)
    
    @Splines.wraps
    @set_design(text="Delete spline")
    def delete_spline(self, i: Bound[SplineControl.num]):
        """Delete currently selected spline."""
        if i < 0:
            i = len(self.tomogram.splines) - 1
        self.tomogram.splines.pop(i)
        self.reset_choices()
        
        # update layer
        features = self.layer_prof.features
        spline_id = features[SPLINE_ID]
        spec = spline_id != i
        self.layer_prof.data = self.layer_prof.data[spec]
        new_features = features[spec].copy()
        spline_id = np.asarray(new_features[SPLINE_ID])
        spline_id[spline_id >= i] -= 1
        new_features[SPLINE_ID] = spline_id
        self._update_splines_in_images()
        self.layer_prof.features = new_features
        self.layer_prof.feature_defaults[SPLINE_ID] = len(self.tomogram.splines)
        need_resample = self.SplineControl.canvas[0].image is not None
        if need_resample:
            self.sample_subtomograms()
        self._need_save = True
        return None
        
    @Splines.wraps
    @set_design(text="Fit splines")
    @thread_worker.with_progress(desc="Spline Fitting")
    def fit_splines(
        self, 
        max_interval: Annotated[nm, {"label": "Max interval (nm)"}] = 30,
        bin_size: OneOf[_get_available_binsize] = 1,
        degree_precision: float = 0.5,
        edge_sigma: Annotated[Optional[nm], {"text": "Do not mask image"}] = 2.0,
        max_shift: nm = 5.0,
    ):
        """
        Fit cylinder with spline curve, using manually selected points.

        Parameters
        ----------
        {max_interval}{bin_size}
        degree_precision : float, default is 0.5
            Precision of xy-tilt degree in angular correlation.
        edge_sigma : bool, default is False
            Check if cylindric structures are densely packed. Initial spline position must 
            be "almost" fitted in dense mode.
        max_shift : nm, default is 5.0
            Maximum shift to be applied to each point of splines.
        """        
        self.tomogram.fit(
            max_interval=max_interval,
            binsize=bin_size,
            degree_precision=degree_precision,
            edge_sigma=edge_sigma,
            max_shift=max_shift,
        )
        self._need_save = True
        return thread_worker.to_callback(self._update_splines_in_images)

    @Splines.wraps
    @set_design(text="Fit splines manually")
    @do_not_record
    def fit_splines_manually(self, max_interval: Annotated[nm, {"label": "Max interval (nm)"}] = 50.0):
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
    def add_anchors(self, interval: Annotated[nm, {"label": "Interval between anchors (nm)", "min": 1.0}] = 25.0):
        """
        Add anchors to splines.

        Parameters
        ----------
        {interval}
        """        
        tomo = self.tomogram
        if tomo.n_splines == 0:
            raise ValueError("Cannot add anchors before adding splines.")
        for i in range(tomo.n_splines):
            tomo.make_anchors(i, interval=interval)
        self._update_splines_in_images()
        self._need_save = True
        return None

    @Analysis.wraps
    @set_design(text="Set radius")
    @thread_worker.with_progress(desc="Measuring Radius")
    def set_radius(
        self,
        radius: Annotated[Optional[nm], {"text": "Measure radii by radial profile."}] = None,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """Measure cylinder radius for each spline path."""        
        self.tomogram.set_radius(radius=radius, binsize=bin_size)
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_design(text="Refine splines")
    @thread_worker.with_progress(desc="Refining splines")
    def refine_splines(
        self,
        max_interval: Annotated[nm, {"label": "Maximum interval (nm)"}] = 30,
        corr_allowed: Annotated[float, {"label": "Correlation allowed", "max": 1.0, "step": 0.1}] = 0.9,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Refine splines using the global cylindric structural parameters.
        
        Parameters
        ----------
        {max_interval}
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        {bin_size}
        """
        tomo = self.tomogram
        
        tomo.refine(
            max_interval=max_interval,
            corr_allowed=corr_allowed,
            binsize=bin_size,
        )
        
        self._need_save = True
        
        @thread_worker.to_callback
        def _refine_splines_on_return():
            self._init_widget_state()
            self._update_splines_in_images()
            
        return _refine_splines_on_return
    
    @Splines.wraps
    @set_design(text="Molecules to spline")
    @confirm(
        text="The existing splines will be removed.\nDo you want to run?",
        condition="len(self.SplineControl._get_splines()) > 0",
    )
    def molecules_to_spline(
        self, 
        layers: SomeOf[get_monomer_layers],
        interval: Annotated[nm, {"label": "Interval (nm)", "min": 1.0}] = 24.5,
    ):
        """
        Create splines from molecules.
        
        This function is useful to refine splines using results of subtomogram 
        alignment. Note that this function only works with molecules that is
        correctly assembled by such as :func:`map_monomers`.

        Parameters
        ----------
        {layers}{interval}
        """        
        splines: list[CylSpline] = []
        for layer in layers:
            spl = utils.molecules_to_spline(layer)
            splines.append(spl)
        
        self.tomogram.splines.clear()
        self.tomogram.splines.extend(splines)
        self.tomogram.make_anchors(interval=interval)
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
    @thread_worker.with_progress(desc="Local Fourier transform", total="self.tomogram.n_splines")
    def local_ft_analysis(
        self,
        interval: Annotated[nm, {"min": 1.0, "step": 0.5}] = 24.5,
        ft_size: Annotated[nm, {"min": 2.0, "step": 0.5}] = 24.5,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Determine cylindrical structural parameters by local Fourier transformation.

        Parameters
        ----------
        {interval}
        ft_size : nm, default is 32.0
            Longitudinal length of local discrete Fourier transformation used for 
            structural analysis.
        {bin_size}
        """
        tomo = self.tomogram
        if tomo.splines[0].radius is None:
            self.tomogram.set_radius()
        tomo.make_anchors(interval=interval)
        
        @thread_worker.to_callback
        def _local_ft_analysis_on_yield(i: int):
            if i == 0:
                self.sample_subtomograms()        
            self._update_splines_in_images()
            self._update_local_properties_in_widget()
        
        for i in range(self.tomogram.n_splines):
            tomo.local_ft_params(i=i, ft_size=ft_size, binsize=bin_size)
            yield _local_ft_analysis_on_yield(i)
        self._current_ft_size = ft_size
        self._need_save = True
        return None

    @Analysis.wraps
    @set_design(text="Global FT analysis")
    @thread_worker.with_progress(desc="Global Fourier transform", total="self.tomogram.n_splines")
    def global_ft_analysis(self, bin_size: OneOf[_get_available_binsize] = 1):
        """
        Determine cylindrical global structural parameters by Fourier transformation.
        
        Parameters
        ----------
        {bin_size}
        """
        tomo = self.tomogram
        if self.tomogram.splines[0].radius is None:
            self.tomogram.set_radius()
            
        @thread_worker.to_callback
        def _global_ft_analysis_on_yield(i: int):
            if i == 0:
                self.sample_subtomograms()
            self._update_splines_in_images()
            self._update_local_properties_in_widget()
        
        @thread_worker.to_callback
        def _global_ft_analysis_on_return():
            df = self.tomogram.collect_globalprops().to_pandas().transpose()
            df.columns = [f"Spline-{i}" for i in range(len(df.columns))]
            _Logger.print_table(df, precision=3)
            self._update_global_properties_in_widget()
        
        for i in range(self.tomogram.n_splines):
            tomo.global_ft_params(i=i, binsize=bin_size)
            yield _global_ft_analysis_on_yield(i)
        self._need_save = True
        return _global_ft_analysis_on_return
    
    def _get_reanalysis_macro(self, path: Path):
        _manual_operations = {
            "open_image", "register_path", "load_splines", "load_molecules",
            "delete_spline", "add_multiscale", "set_multiscale",
        }
        _ui_sym = mk.symbol(self)
        project = CylindraProject.from_json(path)
        macro_path = Path(project.macro)
        macro_expr = mk.parse(macro_path.read_text())
        exprs: list[mk.Expr] = []
        breaked_line: "mk.Expr | None" = None
        for line in macro_expr.args:
            if line.head is not mk.Head.call:
                breaked_line  = line
                break
            _fn, *_ = line.split_call()
            if _fn.head is not mk.Head.getattr:
                breaked_line  = line
                break
            first, *attrs = _fn.split_getattr()
            if first != _ui_sym or len(attrs) != 1:
                breaked_line  = line
                break
            if str(attrs[0]) not in _manual_operations:
                breaked_line  = line
                break
            exprs.append(line)
        if breaked_line is not None:
            exprs.append(mk.Expr(mk.Head.comment, [str(breaked_line) + " ... breaked here."]))
        
        return mk.Expr(mk.Head.block, exprs)

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
    
    @impl_preview(load_project_for_reanalysis, text="Preview extracted code")
    def _preview_load_project_for_reanalysis(self, path: Path):
        macro = self._get_reanalysis_macro(path)
        w = ConsoleTextEdit(value=str(macro))
        w.syntax_highlight("python")
        w.read_only = True
        w.native.setParent(self.native, w.native.windowFlags())
        w.show()
        return None

    @Analysis.wraps
    @set_design(text="Open spectra measurer")
    @thread_worker.with_progress(desc="Calculating power spectra")
    @do_not_record
    def open_spectra_measurer(self):
        """Open the spectra measurer widget to determine cylindric parameters."""
        if self.tomogram.n_splines > 0:
            self.spectra_measurer.load_spline(self.SplineControl.num)
        return thread_worker.to_callback(self.spectra_measurer.show)
    
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
    
    @Molecules_.Mapping.wraps
    @set_design(text="Map monomers")
    @bind_key("M")
    def map_monomers(
        self,
        splines: SomeOf[_get_splines] = (),
        orientation: OneOf[None, "PlusToMinus", "MinusToPlus"] = None,
    ):
        """
        Map points to tubulin molecules using the results of global Fourier transformation.
        
        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        invert : bool
            Whether to map monomers in the inverted direction.
        {orientation}
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        molecules = tomo.map_monomers(i=splines, orientation=orientation)
        
        _Logger.print_html("<code>map_monomers</code>")
        for i, mol in enumerate(molecules):
            _name = f"Mono-{i}"
            self.add_molecules(mol, _name)
            _Logger.print(f"{_name!r}: n = {len(mol)}")
            
        self._need_save = True
        return molecules

    @Molecules_.Mapping.wraps
    @set_design(text="Map centers")
    def map_centers(
        self,
        splines: SomeOf[_get_splines] = (),
        interval: Annotated[Optional[nm], {"text": "Set to dimer length"}] = None,
        orientation: OneOf[None, "PlusToMinus", "MinusToPlus"] = None,
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
        for i, mol in enumerate(mols):
            _name = f"Center-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            _Logger.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return None
    
    @Molecules_.Mapping.wraps
    @set_design(text="Map alogn PF")
    def map_along_pf(
        self,
        splines: SomeOf[_get_splines],
        interval: Annotated[Optional[nm], {"text": "Set to dimer length"}] = None,
        angle_offset: Annotated[float, {"max": 360}] = 0.0,
        orientation: OneOf[None, "PlusToMinus", "MinusToPlus"] = None,
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
        mols = tomo.map_pf_line(i=splines, interval=interval, angle_offset=angle_offset, orientation=orientation)
        _Logger.print_html("<code>map_along_PF</code>")
        for i, mol in enumerate(mols):
            _name = f"PF line-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            _Logger.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return None

    @Molecules_.wraps
    @set_design(text="Show orientation")
    def show_orientation(
        self,
        layer: MoleculesLayer,
        orientation: OneOf["x", "y", "z"] = "z",
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
        
        self.parent_viewer.add_vectors(
            vector_data,
            edge_width=0.3,
            edge_color=[color] * len(mol), 
            length=2.4,
            name=name,
        )
        return None

    @Molecules_.wraps
    @set_options(auto_call=True)
    @set_design(text="Extend molecules")
    def extend_molecules(
        self,
        layer: MoleculesLayer,
        prepend: int = 0,
        append: int = 0,
    ):
        """
        Extend the existing molecules at the edges.
        
        Parameters
        ----------
        {layer}
        prepend : int, default is 0
            Number of molecules to be prepended for each protofilament.
        append : int, default is 0
            Number of molecules to be appended for each protofilament.
        """        
        from scipy.spatial.transform import Rotation

        ndim = 3
        mole = layer.molecules
        npf = utils.roundint(mole.features[Mole.pf].max() + 1)
        pos = mole.pos.reshape(-1, npf, ndim)
        quat = mole.rotator.as_quat()
        if prepend > 0:
            dvec_pre = pos[0] - pos[1]
            pos_pre = np.stack(
                [pos[0] + dvec_pre * n for n in range(1, prepend + 1)], axis=0
            )
            quat_pre = np.concatenate([quat[:npf]] * prepend, axis=0)
        else:
            pos_pre = np.zeros((0, npf, ndim), dtype=np.float32)
            quat_pre = np.zeros((0, 4), dtype=np.float32)
        if append > 0:
            dvec_post = pos[-1] - pos[-2]
            pos_post = np.stack(
                [pos[-1] + dvec_post * n for n in range(1 + append + 1)], axis=0
            )
            quat_post = np.concatenate([quat[-npf:]] * append, axis=0)
        else:
            pos_post = np.zeros((0, npf, ndim), dtype=np.float32)
            quat_post = np.zeros((0, 4), dtype=np.float32)
        
        pos_extended: np.ndarray = np.concatenate([pos_pre, pos, pos_post], axis=0)
        quat_extended = np.concatenate([quat_pre, quat, quat_post], axis=0)
        features = {Mole.pf: pl.Series(np.arange(len(mole_new), dtype=np.uint32)) % npf}
        mole_new = Molecules(
            pos_extended.reshape(-1, ndim),
            Rotation.from_quat(quat_extended),
            features=features,
        )
        
        name = layer.name + "-extended"
        viewer = self.parent_viewer
        if name not in viewer.layers:
            points_layer = add_molecules(self.parent_viewer, mole_new, name)
            layer.visible = False
        else:
            points_layer: Layer = viewer.layers[name]
            if not isinstance(points_layer, MoleculesLayer):
                points_layer.name = points_layer.name + "-0"
            points_layer.molecules = mole_new
            points_layer.selected_data = set()
        return None
    
    @Molecules_.Combine.wraps
    @set_design(text="Concatenate molecules")
    def concatenate_molecules(self, layers: SomeOf[get_monomer_layers], delete_old: bool = True):
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

    @Molecules_.Combine.wraps
    @set_design(text="Merge molecule info")
    def merge_molecule_info(self, pos: MoleculesLayer, rotation: MoleculesLayer, features: MoleculesLayer):
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
        self.add_molecules(mole, name="Mono-merged")
    
    def _get_selected_layer_choice(self, w=None) -> list[str]:
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

    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Split molecules by feature")
    def split_molecules(
        self,
        layer: MoleculesLayer,
        by: OneOf[_get_selected_layer_choice], 
        delete_old: bool = False,
    ):
        """Split molecules by a feature column."""
        n_unique = layer.molecules.features[by].n_unique()
        if n_unique > 48:
            raise ValueError(f"Too many groups ({n_unique}). Did you choose a float column?")
        for _key, mole in layer.molecules.groupby(by):
            self.add_molecules(mole, name=f"{layer.name}_{_key}")
        if delete_old:
            self.parent_viewer.layers.remove(layer)
    
    @impl_preview(split_molecules, auto_call=True)
    def _during_split_molecules_preview(self, layer: MoleculesLayer, by: str):
        with _temp_layer_colors(layer):
            series = layer.molecules.features[by]
            unique_values = series.unique()
            # NOTE: the first color is translucent
            cmap = label_colormap(unique_values.len() + 1, seed=0.9414)
            layer.face_color_cycle = layer.edge_color_cycle = cmap.colors[1:]
            layer.face_color = layer.edge_color = by
            yield

    @Molecules_.wraps
    @set_design(text="Translate molecules")
    def translate_molecules(
        self,
        layer: MoleculesLayer,
        translation: Annotated[
            tuple[nm, nm, nm],
            {"options": {"min": -1000, "max": 1000, "step": 0.1}, "label": "translation Z, Y, X (nm)"}
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
        else:
            out = mole.translate(translation)
        name = f"{layer.name}-Shift"
        layer = self.add_molecules(out, name=name)
        return mole
        
    @impl_preview(translate_molecules, auto_call=True)
    def _during_translate_molecules(self, layer: MoleculesLayer, translation, internal: bool):
        mole = layer.molecules
        if internal:
            out = mole.translate_internal(translation)
        else:
            out = mole.translate(translation)
        viewer = self.parent_viewer
        name = "<Preview>"
        if name in viewer.layers:
            layer: Layer = viewer.layers[name]
            layer.data = out.pos
        else:
            layer = self.add_molecules(out, name=name)
            layer.face_color = layer.edge_color = "crimson"
        try:
            is_active = yield
        finally:
            if not is_active and layer in viewer.layers:
                viewer.layers.remove(layer)
        return out

    @Molecules_.MoleculeFeatures.wraps
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
    
    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Filter molecules")
    def filter_molecules(
        self,
        layer: MoleculesLayer,
        predicate: ExprStr.In[POLARS_NAMESPACE]
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
        expr = eval(str(predicate), POLARS_NAMESPACE, {})
        out = mole.filter(expr)
        name = f"{layer.name}-Filt"
        layer = self.add_molecules(out, name=name)
        return mole
        
    @impl_preview(filter_molecules, auto_call=True)
    def _during_filter_molecules(self, layer: MoleculesLayer, predicate: str):
        mole = layer.molecules
        viewer = self.parent_viewer
        try:
            expr = eval(predicate, POLARS_NAMESPACE, {})
        except Exception:
            yield
            return
        out = mole.filter(expr)
        name = "<Preview>"
        if name in viewer.layers:
            layer: Layer = viewer.layers[name]
            layer.data = out.pos
        else:
            layer = self.add_molecules(out, name=name)
        # filtering changes the number of molecules. We need to update the colors.
        layer.face_color = layer.edge_color = "crimson"
        try:
            is_active = yield
        finally:
            if not is_active and layer in viewer.layers:
                viewer.layers.remove(layer)
        return out
    
    def _get_paint_molecules_choice(self, w=None) -> list[str]:
        # don't use get_function_gui. It causes RecursionError.
        gui = self["paint_molecules"].mgui
        if gui is None or gui.layer.value is None:
            return []
        return gui.layer.value.features.columns
        
    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Paint molecules by features")
    def paint_molecules(
        self,
        layer: MoleculesLayer,
        feature_name: OneOf[_get_paint_molecules_choice],
        low: tuple[float, Color] = (0., "blue"),
        high: tuple[float, Color] = (1., "red"),
    ):
        """
        Paint molecules by a feature.

        Parameters
        ----------
        {layer}
        feature_name : str
            Name of the feature to paint by.
        low : (float, Color), default is (0., "blue")
            The lower bound of the feature value and the corresponding color.
        high : (float, Color), default is (1., "red")
            The upper bound of the feature value and the corresponding color.
        """
        rng = (low[0], high[0])
        arr = np.array([low[1], high[1]])
        layer.set_colormap(feature_name, rng, arr)
    
    @impl_preview(paint_molecules, auto_call=True)
    def _during_preview(self, layer: MoleculesLayer, feature_name: str, low, high):
        with _temp_layer_colors(layer):
            self.paint_molecules(layer, feature_name, low, high)
            yield

    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Calculate molecule features")
    def calculate_molecule_features(
        self,
        layer: MoleculesLayer, 
        column_name: str,
        expression: Annotated[ExprStr, {"namespace": POLARS_NAMESPACE}],
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
            new_feat = feat.with_columns([pl_expr.alias(column_name)])
        else:
            new_feat = feat.with_columns([pl.Series(column_name, pl_expr)])
        layer.features = new_feat
        self.reset_choices()  # choices regarding of features need update

    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Calculate intervals")
    def calculate_intervals(
        self,
        layer: MoleculesLayer,
        spline_precision: Annotated[nm, {"min": 0.05, "max": 5.0, "step": 0.05, "label": "spline precision (nm)"}] = 0.2,
    ):
        """
        Calculate intervals between adjucent molecules.

        Parameters
        ----------
        {layer}
        spline_precision : nm, optional
            Precision in nm that is used to define the direction of molecules.
        """
        mole = layer.molecules
        properties = utils.calc_interval(mole, spline_precision)
        
        layer.features = layer.molecules.features.with_columns([pl.Series(Mole.interval, properties)])
        self.reset_choices()  # choices regarding of features need update
        
        # Set colormap
        _clim = [GVar.yPitchMin, GVar.yPitchMax]
        layer.set_colormap(Mole.interval, _clim, self.label_colormap)
        self._need_save = True
        return None
    
    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Calculate skews")
    def calculate_skews(
        self,
        layer: MoleculesLayer,
        spline_precision: Annotated[nm, {"min": 0.05, "max": 5.0, "step": 0.05, "label": "spline precision (nm)"}] = 0.2,
    ):
        """
        Calculate skews between adjucent molecules.

        Parameters
        ----------
        {layer}
        spline_precision : nm, optional
            Precision in nm that is used to define the direction of molecules.
        """
        mole = layer.molecules
        properties = utils.calc_skew(mole, spline_precision)
        
        extreme = np.max(np.abs(properties))
        layer.features = layer.molecules.features.with_columns([pl.Series(Mole.skew, properties)])
        self.reset_choices()  # choices regarding of features need update

        # Set colormap
        _clim = [-extreme, extreme]
        cmap = Colormap(["blue", "white", "red"])
        layer.set_colormap(Mole.skew, _clim, cmap)
        self._need_save = True
        return None
    
    @Molecules_.MoleculeFeatures.wraps
    @set_design(text="Seam search by feature")
    def seam_search_by_feature(
        self,
        layer: MoleculesLayer,
        feature_name: OneOf[_get_selected_layer_choice],
    ):
        """Search for seams by a feature."""
        feat = layer.features
        if feature_name not in feat.columns:
            raise ValueError(f"Column {feature_name} does not exist.")
        npf = utils.roundint(layer.molecules.features[Mole.pf].max() + 1)
        seam = utils.infer_seam_from_labels(feat[feature_name], npf=npf)
        _id = np.arange(len(feat))
        res = (_id - seam) // npf
        layer.features = layer.molecules.features.with_columns(
            [pl.Series(Mole.isotype, res % 2)]
        )
        return None
    
    @toolbar.wraps
    @set_design(icon=ICON_DIR/"pick_next.png")
    @bind_key("F3")
    @do_not_record
    def pick_next(self):
        """Automatically pick cylinder center using previous two points."""
        stride_nm = self.toolbar.Adjust.stride
        angle_pre = self.toolbar.Adjust.angle_precision
        angle_dev = self.toolbar.Adjust.angle_deviation
        max_shifts = self.toolbar.Adjust.max_shifts
        imgb: ip.ImgArray = self.layer_image.data
        binned_scale = imgb.scale.x
        try:
            # orientation is point0 -> point1
            point0: np.ndarray = self.layer_work.data[-2] / binned_scale  # unit: pixel
            point1: np.ndarray = self.layer_work.data[-1] / binned_scale
        except IndexError:
            raise IndexError("Auto pick needs at least two points in the working layer.")
        
        tomo = self.tomogram
        binsize = utils.roundint(self.layer_image.scale[0]/tomo.scale)  # scale of binned reference image
        
        length_px = tomo.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)
        
        shape = (width_px,) + (utils.roundint((width_px+length_px)/1.41),)*2
        
        orientation = point1[1:] - point0[1:]
        img = utils.crop_tomogram(imgb, point1, shape)
        center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
        angle_deg = utils.angle_corr(img, ang_center=center, drot=angle_dev, nrots=utils.ceilint(angle_dev/angle_pre))
        angle_rad = np.deg2rad(angle_deg)
        dr = np.array([0.0, stride_nm * np.cos(angle_rad), -stride_nm * np.sin(angle_rad)])
        if np.dot(orientation, dr[1:]) > np.dot(orientation, -dr[1:]):
            point2 = point1 + dr / binned_scale
        else:
            point2 = point1 - dr / binned_scale
        img_next = utils.crop_tomogram(imgb, point2, shape)

        utils.centering(img_next, point2, angle_deg, drot=5.0, max_shifts=max_shifts/binned_scale)

        next_data = point2 * binned_scale
        self.layer_work.add(next_data)
        if msg := self._check_path():
            self.layer_work.data = self.layer_work.data[:-1]
            raise ValueError(msg)
        change_viewer_focus(self.parent_viewer, point2, binned_scale)
        return None
    
    @toolbar.wraps
    @set_design(icon=ICON_DIR/"auto_center.png")
    @bind_key("F4")
    @do_not_record
    def auto_center(self):
        """Auto centering of selected points."""        
        imgb: ip.ImgArray = self.layer_image.data
        tomo = self.tomogram
        binsize = utils.roundint(self.layer_image.scale[0]/tomo.scale)  # scale of binned reference image
        selected = self.layer_work.selected_data
        
        length_px = tomo.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)
        
        shape = (width_px,) + (utils.roundint((width_px+length_px)/1.41),)*2
        
        points = self.layer_work.data / imgb.scale.x
        last_i = -1
        for i, point in enumerate(points):
            if i not in selected:
                continue
            img_input = utils.crop_tomogram(imgb, point, shape)
            angle_deg = utils.angle_corr(img_input, ang_center=0, drot=89.5, nrots=31)
            utils.centering(img_input, point, angle_deg, drot=3, nrots=7)
            last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], imgb.scale.x)
        return None
    
    @Image.wraps
    @thread_worker.with_progress(desc="Paint cylinders ...")
    @set_design(text="Paint cylinders")
    def paint_cylinders(self):
        """
        Paint cylinder fragments by its local properties.
        
        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """
        if self._current_ft_size is None:
            raise ValueError("Local structural parameters have not been determined yet.")
        lbl = np.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, list[float]] = {0: [0, 0, 0, 0]}
        tomo = self.tomogram
        all_df = tomo.collect_localprops()
        if all_df is None:
            raise ValueError("No local property found.")
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        binsize = utils.roundint(bin_scale / tomo.scale)
        ft_size = self._current_ft_size
        
        lz, ly, lx = [utils.roundint(r/bin_scale*1.73)*2 + 1 for r in [15, ft_size/2, 15]]
        center = np.array([lz, ly, lx])/2 + 0.5
        z, y, x = np.indices((lz, ly, lx))
        cylinders = []
        matrices = []
        for i, spl in enumerate(tomo.splines):
            # Prepare template hollow image
            r0 = spl.radius / tomo.scale * 0.9 / binsize
            r1 = spl.radius / tomo.scale * 1.1 / binsize
            _sq = (z - lz/2 - 0.5)**2 + (x - lx/2 - 0.5)**2
            domains = []
            dist = [-np.inf] + list(spl.distances()) + [np.inf]
            for j in range(spl.anchors.size):
                domain = (r0**2 < _sq) & (_sq < r1**2)
                ry = min((dist[j+1] - dist[j]) / 2, 
                            (dist[j+2] - dist[j+1]) / 2, 
                            ft_size/2) / bin_scale + 0.5 
                    
                ry = max(utils.ceilint(ry), 1)
                domain[:, :ly//2-ry] = 0
                domain[:, ly//2+ry+1:] = 0
                domain = domain.astype(np.float32)
                domains.append(domain)
                yield
                
            cylinders.append(domains)
            matrices.append(spl.affine_matrix(center=center, inverse=True))
        
        cylinders = np.concatenate(cylinders, axis=0)
        matrices = np.concatenate(matrices, axis=0)
        
        out = np.empty_like(cylinders)
        for i, (img, matrix) in enumerate(zip(cylinders, matrices)):
            out[i] = ndi.affine_transform(img, matrix, order=1, cval=0, prefilter=False)
            yield
        out = out > 0.3
            
        # paint roughly
        for i, crd in enumerate(tomo.collect_anchor_coords()):
            center = tomo.nm2pixel(crd, binsize=binsize)
            sl: list[slice] = []
            outsl: list[slice] = []
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape):
                _sl, _pad = utils.make_slice_and_pad(c - l//2, c + l//2 + 1, size)
                sl.append(_sl)
                outsl.append(
                    slice(_pad[0] if _pad[0] > 0 else None,
                         -_pad[1] if _pad[1] > 0 else None)
                )
                yield

            sl = tuple(sl)
            outsl = tuple(outsl)
            lbl[sl][out[i][outsl]] = i + 1
        
        # paint finely
        ref = self.layer_image.data
        thr = np.percentile(ref[lbl > 0], 5)
        lbl[ref < thr] = 0
        
        # Labels layer properties
        _id = "ID"
        _str = "structure"
        columns = [_id, H.riseAngle, H.yPitch, H.skewAngle, _str]
        df = (
            all_df
            .select([IDName.spline, IDName.pos, H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start])
            .with_columns([
                pl.format("{}-{}", pl.col(IDName.spline), pl.col(IDName.pos)).alias(_id),
                pl.format("{}_{}", pl.col(H.nPF), pl.col(H.start).round(1)).alias(_str),
                pl.col(H.riseAngle),
                pl.col(H.yPitch),
                pl.col(H.skewAngle),
            ])
        ).to_pandas()
        back = pd.DataFrame({c: [np.nan] for c in columns})
        props = pd.concat([back, df[columns]], ignore_index=True)
        
        # Add labels layer
        @thread_worker.to_callback
        def _on_return():
            if self.layer_paint is None:
                self.layer_paint = self.parent_viewer.add_labels(
                    lbl, color=color, scale=self.layer_image.scale,
                    translate=self.layer_image.translate, opacity=0.33, name="Label",
                    properties=props
                )
            else:
                self.layer_paint.data = lbl
                self.layer_paint.properties = props
            self._update_colormap()
        return _on_return
    
    @Image.wraps
    @set_options(auto_call=True)
    @set_design(text="Set colormap")
    def set_colormap(
        self,
        start: Color = (0, 0, 1, 1),
        end: Color = (1, 0, 0, 1),
        limit: Annotated[tuple[float, float], {"options": {"min": -20, "max": 20, "step": 0.01}, "label": "limit (nm)"}] = (4.00, 4.24), 
        color_by: OneOf[H.yPitch, H.skewAngle, H.nPF, H.riseAngle] = H.yPitch,
    ):
        """
        Set the color-map for painting cylinders.
        
        Parameters
        ----------
        start : tuple, default is "blue"
            RGB color of the lower bound of the colormap.
        end : tuple, default is "red"
            RGB color of the higher bound of the colormap.
        limit : tuple, default is (4.00, 4.24)
            Color limit (nm).
        color_by : str, default is "yPitch"
            Select what property image will be colored by.
        """        
        self.label_colormap = Colormap([start, end], display_name="LocalProperties")
        self.label_colorlimit = limit
        self._update_colormap(prop=color_by)
        return None
    
    @Image.wraps
    @set_design(text="Show colorbar")
    @do_not_record
    def show_colorbar(self):
        """Create a colorbar from the current colormap."""
        arr = self.label_colormap.colorbar[:5]  # shape == (5, 28, 4)
        xmin, xmax = self.label_colorlimit
        _Logger.widget.show()
        with _Logger.set_plt(rc_context={"font.size": 15}):
            plt.imshow(arr)
            plt.xticks([0, 27], [f"{xmin:.2f}", f"{xmax:.2f}"])
            plt.yticks([], [])
            plt.show()
        return None
    
    @Image.wraps
    @set_design(text="Simulate cylindric structure")
    @do_not_record
    def open_simulator(self):
        return self.cylinder_simulator.show()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Non-GUI methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
    @nogui
    @do_not_record
    def get_molecules(self, name: str = None) -> Molecules:
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
    def add_molecules(self, molecules: Molecules, name: str = None):
        """Add molecules as a points layer to the viewer."""
        return add_molecules(self.parent_viewer, molecules, name)

    @nogui
    @do_not_record
    def get_loader(
        self,
        name: str = None,
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
    def get_spline(self, i: int = None) -> CylSpline:
        """Get the i-th spline object. Return current one by default."""
        tomo = self.tomogram
        if i is None:
            i = self.SplineControl.num
        return tomo.splines[i]
    
    @SplineControl.num.connect
    @SplineControl.pos.connect
    @SplineControl.footer.focus.connect
    def _focus_on(self):
        """Change camera focus to the position of current spline fragment."""
        if self.layer_paint is None:
            return None
        if not self.SplineControl.footer.focus:
            self.layer_paint.show_selected_label = False
            return None
        
        viewer = self.parent_viewer
        i = self.SplineControl.num
        j = self.SplineControl.pos
        
        tomo = self.tomogram
        spl = tomo.splines[i]
        pos = spl.anchors[j]
        next_center = spl(pos) / tomo.scale
        change_viewer_focus(viewer, next_center, tomo.scale)
        
        self.layer_paint.show_selected_label = True
        
        j_offset = sum(spl.anchors.size for spl in tomo.splines[:i])
        self.layer_paint.selected_label = j_offset + j + 1
        return None

    def _update_colormap(self, prop: str = H.yPitch):
        if self.layer_paint is None:
            return None
        color = {0: np.array([0., 0., 0., 0.], dtype=np.float32),
                 None: np.array([0., 0., 0., 1.], dtype=np.float32)}
        lim0, lim1 = self.label_colorlimit
        df = self.tomogram.collect_localprops()[prop]
        for i, value in enumerate(df):
            color[i + 1] = self.label_colormap.map((value - lim0)/(lim1 - lim0))
        self.layer_paint.color = color
        return None

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
        change_viewer_focus(viewer, np.asarray(imgb.shape)/2, imgb.scale.x)
        
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
            _name = os.path.join(*parts[-2:])
        except Exception:
            _name = f"Tomogram<{hex(id(tomo))}>"
        _Logger.print_html(f"<h2>{_name}</h2>")
        self.clear_all()
        if filt:
            self.filter_reference_image()
    
    def _check_path(self) -> str:
        tomo = self.tomogram
        imgshape_nm = np.array(tomo.image.shape) * tomo.image.scale.x
        if self.layer_work.data.shape[0] == 0:
            return ""
        else:
            point0 = self.layer_work.data[-1]
            box_size = (GVar.fitWidth,) + ((GVar.fitWidth+GVar.fitLength)/1.41,)*2
            
            if not np.all([r/4 <= p < s - r/4
                           for p, s, r in zip(point0, imgshape_nm, box_size)]):
                # outside image
                return "Outside boundary."        
        return ""
    
    def _on_layer_removing(self, event):
        # NOTE: To make recorded macro completely reproducible, removing molecules 
        # from the viewer layer list must always be monitored.
        layer: Layer = self.parent_viewer.layers[event.index]
        if isinstance(layer, MoleculesLayer):
            expr = mk.Mock(mk.symbol(self)).parent_viewer.layers[layer.name].expr
            self.macro.append(mk.Expr("del", [expr]))
        return
    
    def _on_layer_removed(self, event):
        idx: int = event.index
        layer: Layer = event.value
        if layer in (self.layer_image, self.layer_prof, self.layer_work, self.layer_paint):
            import warnings
            self.parent_viewer.layers.insert(idx, layer)
            warnings.warn(f"Cannot remove layer {layer.name!r}", UserWarning)

    def _init_layers(self):
        viewer: napari.Viewer = self.parent_viewer
        viewer.layers.events.removing.disconnect(self._on_layer_removing)
        viewer.layers.events.removed.disconnect(self._on_layer_removed)
        
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
        if spl.globalprops is not None:
            headers = [H.yPitch, H.skewAngle, H.nPF, H.start]
            pitch, skew, npf, start = spl.globalprops[headers].row(0)
            radius = spl.radius
            ori = spl.orientation
            self.GlobalProperties._set_text(pitch, skew, npf, start, radius, ori)
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
        if spl.localprops is not None:
            pitch, skew, npf, start = spl.localprops.select(
                [H.yPitch, H.skewAngle, H.nPF, H.start]
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
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        self.layer_prof.feature_defaults[SPLINE_ID] = i
        self.layer_prof.add(fit)
        self.overview.add_curve(
            fit[:, 2]/scale, fit[:, 1]/scale, color="lime", lw=2, name=f"spline-{i}",
        )
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
        self.overview.layers.clear()
        self.layer_prof.data = []
        scale = self.layer_image.scale[0]
        for i, spl in enumerate(self.tomogram.splines):
            self._add_spline_to_images(spl, i)
            if spl._anchors is None:
                continue
            coords = spl()
            self.overview.add_scatter(
                coords[:, 2]/scale, 
                coords[:, 1]/scale,
                color="lime", 
                symbol="x",
                lw=2,
                size=10,
                name=f"spline-{i}-anc",
            )
        return None


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Preview methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @impl_preview(load_project)
    def _preview_text(self, path: str):
        pviewer = CylindraProject.from_json(path).make_project_viewer()
        pviewer.native.setParent(self.native, pviewer.native.windowFlags())
        return pviewer.show()
    
    @impl_preview(load_molecules)
    def _preview_table(self, paths: list[str]):
        return _previews.view_tables(paths, parent=self)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Setups
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @setup_function_gui(paint_molecules)
    def _(self, gui):
        from polars.datatypes import NumericType, IntegralType
        
        gui.layer.changed.connect(gui.feature_name.reset_choices)
        
        @gui.layer.changed.connect
        @gui.feature_name.changed.connect
        def _on_feature_change():
            feature_name: str = gui.feature_name.value
            if feature_name is None:
                return
            layer: MoleculesLayer = gui.layer.value
            series = layer.molecules.features[feature_name]
            if issubclass(series.dtype, NumericType):
                gui.low[0].min = gui.high[0].min = series.min()
                gui.low[0].max = gui.high[0].max = series.max()
                gui.low[0].value = gui.low[0].min
                gui.high[0].value = gui.high[0].max
                if issubclass(series.dtype, IntegralType):
                    gui.low[0].step = gui.high[0].step = 1
                else:
                    gui.low[0].step = gui.high[0].step = None

        return None
    
    @setup_function_gui(split_molecules)
    @setup_function_gui(seam_search_by_feature)
    def _(self, gui):
        gui[0].changed.connect(gui[1].reset_choices)    

############################################################################################
#   Other helper functions
############################################################################################

def _multi_affine(images, matrices, cval: float = 0, order=1):
    out = np.empty_like(images)
    for i, (img, matrix) in enumerate(zip(images, matrices)):
        out[i] = ndi.affine_transform(
            img, matrix, order=order, cval=cval, prefilter=order>1
        )
    return out

@contextmanager
def _temp_layer_colors(layer: MoleculesLayer):
    """Temporarily change the colors of a layer and restore them afterwards."""
    fc = layer.face_color
    ec = layer.edge_color
    fcmap = layer.face_colormap
    ecmap = layer.edge_colormap
    fclim = layer.face_contrast_limits
    eclim = layer.edge_contrast_limits
    try:
        yield
    finally:
        layer.face_color = fc
        layer.edge_color = ec
        layer.face_colormap = fcmap
        layer.edge_colormap = ecmap
        layer.face_contrast_limits = fclim
        layer.edge_contrast_limits = eclim
