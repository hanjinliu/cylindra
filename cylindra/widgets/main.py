import os
import re
from pathlib import Path
from timeit import default_timer
from typing import Annotated, Union
import warnings

import impy as ip
import macrokit as mk
import matplotlib.pyplot as plt
import napari
import numpy as np
import polars as pl
import pandas as pd
from acryo import Molecules, SubtomogramLoader
from acryo.alignment import PCCAlignment, ZNCCAlignment
from magicclass import (MagicTemplate, bind_key, build_help, confirm,
                        do_not_record, field, get_function_gui, magicclass, impl_preview, nogui,
                        set_design, set_options)
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.types import Bound, Color, OneOf, Optional, SomeOf
from magicclass.utils import thread_worker
from magicclass.widgets import ConsoleTextEdit, FloatRangeSlider, Logger
from napari.layers import Image, Labels, Layer, Points
from napari.utils import Colormap
from scipy import ndimage as ndi

from cylindra import utils
from cylindra.components import CylSpline, CylTomogram
from cylindra.const import (
    ALN_SUFFIX, MOLECULES, SELECTION_LAYER_NAME,
    WORKING_LAYER_NAME, GlobalVariables as GVar, PropertyNames as H, 
    MoleculesHeader as Mole, Ori, nm
)
from cylindra.types import MonomerLayer, get_monomer_layers
from cylindra.project import CylindraProject

# widgets
from cylindra.widgets import _previews, _shared_doc, subwidgets, widget_utils
from cylindra.widgets.feature_control import FeatureControl
from cylindra.widgets.image_processor import ImageProcessor
from cylindra.widgets.properties import GlobalPropertiesWidget, LocalPropertiesWidget
from cylindra.widgets.spline_control import SplineControl
from cylindra.widgets.spline_clipper import SplineClipper
from cylindra.widgets.spline_fitter import SplineFitter
from cylindra.widgets.sweeper import SplineSweeper
from cylindra.widgets.simulator import CylinderSimulator
from cylindra.widgets.measure import SpectraMeasurer
from cylindra.widgets.widget_utils import (
    FileFilter, add_molecules, change_viewer_focus, update_features
)

ICON_DIR = Path(__file__).parent / "icons"
SPLINE_ID = "spline-id"

INTERPOLATION_CHOICES = (("nearest", 0), ("linear", 1), ("cubic", 3))
METHOD_CHOICES = (
    ("Phase Cross Correlation", "pcc"),
    ("Zero-mean Normalized Cross Correlation", "zncc"),
)

# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_ZRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_YRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_XRotation = Annotated[tuple[float, float], {"options": {"max": 90.0, "step": 0.1}}]
_MaxShifts = Annotated[tuple[nm, nm, nm], {"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"}]
_SubVolumeSize = Annotated[Optional[nm], {"text": "Use template shape", "options": {"value": 12., "max": 100.}, "label": "size (nm)"}]


def _fmt_layer_name(fmt: str):
    """Define a formatter for progressbar description."""
    def _formatter(**kwargs):
        layer: Layer = kwargs["layer"]
        return fmt.format(layer.name)
    return _formatter

def _get_alignment(method: str):
    if method == "zncc":
        return ZNCCAlignment
    elif method == "pcc":
        return PCCAlignment
    else:
        raise ValueError(f"Method {method!r} is unknown.")


############################################################################################
#   The Main Widget of cylindra
############################################################################################

@magicclass(widget_type="scrollable", name="cylindra")
@_shared_doc.update_cls
class CylindraMainWidget(MagicTemplate):
    # Main GUI class.
    
    _SplineFitter = field(SplineFitter, name="Spline fitter")  # Widget for manual spline fitting
    _SplineClipper = field(SplineClipper, name="Spline clipper")  # Widget for manual spline clipping
    _SplineSweeper = field(SplineSweeper, name="Spline sweeper")  # Widget for sweeping along splines
    _ImageProcessor = field(ImageProcessor, name="Image Processor")  # Widget for pre-filtering/pre-processing
    _FeatureControl = field(FeatureControl, name="Feature Control")  # Widget for visualizing/analyzing features
    _Simulator = field(CylinderSimulator, name="Cylinder Simulator")  # Widget for tomogram simulator
    _SpectraMeasurer = field(SpectraMeasurer, name="FFT Measurer")  # Widget for measuring FFT parameters from a 2D power spectra
    
    # The logger widget.
    @magicclass(labels=False, name="Logger")
    @set_design(min_height=200)
    class _LoggerWindow(MagicTemplate):
        log = field(Logger, name="Log")
    
    @property
    def log(self):
        """Return the logger widget."""
        return self._LoggerWindow.log
    
    @property
    def project_directory(self) -> Union[Path, None]:
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
        self.objectName()  # load napari types
        
    def __post_init__(self):
        self.set_colormap()
        self.min_width = 400
        self.LocalProperties.collapsed = False
        self.GlobalProperties.collapsed = False
        self.overview.min_height = 300
        
        return None

    @property
    def sub_viewer(self) -> napari.Viewer:
        """Return the sub-viewer that is used for subtomogram averaging."""
        return self._subtomogram_averaging._viewer

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
        total="(1+n_refine+int(local_props)+int(global_props))*len(splines)",
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
                    df = self.tomogram.collect_globalprops(i=splines).transpose()
                    df.columns = [f"Spline-{i}" for i in splines]
                    self.log.print_table(df, precision=3)
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
    
    @Others.Macro.wraps
    @set_options(path={"filter": FileFilter.PY})
    @set_design(text="Run file")
    @do_not_record
    def run_file(self, path: Path):
        """Run a Python script file."""
        with open(path, mode="r") as f:
            txt = f.read()
        macro = mk.parse(txt)
        _ui = str(str(mk.symbol(self)))
        with self.macro.blocked():
            self._format_macro(macro).eval({}, {_ui: self})
        self.macro.extend(macro.args)
        return None
        
    @Others.Macro.wraps
    @set_design(text="Show macro")
    @do_not_record
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
    @set_design(text="Open logger")
    @do_not_record
    def open_logger(self):
        """Open logger window."""
        self._LoggerWindow.show()
        return None

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
        versions = widget_utils.get_versions()
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
    @set_options(filter={"label": "Filter the reference image layer."})
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
    def load_project(self, path: Annotated[Path, {"filter": FileFilter.JSON}], filter: bool = True):
        """Load a project json file."""
        project = CylindraProject.from_json(path)
        
        self.tomogram = CylTomogram.imread(
            path=project.image, 
            scale=project.scale, 
            binsize=project.multiscales, 
        )
        
        self._current_ft_size = project.current_ft_size
        self._macro_offset = len(self.macro)
        
        # load splines
        splines = [CylSpline.from_json(path) for path in project.splines]
        localprops_path = project.localprops
        if localprops_path is not None:
            all_localprops = dict(iter(pd.read_csv(localprops_path).groupby("SplineID")))
        else:
            all_localprops = {}
        globalprops_path = project.globalprops
        if globalprops_path is not None:
            all_globalprops = dict(pd.read_csv(globalprops_path, index_col=0).iterrows())
        else:
            all_globalprops = {}
        
        for i, spl in enumerate(splines):
            spl.localprops = all_localprops.get(i, None)
            if spl.localprops is not None:
                spl._anchors = np.asarray(spl.localprops.get(H.splPosition))
                spl.localprops.pop("SplineID")
                spl.localprops.pop("PosID")
                spl.localprops.index = range(len(spl.localprops))
            spl.globalprops = all_globalprops.get(i, None)
            if spl.globalprops is not None:
                try:
                    spl.radius = spl.globalprops.pop("radius")
                except KeyError:
                    pass
                try:
                    spl.orientation = spl.globalprops.pop("orientation")
                except KeyError:
                    pass
        
        @thread_worker.to_callback
        def _load_project_on_return():
            self._send_tomogram_to_viewer(filt=filter)
            
            if splines:
                self.tomogram._splines = splines
                self._update_splines_in_images()
                with self.macro.blocked():
                    self.sample_subtomograms()
            
            # load molecules
            for path in project.molecules:
                mole = Molecules.from_csv(path)
                add_molecules(self.parent_viewer, mole, name=Path(path).stem)
            
            # load global variables
            if project.global_variables:
                with self.macro.blocked():
                    self.Others.Global_variables.load_variables(project.global_variables)
            
            # append macro
            with open(project.macro) as f:
                txt = f.read()
                
            macro = mk.parse(txt)
            self.macro.extend(macro.args)

            # load subtomogram analyzer state
            self._subtomogram_averaging.template_path = project.template_image or ""
            self._subtomogram_averaging._set_mask_params(project.mask_parameters)
            self.reset_choices()
            self._need_save = False
        
        return _load_project_on_return
    
    @File.wraps
    @set_options(
        json_path={"mode": "w", "filter": FileFilter.JSON},
        results_dir={"text": "Save at the same directory", "options": {"mode": "d"}}
    )
    @set_design(text="Save project")
    def save_project(self, json_path: Path, results_dir: Optional[Path] = None):
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
    @set_options(paths={"filter": FileFilter.JSON})
    @set_design(text="Load splines")
    def load_splines(self, paths: list[Path]):
        """
        Load splines using a list of json paths.

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
    @set_options(paths={"filter": FileFilter.CSV})
    @set_design(text="Load molecules")
    def load_molecules(self, paths: list[Path]):
        """Load molecules from a csv file."""
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        for path in paths:
            mole = Molecules.from_csv(path)
            name = Path(path).stem
            add_molecules(self.parent_viewer, mole, name)
        return None
    
    @File.wraps
    @set_options(save_path={"mode": "w", "filter": FileFilter.JSON})
    @set_design(text="Save spline")
    def save_spline(self, spline: OneOf[_get_splines], save_path: Path):
        """Save splines as a json file."""
        spl = self.tomogram.splines[spline]
        spl.to_json(save_path)
        return None
        
    @File.wraps
    @set_design(text="Save molecules")
    @set_options(save_path={"mode": "w", "filter": FileFilter.CSV})
    def save_molecules(self, layer: MonomerLayer, save_path: Path):
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
        mole: Molecules = layer.metadata[MOLECULES]
        mole.to_csv(save_path)
        return None
    
    @File.wraps
    @set_design(text="Process images")
    @do_not_record
    def process_images(self):
        """Open image processor."""
        return self._ImageProcessor.show()
    
    @File.wraps
    @set_design(text="View project")
    @set_options(path={"filter": FileFilter.JSON})
    @do_not_record
    def view_project(self, path: Path):
        pviewer = CylindraProject.from_json(path).make_project_viewer()
        pviewer.native.setParent(self.native, pviewer.native.windowFlags())
        return pviewer.show()
    
    @Image.wraps
    @set_design(text="Show image info")
    def show_image_info(self):
        """Show information of current active tomogram."""
        tomo = self.tomogram
        img = tomo.image
        source = tomo.metadata.get("source", "Unknown")
        scale = tomo.scale
        shape_px = ", ".join(f"{s} px" for s in img.shape)
        shape_nm = ", ".join(f"{s*scale:.2f} nm" for s in img.shape)
        value = (
            f"File: {source}\n"
            f"Voxel size: {scale:.4f} nm/px\n"
            f"ZYX-Shape: ({shape_px}), ({shape_nm})"
        )
        w = ConsoleTextEdit(value=value)
        w.read_only = True
        w.native.setParent(self.native, w.native.windowFlags())
        w.show()
        return None

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
    @dask_thread_worker.with_progress(desc="Adding multiscale (bin = {bin_size})".format)
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
        self._SplineSweeper.show()
        return self._SplineSweeper.refresh_widget_state()
        
    @Image.wraps
    @set_design(text="Sample subtomograms")
    def sample_subtomograms(self):
        """Sample subtomograms at the anchor points on splines"""
        self._SplineFitter.close()
        
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

    @Splines.wraps
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
    
    @Splines.wraps
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
    @set_options(
        max_interval={"label": "Max interval (nm)"},
        edge_sigma={"text": "Do not mask image"},
    )
    @set_design(text="Fit splines")
    @thread_worker.with_progress(desc="Spline Fitting")
    def fit_splines(
        self, 
        max_interval: nm = 30,
        bin_size: OneOf[_get_available_binsize] = 1,
        degree_precision: float = 0.5,
        edge_sigma: Optional[nm] = 2.0,
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
    @set_options(max_interval={"label": "Max interval (nm)"})
    @set_design(text="Fit splines manually")
    @do_not_record
    def fit_splines_manually(self, max_interval: nm = 50.0):
        """
        Open a spline fitter window and fit cylinder with spline manually.

        Parameters
        ----------
        {max_interval}
        """        
        self._SplineFitter._load_parent_state(max_interval=max_interval)
        self._SplineFitter.show()
        return None

    @Splines.wraps
    @set_design(text="Add anchors")
    @set_options(interval={"label": "Interval between anchors (nm)", "min": 1.0})
    def add_anchors(self, interval: nm = 25.0):
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
    @set_options(radius={"text": "Measure radii by radial profile."})
    @set_design(text="Set radius")
    @thread_worker.with_progress(desc="Measuring Radius")
    def set_radius(self, radius: Optional[nm] = None, bin_size: OneOf[_get_available_binsize] = 1):
        """Measure cylinder radius for each spline path."""        
        self.tomogram.set_radius(radius=radius, binsize=bin_size)
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_options(
        max_interval={"label": "Maximum interval (nm)"},
        corr_allowed={"label": "Correlation allowed", "max": 1.0, "step": 0.1},
    )
    @set_design(text="Refine splines")
    @thread_worker.with_progress(desc="Refining splines")
    def refine_splines(
        self,
        max_interval: nm = 30,
        corr_allowed: float = 0.9,
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
    @set_options(interval={"label": "Interval (nm)", "min": 1.0})
    @set_design(text="Molecules to spline")
    @confirm(
        text="The existing splines will be removed.\nDo you want to run?",
        condition="len(self.SplineControl._get_splines()) > 0",
    )
    def molecules_to_spline(
        self, 
        layers: SomeOf[get_monomer_layers],
        interval: nm = 24.5,
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
            spl = widget_utils.molecules_to_spline(layer)
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
        """Open the spline clipper widget."""
        self._SplineClipper.show()
        if self.tomogram.n_splines > 0:
            self._SplineClipper.load_spline(self.SplineControl.num)
        
    @Analysis.wraps
    @set_options(
        interval={"min": 1.0, "step": 0.5},
        ft_size={"min": 2.0, "step": 0.5},
    )
    @set_design(text="Local FT analysis")
    @thread_worker.with_progress(desc="Local Fourier transform", total="self.tomogram.n_splines")
    def local_ft_analysis(
        self,
        interval: nm = 24.5,
        ft_size: nm = 24.5,
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
            df = self.tomogram.collect_globalprops().transpose()
            df.columns = [f"Spline-{i}" for i in range(len(df.columns))]
            self.log.print_table(df, precision=3)
            self._update_global_properties_in_widget()
        
        for i in range(self.tomogram.n_splines):
            tomo.global_ft_params(i=i, binsize=bin_size)
            yield _global_ft_analysis_on_yield(i)
        self._need_save = True
        return _global_ft_analysis_on_return
    
    @Analysis.wraps
    @set_design(text="Open spectra measurer")
    @do_not_record
    def open_spectra_measurer(self):
        """Open the spectra measurer widget to determine cylindric parameters."""
        return self._SpectraMeasurer.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Molecules_.Mapping.wraps
    @set_options(length={"text": "Use full length"})
    @set_design(text="Map monomers")
    @bind_key("M")
    def map_monomers(
        self,
        splines: SomeOf[_get_splines] = (),
        length: Optional[nm] = None,
    ):
        """
        Map points to tubulin molecules using the results of global Fourier transformation.
        
        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        length : nm, optional
            Length from the tip where monomers will be mapped.
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        molecules = tomo.map_monomers(i=splines, length=length)
        
        self.log.print_html("<code>Map_monomers</code>")
        for i, mol in enumerate(molecules):
            _name = f"Mono-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.log.print(f"{_name!r}: n = {len(mol)}")
            
        self._need_save = True
        return molecules

    @Molecules_.Mapping.wraps
    @set_options(
        interval={"text": "Set to dimer length"},
        length={"text": "Use full length"}
    )
    @set_design(text="Map centers")
    def map_centers(
        self,
        splines: SomeOf[_get_splines] = (),
        interval: Optional[nm] = None,
        length: Optional[nm] = None,
    ):
        """
        Map molecules along splines. Each molecule is rotated by skew angle.
        
        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        {interval}
        length : nm, optional
            Length from the tip where monomers will be mapped.
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        mols = tomo.map_centers(i=splines, interval=interval, length=length)
        self.log.print_html("<code>Map_centers</code>")
        for i, mol in enumerate(mols):
            _name = f"Center-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.log.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return None
    
    @Molecules_.Mapping.wraps
    @set_options(
        interval={"text": "Set to dimer length"},
        angle_offset={"max": 360}
    )
    @set_design(text="Map alogn PF")
    def map_along_pf(
        self,
        splines: SomeOf[_get_splines],
        interval: Optional[nm] = None,
        angle_offset: float = 0.0,
    ):
        """
        Map molecules along splines. Each molecule is rotated by skew angle.
        
        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        {interval}
        angle_offset : float, default is 0.0
            Offset of PF angle in radian.
        """
        tomo = self.tomogram
        mols = tomo.map_pf_line(i=splines, interval=interval, angle_offset=angle_offset)
        self.log.print_html("<code>Map_along_PF</code>")
        for i, mol in enumerate(mols):
            _name = f"PF line-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.log.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return None

    @Molecules_.wraps
    @set_design(text="Show orientation")
    def show_orientation(
        self,
        layer: MonomerLayer,
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
        mol: Molecules = layer.metadata[MOLECULES]
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
        layer: MonomerLayer,
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
        ndim = 3
        mole: Molecules = layer.metadata[MOLECULES]
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
        from scipy.spatial.transform import Rotation
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
            points_layer: Points = viewer.layers[name]
            points_layer.data = mole_new.pos
            points_layer.selected_data = set()
            points_layer.metadata[MOLECULES] = mole_new
            update_features(points_layer, features)
        return None
    
    @Molecules_.wraps
    @set_design(text="Concatenate molecules")
    def concatenate_molecules(self, layers: SomeOf[get_monomer_layers]):
        """
        Concatenate selected monomer layers and create a new layer.

        Parameters
        ----------
        {layers}
        """
        if len(layers) == 0:
            raise ValueError("No layer selected.")
        molecules: list[Molecules] = [layer.metadata[MOLECULES] for layer in layers]
        all_molecules = Molecules.concat(molecules)
        points = add_molecules(self.parent_viewer, all_molecules, name="Mono-concat")
        layer_names: list[str] = []
        for layer in layers:
            layer.visible = False
            layer_names.append(layer.name)
        
        self.log.print_html("<code>concatenate_molecules</code>")
        self.log.print("Concatenated:", ", ".join(layer_names))
        self.log.print(f"{points.name!r}: n = {len(all_molecules)}")
        return None

    @Molecules_.wraps
    @set_options(
        spline_precision={"min": 0.05, "max": 5.0, "step": 0.05, "label": "spline precision (nm)"}
    )
    @set_design(text="Calculate intervals")
    def calculate_intervals(
        self,
        layer: MonomerLayer,
        spline_precision: nm = 0.2,
    ):
        """
        Calculate intervals between adjucent molecules.
        
        If filter is applied, connections and boundary padding mode are safely defined using 
        global properties. For instance, with 13_3 microtubule, the 13-th monomer in the first 
        round is connected to the 1st monomer in the 4th round.

        Parameters
        ----------
        {layer}
        spline_precision : nm, optional
            Precision in nm that is used to define the direction of molecules for calculating
            projective interval.
        """
        mole: Molecules = layer.metadata[MOLECULES]
        spl = widget_utils.molecules_to_spline(layer)
        try:
            pf_label = mole.features[Mole.pf]
            pos_list: list[np.ndarray] = []  # each shape: (y, ndim)
            for pf in range(pf_label.max() + 1):
                pos_list.append(mole.pos[pf_label == pf])
            pos = np.stack(pos_list, axis=1)  # shape: (y, pf, ndim)
            
        except Exception as e:
            raise TypeError(
                f"Reshaping failed. Molecules represented by layer {layer.name} must be "
                f"correctly labeled at {Mole.pf!r} feature. Original error is\n"
                f"{type(e).__name__}: {e}"
            ) from e
        
        u = spl.world_to_y(mole.pos, precision=spline_precision)
        spl_vec = spl(u, der=1)
        
        from ..utils import diff
        y_interval = diff(pos, spl_vec)
        
        properties = y_interval.ravel()
        if properties[0] < 0:
            properties = -properties
        _clim = [GVar.yPitchMin, GVar.yPitchMax]
        
        update_features(layer, {Mole.interval: properties})
        self.reset_choices()  # choices regarding of features need update
        
        # Set colormap
        layer.face_color = layer.edge_color = Mole.interval
        layer.face_colormap = layer.edge_colormap = self.label_colormap
        layer.face_contrast_limits = layer.edge_contrast_limits = _clim
        layer.refresh()
        self._need_save = True
        return None
    
    @Molecules_.wraps
    @set_design(text="Open feature control")
    @do_not_record
    def open_feature_control(self):
        """Open the molecule-feature control widget."""
        self._FeatureControl.show()
        self._FeatureControl._update_table_and_expr()
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Subtomogram averaging methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Analysis.wraps
    @set_design(text="Open subtomogram analyzer")
    @do_not_record
    def open_subtomogram_analyzer(self):
        """Open the subtomogram analyzer dock widget."""
        return self._subtomogram_averaging.show()
    
    _subtomogram_averaging = subwidgets.SubtomogramAveraging
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_design(text="Average all")
    @dask_thread_worker.with_progress(desc= _fmt_layer_name("Subtomogram averaging of {!r}"))
    def average_all(
        self,
        layer: MonomerLayer,
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Subtomogram averaging using all the subvolumes.

        >>> loader = ui.tomogram.get_subtomogram_loader(molecules, shape)
        >>> averaged = ui.tomogram
            
        Parameters
        ----------
        {layer}{size}{interpolation}{bin_size}
        """
        t0 = default_timer()
        molecules: Molecules = layer.metadata[MOLECULES]
        tomo = self.tomogram
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        loader = tomo.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation
        )
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale)
        self.log.print_html(f"<code>average_all</code> ({default_timer() - t0:.1f} sec)")
        
        return thread_worker.to_callback(
            self._subtomogram_averaging._show_reconstruction, img, f"[AVG]{layer.name}"
        )
        
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_design(text="Average subset")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Subtomogram averaging (subset) of {!r}"))
    def average_subset(
        self,
        layer: MonomerLayer,
        size: _SubVolumeSize = None,
        method: OneOf["steps", "first", "last", "random"] = "steps", 
        number: int = 64,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Subtomogram averaging using a subset of subvolumes.
        
        This function is equivalent to

        Parameters
        ----------
        {layer}{size}
        method : str, optional
            How to choose subtomogram subset. 
            (1) steps: Each 'steps' subtomograms from the tip of spline. 
            (2) first: First subtomograms.
            (3) last: Last subtomograms.
            (4) random: choose randomly.
        number : int, default is 64
            Number of subtomograms to use.
        {bin_size}
        """
        t0 = default_timer()
        molecules: Molecules = layer.metadata[MOLECULES]
        nmole = len(molecules)
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        if nmole < number:
            raise ValueError(f"There are only {nmole} subtomograms.")
        if method == "steps":
            step = nmole//number
            sl = slice(0, step * number, step)
        elif method == "first":
            sl = slice(0, number)
        elif method == "last":
            sl = slice(-number, -1)
        elif method == "random":
            sl_all = np.arange(nmole, dtype=np.uint32)
            np.random.shuffle(sl_all)
            sl = sl_all[:number]
        else:
            raise NotImplementedError(method)
        mole = molecules.subset(sl)
        loader = self.tomogram.get_subtomogram_loader(
            mole, shape, binsize=bin_size, order=1
        )
        
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale)
        self.log.print_html(f"<code>average_subset</code> ({default_timer() - t0:.1f} sec)")
        return thread_worker.to_callback(
            self._subtomogram_averaging._show_reconstruction,
            img,
            f"[AVG(n={number})]{layer.name}"
        )
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        n_set={"min": 1, "label": "number of image pairs"},
    )
    @set_design(text="Split-and-average")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Split-and-averaging of {!r}"))
    def split_and_average(
        self,
        layer: MonomerLayer,
        n_set: int = 1,
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Split molecules into two groups and average separately.

        Parameters
        ----------
        {layer}
        n_set : int, default is 1
            How many pairs of average will be calculated.
        {size}{interpolation}{bin_size}
        """
        t0 = default_timer()
        molecules: Molecules = layer.metadata[MOLECULES]
        tomo = self.tomogram
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        loader = tomo.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation
        )
        axes = "ipzyx" if n_set > 1 else "pzyx"
        img = ip.asarray(loader.average_split(n_set=n_set), axes=axes)
        img.set_scale(zyx=loader.scale)
        self.log.print_html(f"<code>split_and_average</code> ({default_timer() - t0:.1f} sec)")

        return thread_worker.to_callback(
            self._subtomogram_averaging._show_reconstruction, img, f"[Split]{layer.name}"
        )
    
    @_subtomogram_averaging.Refinement.wraps
    @set_design(text="Align averaged")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Aligning averaged image of {!r}"))
    def align_averaged(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        z_rotation: _ZRotation = (3., 3.),
        y_rotation: _YRotation = (15., 3.),
        x_rotation: _XRotation = (3., 3.),
        bin_size: OneOf[_get_available_binsize] = 1,
        method: OneOf[METHOD_CHOICES] = "zncc",
    ):
        """
        Align the averaged image at current monomers to the template image.
        
        This function creates a new layer with transformed monomers, which should
        align well with template image.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{z_rotation}{y_rotation}{x_rotation}{bin_size}{method}
        """
        t0 = default_timer()
        mole: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        if mask is not None and template.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch between tempalte image ({tuple(template.shape)}) "
                f"and mask image ({tuple(mask.shape)})."
            )
        
        loader, template, mask = self._check_binning_for_alignment(
            template, mask, bin_size, mole, order=1
        )
        _scale = self.tomogram.scale * bin_size
        npf = mole.features[Mole.pf].max() + 1
        dy = np.sqrt(np.sum((mole.pos[0] - mole.pos[1])**2))  # longitudinal shift
        dx = np.sqrt(np.sum((mole.pos[0] - mole.pos[npf])**2))  # lateral shift
        
        max_shifts = tuple(np.array([dy*0.6, dy*0.6, dx*0.6])/_scale)
        img = loader.average()
        
        if bin_size > 1 and img.shape != template.shape:
            # if multiscaled image is used, there could be shape mismatch
            sl = tuple(slice(0, s) for s in template.shape)
            img = img[sl]

        from scipy.spatial.transform import Rotation
        model_cls = _get_alignment(method)
        model = model_cls(
            template.value,
            mask,
            cutoff=1.0,
            rotations=(z_rotation, y_rotation, x_rotation),
            tilt_range=None,  # NOTE: because input is an average
        )
        img_trans, result = model.fit(img, max_shifts=max_shifts)
        rotator = Rotation.from_quat(result.quat)
        mole_trans = mole.linear_transform(result.shift * _scale, rotator)
        
        # logging
        self.log.print_html(f"<code>align_averaged</code> ({default_timer() - t0:.1f} sec)")
        shift_nm = result.shift * _scale
        vec_str = ", ".join(f"{x}<sub>shift</sub>" for x in "XYZ")
        rotvec_str = ", ".join(f"{x}<sub>rot</sub>" for x in "XYZ")
        shift_nm_str = ", ".join(f"{s:.2f} nm" for s in shift_nm[::-1])
        rot_str = ", ".join(f"{s:.2f}" for s in rotator.as_rotvec()[::-1])
        self.log.print_html(f"{rotvec_str} = {rot_str}, {vec_str} = {shift_nm_str}")

        self._need_save = True
        img_trans = ip.asarray(img_trans, axes="zyx")
        img_trans.set_scale(zyx=_scale)
        
        @thread_worker.to_callback
        def _align_averaged_on_return():
            points = add_molecules(
                self.parent_viewer, 
                mole_trans,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
            )
            img_norm = utils.normalize_image(img_trans)
            temp_norm = utils.normalize_image(template)
            merge: np.ndarray = np.stack([img_norm, temp_norm, img_norm], axis=-1)
            layer.visible = False
            self.log.print(f"{layer.name!r} --> {points.name!r}")
            with self.log.set_plt():
                widget_utils.plot_projections(merge)

        return _align_averaged_on_return

    @_subtomogram_averaging.Refinement.wraps
    @set_design(text="Align all")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Alignment of {!r}"))
    def align_all(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        tilt_range: Bound[_subtomogram_averaging.tilt_range] = None,
        max_shifts: _MaxShifts = (1., 1., 1.),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
        {layer}{template_path}{mask_params}{tilt_range}{max_shifts}{z_rotation}{y_rotation}
        {x_rotation}{cutoff}{interpolation}{method}{bin_size}
        """
        t0 = default_timer()
        molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        
        loader, template, mask = self._check_binning_for_alignment(
            template, mask, binsize=bin_size, molecules=molecules, order=interpolation,
        )
        model_cls = _get_alignment(method)
        aligned_loader = loader.align(
            template=template.value, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=model_cls,
            tilt_range=tilt_range,
        )
        
        self.log.print_html(f"<code>align_all</code> ({default_timer() - t0:.1f} sec)")
        self._need_save = True
        return self._align_all_on_return(aligned_loader, layer)
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(size={"min": 1., "max": 100., "label": "sub-volume size (nm)"})
    @set_design(text="Align all (template-free)")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Template-free alignment of {!r}"))
    def align_all_template_free(
        self,
        layer: MonomerLayer,
        tilt_range: Bound[_subtomogram_averaging.tilt_range] = None,
        size: nm = 12.,
        max_shifts: _MaxShifts = (1., 1., 1.),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
        {layer}{tilt_range}{size}{max_shifts}{z_rotation}{y_rotation}{x_rotation}{cutoff}
        {interpolation}{method}{bin_size}
        """
        t0 = default_timer()
        molecules = layer.metadata[MOLECULES]
        loader, _, _ = self._check_binning_for_alignment(
            template=None, 
            mask=None, 
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
            shape=(size,)*3,
        )
        model_cls = _get_alignment(method)
        aligned_loader = loader.align_no_template(
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=model_cls,
            tilt_range=tilt_range,
        )
        
        self.log.print_html(f"<code>align_all_template_free</code> ({default_timer() - t0:.1f} sec)")
        self._need_save = True
        return self._align_all_on_return(aligned_loader, layer)
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(other_templates={"filter": FileFilter.IMAGE})
    @set_design(text="Align all (multi-template)")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Multi-template alignment of {!r}"))
    def align_all_multi_template(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        other_templates: list[Path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        tilt_range: Bound[_subtomogram_averaging.tilt_range] = None,
        max_shifts: _MaxShifts = (1., 1., 1.),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
        {layer}{template_path}
        other_templates : list of Path or str
            Path to other template images.
        {mask_params}{tilt_range}{max_shifts}{z_rotation}{y_rotation}{x_rotation}{cutoff}
        {interpolation}{method}{bin_size}
        """
        t0 = default_timer()
        molecules = layer.metadata[MOLECULES]
        templates = [self._subtomogram_averaging._get_template(path=template_path)]
        for path in other_templates:
            img = ip.imread(path)
            scale_ratio = img.scale.x / self.tomogram.scale
            if scale_ratio < 0.99 or 1.01 < scale_ratio:
                img = img.rescale(scale_ratio)
            templates.append(img)

        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        loader, templates, mask = self._check_binning_for_alignment(
            templates,
            mask,
            binsize=bin_size,
            molecules=molecules, 
            order=interpolation,
        )
        model_cls = _get_alignment(method)
        aligned_loader = loader.align_multi_templates(
            templates=[np.asarray(t) for t in templates], 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=model_cls,
            tilt_range=tilt_range,
        )
        self.log.print_html(f"<code>align_all_multi_template</code> ({default_timer() - t0:.1f} sec)")
        self._need_save = True
        return self._align_all_on_return(aligned_loader, layer)
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        distance_range={"options": {"min": 0.0, "max": 10.0, "step": 0.1}, "label": "distance range (nm)"},
        upsample_factor={"min": 1, "max": 20},
    )
    @set_design(text="Viterbi Alignment")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Viterbi-alignment of {!r}"))
    def align_all_viterbi(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params] = None,
        tilt_range: Bound[_subtomogram_averaging.tilt_range] = None,
        max_shifts: _MaxShifts = (0.6, 0.6, 0.6),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        distance_range: tuple[nm, nm] = (3.9, 4.4),
        max_angle: Optional[float] = 6.0,
        upsample_factor: int = 5,
    ):
        """
        Constrained subtomogram alignment using ZNCC landscaping and Viterbi algorithm.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{tilt_range}{max_shifts}{z_rotation}{y_rotation}
        {x_rotation}{cutoff}{interpolation}
        distance_range : tuple of float, default is (3.9, 4.4)
            Range of allowed distance between monomers.
        upsample_factor : int, default is 5
            Upsampling factor of ZNCC landscape. Be careful not to set this parameter too 
            large. Calculation will take much longer for larger ``upsample_factor``. 
            Doubling ``upsample_factor`` results in 2^6 = 64 times longer calculation time.
        """
        from dask import array as da
        from dask import delayed
        t0 = default_timer()
        molecules: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        shape_nm = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape=shape_nm, order=interpolation
        )
        if max_angle is not None:
            max_angle = np.deg2rad(max_angle)
        max_shifts_px = tuple(s / self.tomogram.scale for s in max_shifts)
        search_size = tuple(int(px * upsample_factor) * 2 + 1 for px in max_shifts_px)
        self.log.print_html(f"Search size (px): {search_size}")
        model = ZNCCAlignment(
            template.value,
            mask,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            tilt_range=tilt_range
        )
        
        templates_ft = model._get_template_input()  # 3D (no rotation) or 4D (has rotation)
        
        def func(img0: np.ndarray, template_ft: ip.ImgArray, max_shifts, quat):
            img0 = ip.asarray(img0 * mask, axes="zyx").lowpass_filter(cutoff=cutoff)
            template_ft = template_ft * model._get_missing_wedge_mask(quat)
            lds = utils.zncc_landscape(
                img0, template_ft.ifft(shift=False), max_shifts=max_shifts, upsample_factor=upsample_factor
            )
            return np.asarray(lds)
        
        has_rotation = templates_ft.ndim > 3
        if not has_rotation:
            tasks = loader.construct_mapping_tasks(
                func,
                ip.asarray(templates_ft, axes="zyx"),
                max_shifts=max_shifts_px,
                var_kwarg={"quat": molecules.quaternion()},
            )
            score = np.stack(da.compute(tasks)[0], axis=0)
        else:
            all_tasks = [
                da.stack(
                    [
                        da.from_delayed(a, shape=search_size, dtype=np.float32)
                        for a in loader.construct_mapping_tasks(
                            func,
                            ip.asarray(template_ft, axes="zyx"),
                            max_shifts=max_shifts_px,
                            var_kwarg={"quat": molecules.quaternion()},
                        )
                    ],
                    axis=0,
                )
                for template_ft in templates_ft
            ]
            all_tasks = da.stack(all_tasks, axis=0)
            tasks = da.max(all_tasks, axis=0)
            argmax = da.argmax(all_tasks, axis=0)
            out = da.compute([tasks, argmax], argmax)[0]
            score, argmax = out

        scale = self.tomogram.scale
        npf = molecules.features[Mole.pf].max() + 1
        
        slices = [np.asarray(molecules.features[Mole.pf] == i) for i in range(npf)]
        offset = np.array(shape_nm) / 2 - scale
        molecules_origin = molecules.translate_internal(-offset)
        mole_list = [molecules_origin.subset(sl) for sl in slices]  # split each protofilament
        
        dist_min, dist_max = np.array(distance_range) / scale * upsample_factor
        scores = [score[sl] for sl in slices]

        delayed_viterbi = delayed(utils.viterbi)
        viterbi_tasks = [
            delayed_viterbi(s, m.pos / scale * upsample_factor, m.z, m.y, m.x, dist_min, dist_max, max_angle)
            for s, m in zip(scores, mole_list)
        ]
        vit_out: list[tuple[np.ndarray, float]] = da.compute(viterbi_tasks)[0]
        
        offset = (np.array(max_shifts_px) * upsample_factor).astype(np.int32)
        all_shifts_px = np.empty((len(molecules), 3), dtype=np.float32)
        for i, (shift, _) in enumerate(vit_out):
            all_shifts_px[slices[i], :] = (shift - offset) / upsample_factor
        all_shifts = all_shifts_px * scale
        
        molecules_opt = molecules.translate_internal(all_shifts)
        if has_rotation:
            quats = np.zeros((len(molecules), 4), dtype=np.float32)
            for i, (shift, _) in enumerate(vit_out):
                _sl = slices[i]
                sub_quats = quats[_sl, :]
                for j, each_shift in enumerate(shift):
                    idx = argmax[_sl, :][j, each_shift[0], each_shift[1], each_shift[2]]
                    sub_quats[j] = model.quaternions[idx]
                quats[_sl, :] = sub_quats

            molecules_opt = molecules_opt.rotate_by_quaternion(quats)
            from scipy.spatial.transform import Rotation
            rotvec = Rotation.from_quat(quats).as_rotvec()
            molecules_opt.features = molecules_opt.features.with_columns(
                [
                    pl.Series("rotvec-z", rotvec[:, 0]),
                    pl.Series("rotvec-y", rotvec[:, 1]),
                    pl.Series("rotvec-x", rotvec[:, 2]),
                ]
            )
        
        molecules_opt.features = molecules_opt.features.with_columns(
            [
                pl.Series("shift-z", all_shifts[:, 0]),
                pl.Series("shift-y", all_shifts[:, 1]),
                pl.Series("shift-x", all_shifts[:, 2]),
            ]
        )
        self.log.print_html(f"<code>align_all_viterbi</code> ({default_timer() - t0:.1f} sec)")
        self._need_save = True
        aligned_loader = SubtomogramLoader(
            self.tomogram.image.value, 
            molecules_opt, 
            order=interpolation, 
            output_shape=template.shape,
        )
        return self._align_all_on_return(aligned_loader, layer)

    @_subtomogram_averaging.Refinement.wraps
    @set_options(molecule_subset={"text": "Use all molecules", "options": {"value": 200, "min": 1}})
    @set_design(text="Polarity check")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Polarity check of {!r}"))
    def polarity_check(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        max_shifts: _MaxShifts = (1., 1., 1.),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        method: OneOf[METHOD_CHOICES] = "zncc",
        molecule_subset: Optional[int] = None,
    ):
        """
        Check/determine the polarity by forward/reverse alignment.
        
        This method first conducts two alignment tasks using forward and revernse templates.
        Input template image will be considered as the forward template, while reverse
        template will be created by 180-degree rotation in XY plane. To avoid template bias,
        input template image will first be centered at its geometrical centroid. The
        orientations of molecules will be updated **in-place**.
        
        Parameters
        ----------
        {layer}{template_path}{max_shifts}{z_rotation}{y_rotation}{x_rotation}{cutoff}
        {interpolation}{method}
        molecule_subset : int, optional
            If specified, only a subset of molecules will be used to speed up polarity 
            determination.
        """
        t0 = default_timer()
        molecules: Molecules = layer.metadata[MOLECULES]
        if molecule_subset is not None:
            molecules = molecules.subset(slice(0, molecule_subset))
        template = self._subtomogram_averaging._get_template(path=template_path)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape, order=interpolation
        )
        
        # center template image by its centroid
        from skimage.measure import moments
        mom = moments(template, order=1)
        centroid = np.array([mom[1, 0, 0], mom[0, 1, 0], mom[0, 0, 1]]) / mom[0, 0, 0]
        shift = centroid - np.array(template.shape)/2 + 0.5
        template_centered_fw = template.affine(translation=shift)
        mask_fw = template_centered_fw.threshold().smooth_mask(sigma=1., dilate_radius=1.)
        template_centered_rv = template_centered_fw[:, ::-1, ::-1]
        mask_rv = mask_fw[:, ::-1, ::-1]
        
        # forward/reverse alignment
        model_cls = _get_alignment(method)
        loader_kwargs = dict(
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=model_cls,
        )
        aligned_loader_fw = loader.align(
            template=template_centered_fw, mask=mask_fw, **loader_kwargs,
        )
        aligned_loader_rv = loader.align(
            template=template_centered_rv, mask=mask_rv, **loader_kwargs,
        )
        
        # calculate forward/reverse averages.
        dask_array_fw = aligned_loader_fw.construct_dask()
        dask_array_rv = aligned_loader_rv.construct_dask()
        from dask import array as da
        avg_fw, avg_rv = da.compute(
            [da.mean(dask_array_fw, axis=0), 
             da.mean(dask_array_rv, axis=0)]
        )[0]
        
        avg_fw = ip.asarray(avg_fw, axes="zyx")
        avg_fw.set_scale(template)
        avg_rv = ip.asarray(avg_rv, axes="zyx")
        avg_rv.set_scale(template)
        
        # calculate cross correlations.
        zncc_fw = ip.zncc(template_centered_fw, avg_fw, mask=mask_fw>0.5)
        zncc_rv = ip.zncc(template_centered_rv, avg_rv, mask=mask_rv>0.5)
        if zncc_fw < zncc_rv:
            molecules.rotate_by_rotvec_internal([np.pi, 0., 0.], copy=False)
        self.log.print_html(f"<code>polarity_check</code> ({default_timer() - t0:.1f} sec)")

        @thread_worker.to_callback
        def _polarity_check_on_return():
            with self.log.set_plt():
                widget_utils.plot_forward_and_reverse(
                    template_centered_fw, avg_fw, zncc_fw,
                    template_centered_rv, avg_rv, zncc_rv,
                )
                plt.tight_layout()
                plt.show()

        return _polarity_check_on_return

    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_design(text="Calculate correlation")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Calculating correlation of {!r}"))
    def calculate_correlation(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging._get_template],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        show_average: bool = True,
    ):
        t0 = default_timer()
        molecules: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=1, order=interpolation
        )
        img_avg = ip.asarray(loader.average(), axes="zyx")
        img_avg.set_scale(zyx=loader.scale)
        zncc = ip.zncc(img_avg * mask, template * mask)
        self.log.print_html(f"<code>calculate_correlation</code> ({default_timer() - t0:.1f} sec)")
        self.log.print_html(f"Cross correlation with template = <b>{zncc:.3f}</b>")
        if not show_average:
            return
        return thread_worker.to_callback(
            self._subtomogram_averaging._show_reconstruction, img_avg, f"[AVG]{layer.name}"
        )

    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        seed={"text": "Do not use random seed."},
        n_set={"min": 1, "label": "number of image pairs"},
        dfreq={"label": "Frequency precision", "text": "Choose proper value", "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02}},
    )
    @set_design(text="Calculate FSC")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Calculating FSC of {!r}"))
    def calculate_fsc(
        self,
        layer: MonomerLayer,
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        size: _SubVolumeSize = None,
        seed: Optional[int] = 0,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        n_set: int = 1,
        show_average: bool = True,
        dfreq: Optional[float] = None,
    ):
        """
        Calculate Fourier Shell Correlation using the selected monomer layer.

        Parameters
        ----------
        {layer}{mask_params}{size}
        seed : int, optional
            Random seed used for subtomogram sampling.
        {interpolation}
        n_set : int, default is 1
            How many sets of image pairs will be generated to average FSC.
        show_average : bool, default is True
            If true, subtomogram averaging will be shown after FSC calculation.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        t0 = default_timer()
        mole: Molecules = layer.metadata[MOLECULES]
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        loader = self.tomogram.get_subtomogram_loader(
            mole,
            shape,
            order=interpolation,
        )
        if mask is None:
            mask = 1.
        if dfreq is None:
            dfreq = 1.5 / min(shape) * loader.scale
        img = ip.asarray(
            loader.average_split(n_set=n_set, seed=seed, squeeze=False),
            axes="ipzyx",
        )
        
        fsc_all: list[np.ndarray] = []
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = ip.fsc(img0*mask, img1*mask, dfreq=dfreq)
            fsc_all.append(fsc)
        if show_average:
            img_avg = ip.asarray(img[0, 0] + img[0, 1], axes="zyx") / len(mole)
            img_avg.set_scale(zyx=loader.scale)
        else:
            img_avg = None
            
        fsc_all = np.stack(fsc_all, axis=1)
        self.log.print_html(f"<code>calculate_fsc</code> ({default_timer() - t0:.1f} sec)")
        
        @thread_worker.to_callback
        def _calculate_fsc_on_return():
            fsc_mean = np.mean(fsc_all, axis=1)
            fsc_std = np.std(fsc_all, axis=1)
            crit_0143 = 0.143
            crit_0500 = 0.500
            
            self.log.print_html(f"<b>Fourier Shell Correlation of {layer.name!r}</b>")
            with self.log.set_plt(rc_context={"font.size": 15}):
                widget_utils.plot_fsc(freq, fsc_mean, fsc_std, [crit_0143, crit_0500], self.tomogram.scale)
            
            resolution_0143 = widget_utils.calc_resolution(freq, fsc_mean, crit_0143, self.tomogram.scale)
            resolution_0500 = widget_utils.calc_resolution(freq, fsc_mean, crit_0500, self.tomogram.scale)
            str_0143 = "N.A." if resolution_0143 == 0 else f"{resolution_0143:.3f} nm"
            str_0500 = "N.A." if resolution_0500 == 0 else f"{resolution_0500:.3f} nm"
            
            self.log.print_html(f"Resolution at FSC=0.5 ... <b>{str_0500}</b>")
            self.log.print_html(f"Resolution at FSC=0.143 ... <b>{str_0143}</b>")
            self._LoggerWindow.show()
            
            if img_avg is not None:
                _rec_layer: "Image" = self._subtomogram_averaging._show_reconstruction(
                    img_avg, name = f"[AVG]{layer.name}",
                )
                _rec_layer.metadata["FSC-freq"] = freq
                _rec_layer.metadata["FSC-mean"] = fsc_mean
        return _calculate_fsc_on_return
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(npf={"text": "Use global properties"})
    @set_design(text="Seam search")
    @dask_thread_worker.with_progress(desc=_fmt_layer_name("Seam search of {!r}"))
    def seam_search(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        npf: Optional[int] = None,
        cutoff: _CutoffFreq = 0.5,
    ):
        """
        Search for the best seam position.
        
        Try all patterns of seam positions and compare cross correlation values. If molecule
        assembly has 13 protofilaments, this method will try 26 patterns.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{interpolation}
        npf : int, optional
            Number of protofilaments. By default the global properties stored in the 
            corresponding spline will be used.
        {cutoff}
        """
        mole: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(mole, shape, order=interpolation)
        if npf is None:
            npf = mole.features[Mole.pf].max() + 1

        corrs, img_ave, all_labels = utils.try_all_seams(
            loader=loader, npf=npf, template=template, mask=mask, cutoff=cutoff
        )
        
        self._need_save = True

        @thread_worker.to_callback
        def _seam_search_on_return():
            self._subtomogram_averaging._show_reconstruction(img_ave, layer.name)
            self._LoggerWindow.show()
            
            # calculate score and the best PF position
            corr1, corr2 = corrs[:npf], corrs[npf:]
            score = np.empty_like(corrs)
            score[:npf] = corr1 - corr2
            score[npf:] = corr2 - corr1
            imax = np.argmax(score)
                
            # plot all the correlation
            self.log.print_html("<code>Seam_search</code>")
            with self.log.set_plt(rc_context={"font.size": 15}):
                self.log.print(f"layer = {layer.name!r}")
                self.log.print(f"template = {str(template_path)!r}")
                widget_utils.plot_seam_search_result(score, npf)
                
            self.sub_viewer.layers[-1].metadata["Correlation"] = corrs
            self.sub_viewer.layers[-1].metadata["Score"] = score
            
            update_features(layer, {Mole.isotype: all_labels[imax].astype(np.uint8)})
            layer.metadata["seam-search-score"] = score
        
        return _seam_search_on_return

    @_subtomogram_averaging.Tools.wraps
    @set_options(feature_name={"text": "Do not color molecules."})
    @set_design(text="Render molecules")
    def render_molecules(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        feature_name: Optional[str] = None,
        cutoff: _CutoffFreq = 0.5,
    ):
        """
        Render molecules using the template image.
        
        This method is only for visualization purpose. Iso-surface will be calculated
        using the input template image and mapped to every molecule position. The input
        template image does not have to be the image used for subtomogram alignment.

        Parameters
        ----------
        {layer}{template_path}{mask_params}
        feature_name : str, optional
            Feature name used for coloring.
        cutoff : float, optional
            Cutoff frequency of low-pass filter to smooth template image. This parameter
            is for visualization only.
        """        
        from skimage.measure import marching_cubes

        # prepare template and mask
        template = self._subtomogram_averaging._get_template(template_path).copy()
        if cutoff is not None:
            with utils.set_gpu():
                template.lowpass_filter(cutoff=cutoff, update=True)
        soft_mask = self._subtomogram_averaging._get_mask(mask_params)
        if soft_mask is None:
            mask = np.ones_like(template)
        else:
            mask = soft_mask > 0.2
            template[~mask] = template.min()
        
        mole: Molecules = layer.metadata[MOLECULES]
        nmol = len(mole)

        # check feature name
        if feature_name is not None:
            if feature_name in layer.features.columns:
                pass
            elif getattr(Mole, feature_name, "") in layer.features.columns:
                feature_name = getattr(Mole, feature_name)
            if feature_name not in layer.features.columns:
                raise ValueError(
                    f"Feature {feature_name} not found in layer {layer.name}. Must be in "
                    f"{set(layer.features.columns)}"
                )
                
        # create surface
        verts, faces, _, _ = marching_cubes(
            template, step_size=1, spacing=template.scale, mask=mask,
        )
        
        nverts = verts.shape[0]
        all_verts = np.empty((nmol * nverts, 3), dtype=np.float32)
        all_faces = np.concatenate([faces + i*nverts for i in range(nmol)], axis=0)
        center = np.array(template.shape)/2 + 0.5
        
        for i, v in enumerate(verts):
            v_transformed = mole.rotator.apply(v - center * np.array(template.scale)) + mole.pos
            all_verts[i::nverts] = v_transformed
        
        if feature_name is None:
            data = (all_verts, all_faces)
        else:
            all_values = np.stack([layer.features[feature_name]]*nverts, axis=1).ravel()
            data = (all_verts, all_faces, all_values)
            
        self.parent_viewer.add_surface(
            data=data, 
            colormap=self.label_colormap, 
            shading="smooth",
            name=f"Rendered {layer.name}",
        )
        return None
    
    @impl_preview(render_molecules)
    def _preview_rendering(self, template_path: str, mask_params, cutoff: float):
        from skimage.measure import marching_cubes

        # prepare template and mask
        template = self._subtomogram_averaging._get_template(template_path).copy()
        if cutoff is not None:
            with utils.set_gpu():
                template.lowpass_filter(cutoff=cutoff, update=True)
        soft_mask = self._subtomogram_averaging._get_mask(mask_params)
        if soft_mask is None:
            mask = np.ones_like(template)
        else:
            mask = soft_mask > 0.2
            template[~mask] = template.min()
            
        # create surface
        verts, faces, _, _ = marching_cubes(
            template, step_size=1, spacing=template.scale, mask=mask > 0.2,
        )
        _previews.view_surface([verts, faces], parent=self)
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
        all_localprops = tomo.collect_localprops()
        if all_localprops is None:
            raise ValueError("No local property found.")
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        binsize = utils.roundint(bin_scale/tomo.scale)
        ft_size = self._current_ft_size
        
        lz, ly, lx = [utils.roundint(r/bin_scale*1.73)*2 + 1 for r in [15, ft_size/2, 15]]
        center = np.array([lz, ly, lx])/2 + 0.5
        z, y, x = np.indices((lz, ly, lx))
        cylinders = []
        matrices = []
        for i, spl in enumerate(tomo.splines):
            # Prepare template hollow image
            r0 = spl.radius/tomo.scale*0.9/binsize
            r1 = spl.radius/tomo.scale*1.1/binsize
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
                
            cylinders.append(domains)
            matrices.append(spl.affine_matrix(center=center, inverse=True))
        
        cylinders = np.concatenate(cylinders, axis=0)
        matrices = np.concatenate(matrices, axis=0)
        out = _multi_affine(cylinders, matrices) > 0.3
            
        # paint roughly
        for i, crd in enumerate(tomo.collect_anchor_coords()):
            center = tomo.nm2pixel(crd, binsize=binsize)
            sl = []
            outsl = []
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape):
                _sl, _pad = utils.make_slice_and_pad(c - l//2, c + l//2 + 1, size)
                sl.append(_sl)
                outsl.append(
                    slice(_pad[0] if _pad[0] > 0 else None,
                         -_pad[1] if _pad[1] > 0 else None)
                )

            sl = tuple(sl)
            outsl = tuple(outsl)
            lbl[sl][out[i][outsl]] = i + 1
        
        # paint finely
        ref = self.layer_image.data
        thr = np.percentile(ref[lbl>0], 5)
        lbl[ref<thr] = 0
        
        # Labels layer properties
        _id = "ID"
        _structure = "structure"
        columns = [_id, H.riseAngle, H.yPitch, H.skewAngle, _structure]
        df = all_localprops[[H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]]
        df_reset = df.reset_index()
        df_reset[_id] = df_reset.apply(
            lambda x: "{}-{}".format(int(x["SplineID"]), int(x["PosID"])), 
            axis=1
            )
        df_reset[_structure] = df_reset.apply(
            lambda x: "{npf}_{start:.1f}".format(npf=int(x[H.nPF]), start=x[H.start]), 
            axis=1
            )
        
        back = pd.DataFrame({c: [np.nan] for c in columns})
        props = pd.concat([back, df_reset[columns]])
        
        # Add labels layer
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
        return None
    
    @Image.wraps
    @set_options(
        limit={"options": {"min": -20, "max": 20, "step": 0.01}, "label": "limit (nm)"},
        auto_call=True,
    )
    @set_design(text="Set colormap")
    def set_colormap(
        self,
        start: Color = (0, 0, 1, 1),
        end: Color = (1, 0, 0, 1),
        limit: tuple[float, float] = (4.00, 4.24), 
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
        self.label_colormap = Colormap([start, end], name="LocalProperties")
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
        self._LoggerWindow.show()
        with self.log.set_plt(rc_context={"font.size": 15}):
            plt.imshow(arr)
            plt.xticks([0, 27], [f"{xmin:.2f}", f"{xmax:.2f}"])
            plt.yticks([], [])
            plt.show()
        return None
    
    @Image.wraps
    @set_design(text="Simulate cylindric structure")
    @do_not_record
    def open_simulator(self):
        return self._Simulator.show()
    
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
                if MOLECULES in layer.metadata.keys():
                    name = layer.name
                    break
            else:
                raise ValueError("No molecules found in the layer list.")
        return self.parent_viewer.layers[name].metadata[MOLECULES]

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
        shape = self._subtomogram_averaging._get_shape_in_nm()
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
    
    def _check_binning_for_alignment(
        self,
        template: Union[ip.ImgArray, list[ip.ImgArray]],
        mask: Union[ip.ImgArray, None],
        binsize: int,
        molecules: Molecules,
        order: int = 1,
        shape: tuple[nm, nm, nm] = None,
    ) -> tuple[SubtomogramLoader, ip.ImgArray, Union[np.ndarray, None]]:
        """
        Returns proper subtomogram loader, template image and mask image that matche the 
        bin size.
        """
        if shape is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=binsize, order=order
        )
        if binsize > 1:
            if template is None:
                pass
            elif isinstance(template, list):
                template = [tmp.binning(binsize, check_edges=False) for tmp in template]
            else:
                template = template.binning(binsize, check_edges=False)
            if mask is not None:
                mask = mask.binning(binsize, check_edges=False)
        if isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        return loader, template, mask
    
    @thread_worker.to_callback
    def _align_all_on_return(self, aligned_loader: SubtomogramLoader, layer: MonomerLayer):
        points = add_molecules(
            self.parent_viewer, 
            aligned_loader.molecules,
            name=_coerce_aligned_name(layer.name, self.parent_viewer),
        )
        layer.visible = False
        self.log.print(f"{layer.name!r} --> {points.name!r}")
        return None

    def _update_colormap(self, prop: str = H.yPitch):
        if self.layer_paint is None:
            return None
        color = {0: np.array([0., 0., 0., 0.], dtype=np.float32),
                 None: np.array([0., 0., 0., 1.], dtype=np.float32)}
        lim0, lim1 = self.label_colorlimit
        df = self.tomogram.collect_localprops()[prop]
        for i, value in enumerate(df):
            color[i+1] = self.label_colormap.map((value - lim0)/(lim1 - lim0))
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
        self.log.print_html(f"<h2>{_name}</h2>")
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
        if MOLECULES in layer.metadata.keys():
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
            if MOLECULES in layer.metadata.keys():
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
            pitch, skew, npf, start = spl.globalprops[headers]
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
            headers = [H.yPitch, H.skewAngle, H.nPF, H.start]
            pitch, skew, npf, start = spl.localprops[headers].iloc[j]
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
    
    @average_all.started.connect
    @align_averaged.started.connect
    @align_all.started.connect
    @calculate_fsc.started.connect
    def _show_subtomogram_averaging(self):
        return self._subtomogram_averaging.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Preview methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @impl_preview(load_project)
    def _preview_text(self, path: str):
        pviewer = CylindraProject.from_json(path).make_project_viewer()
        pviewer.native.setParent(self.native, pviewer.native.windowFlags())
        return pviewer.show()
    
    @impl_preview(load_molecules)
    def _preview_table(self, paths: list[str]):
        return _previews.view_tables(paths, parent=self)

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

def _coerce_aligned_name(name: str, viewer: "napari.Viewer"):
    num = 1
    if re.match(fr".*-{ALN_SUFFIX}(\d)+", name):
        try:
            *pre, suf = name.split(f"-{ALN_SUFFIX}")
            num = int(suf) + 1
            name = "".join(pre)
        except Exception:
            num = 1
    
    existing_names = set(layer.name for layer in viewer.layers)
    while name + f"-{ALN_SUFFIX}{num}" in existing_names:
        num += 1
    return name + f"-{ALN_SUFFIX}{num}"
