import os
import re
from typing import Iterable, Iterator, Union, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import napari
from napari.utils import Colormap
from napari.qt import create_worker
from napari.layers import Points, Image, Labels

import impy as ip

from magicclass import (
    magicclass,
    magictoolbar,
    magicmenu,
    field,
    vfield,
    set_design,
    set_options,
    do_not_record,
    Bound,
    Optional,
    MagicTemplate,
    bind_key,
    build_help,
    nogui
    )
from magicclass.widgets import (
    TupleEdit,
    ListEdit,
    Separator,
    RadioButtons,
    ColorEdit,
    ConsoleTextEdit,
    Figure,
    Container,
    Select,
    )
from magicclass.ext.pyqtgraph import QtImageCanvas

from ..components import SubtomogramLoader, Molecules, MtSpline, MtTomogram
from ..components.tomogram import angle_corr
from ..utils import (
    crop_tomogram,
    make_slice_and_pad,
    map_coordinates,
    mirror_pcc,
    pad_template, 
    pad_mt_edges,
    roundint,
    ceilint,
    set_gpu
    )
from ..const import nm, H, Ori, GVar, Sep, Mole
from ..const import WORKING_LAYER_NAME, SELECTION_LAYER_NAME, SOURCE, ALN_SUFFIX, MOLECULES
from ..types import MonomerLayer

from .properties import GlobalPropertiesWidget, LocalPropertiesWidget
from .spline_control import SplineControl
from .spline_fitter import SplineFitter
from .tomogram_list import TomogramList
from .worker import WorkerControl, dispatch_worker, Worker
from .widget_utils import add_molecules, change_viewer_focus, update_features
from ..ext.etomo import PEET

ICON_DIR = Path(__file__).parent / "icons"
SPLINE_ID = "spline-id"

### The main widget ###
    
@magicclass(widget_type="scrollable", name="MTProps widget")
class MTPropsWidget(MagicTemplate):
    # Main GUI class.
    
    ### widgets ###
    
    _WorkerControl = field(WorkerControl, name="Worker control")
    _SplineFitter = field(SplineFitter, name="Spline fitter")
    _TomogramList = field(TomogramList, name="Tomogram list")
    
    @magicmenu
    class File(MagicTemplate):
        """File I/O."""  
        def Open_image(self): ...
        def Open_tomogram_list(self): ...
        def Load_json(self): ...
        sep0 = field(Separator)
        def Save_results_as_json(self): ...
        def Save_results_as_csv(self): ...
        def Save_molecules(self): ...
        sep1 = field(Separator)
        PEET = PEET

    @magicmenu
    class Image(MagicTemplate):
        """Image processing and visualization"""
        def Show_image_info(self): ...
        def Apply_lowpass_to_reference_image(self): ...
        def Invert_tomogram(self): ...
        sep0 = field(Separator)
        def show_current_ft(self): ...
        def show_global_ft(self): ...
        def show_r_proj(self): ...
        def show_global_r_proj(self): ...
        sep1 = field(Separator)
        def Sample_subtomograms(self): ...
        def Paint_MT(self): ...
        def Set_colormap(self): ...
        def Show_colorbar(self): ...
    
    @magicmenu
    class Splines(MagicTemplate):
        """Spline fitting and operations."""
        def Show_splines(self): ...
        def Add_anchors(self): ...
        sep0 = field(Separator)
        def Invert_spline(self): ...
        def Align_to_polarity(self): ...
        def Clip_spline(self): ...
        sep1 = field(Separator)
        def Fit_splines(self): ...
        def Fit_splines_manually(self): ...
        def Refine_splines(self): ...

    @magicmenu
    class Molecules(MagicTemplate):
        def Show_orientation(self): ...
        def Calculate_intervals(self): ...
        def Show_isotypes(self): ...
        
    @magicmenu
    class Analysis(MagicTemplate):
        """Analysis of tomograms."""
        def Set_radius(self): ...
        def Local_FT_analysis(self): ...
        def Global_FT_analysis(self): ...
        sep0 = field(Separator)
        @magicmenu
        class Mapping(MagicTemplate):
            def Map_monomers(self): ...
            def Map_monomers_manually(self): ...
            def Map_centers(self): ...
            def Map_along_PF(self): ...
        def Open_subtomogram_analyzer(self): ...
    
    @magicmenu
    class Others(MagicTemplate):
        """Other menus."""
        def Open_help(self): ...
        def Create_macro(self): ...
        def Global_variables(self): ...
        def Clear_cache(self): ...
        def MTProps_info(self): ...
        
    @magictoolbar(labels=False)
    class toolbar(MagicTemplate):
        """Frequently used operations."""        
        def register_path(self): ...
        def open_runner(self): ...
        sep0 = field(Separator)
        def pick_next(self): ...
        def auto_center(self): ...
        @magicmenu(icon_path=ICON_DIR/"adjust_intervals.png")
        class Adjust(MagicTemplate):
            """Adjust auto picker"""
            stride = vfield(50.0, widget_type="FloatSlider", options={"min": 10, "max": 100, "tooltip": "Stride length (nm) of auto picker"}, record=False)
            angle_deviation = vfield(25.0, widget_type="FloatSlider", options={"min": 1.0, "max": 40.0, "step": 0.5, "tooltip": "Angle deviation (degree) of auto picker"}, record=False)
            angle_precision = vfield(1.0, widget_type="FloatSlider", options={"min": 0.5, "max": 5.0, "step": 0.1, "tooltip": "Angle precision (degree) of auto picker"}, record=False)
        sep1 = field(Separator)
        def clear_current(self): ...
        def clear_all(self): ...
    
    SplineControl = SplineControl
    LocalProperties = field(LocalPropertiesWidget, name="Local Properties")
    GlobalProperties = field(GlobalPropertiesWidget, name="Global Properties")
    
    @magicclass(widget_type="tabbed")
    class Panels(MagicTemplate):
        """Panels for output."""
        overview = field(QtImageCanvas, name="Overview", options={"tooltip": "Overview of splines"})
        image2D = field(QtImageCanvas, options={"tooltip": "2-D image viewer."})
    
    ### methods ###
    
    def __init__(self):
        self.tomogram: MtTomogram = None
        self._last_ft_size: nm = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        
    def __post_init__(self):
        self.Set_colormap()
        self.min_width = 400
        self.LocalProperties.collapsed = False
        self.GlobalProperties.collapsed = False
        self.Panels.min_height = 300

    @property
    def sub_viewer(self) -> napari.Viewer:
        """Return the sub-viewer that is used for subtomogram averaging."""
        return self._subtomogram_averaging._viewer

    def _get_splines(self, widget=None) -> List[Tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        tomo = self.tomogram
        if tomo is None:
            return []
        return [(str(spl), i) for i, spl in enumerate(tomo.splines)]
        
    def _get_spline_coordinates(self, widget=None) -> np.ndarray:
        """Get coordinates of the manually picked spline."""
        coords = self.layer_work.data
        return np.round(coords, 3)
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"add_spline.png")
    @bind_key("F1")
    def register_path(self, coords: Bound[_get_spline_coordinates] = None):
        """Register current selected points as a MT path."""        
        if coords is None:
            coords = self.layer_work.data
        else:
            coords = np.asarray(coords)
        
        if coords.size == 0:
            return None

        tomo = self.tomogram
        tomo.add_spline(coords)
        spl = tomo.splines[-1]
        
        # draw path
        self._add_spline_to_images(spl, tomo.n_splines)
        self.layer_work.data = []
        self.layer_prof.selected_data = set()
        self.reset_choices()
        return None
    
    @magicclass(name="Run MTProps")
    class _runner(MagicTemplate):
        def _get_splines(self, widget=None) -> List[Tuple[str, int]]:
            """Get list of spline objects for categorical widgets."""
            try:
                tomo = self.find_ancestor(MTPropsWidget).tomogram
            except Exception:
                return []
            if tomo is None:
                return []
            return [(str(spl), i) for i, spl in enumerate(tomo.splines)]
        
        all_splines = vfield(True, options={"text": "Run for all the splines.", "tooltip": "Uncheck to select along which spline algorithms will be executed."}, record=False)
        splines = vfield(Select, options={"choices": _get_splines, "visible": False})
        dense_mode = vfield(True, options={"label": "Use dense-mode", "tooltip": "Check if microtubules are densely packed. Initial spline position must be 'almost' fitted in dense mode."}, record=False)
        @magicclass(widget_type="groupbox", name="Parameters")
        class params1:
            """Parameters used in spline fitting."""
            edge_sigma = vfield(2.0, options={"label": "edge sigma", "tooltip": "Sharpness of dense-mode mask at the edges."}, record=False)
            max_shift = vfield(5.0, options={"label": "Maximum shift (nm)", "max": 50.0, "step": 0.5, "tooltip": "Maximum shift in nm of manually selected spline to the true center."}, record=False)
        n_refine = vfield(1, options={"label": "Refinement iteration", "max": 4, "tooltip": "Iteration number of spline refinement."}, record=False)
        local_props = vfield(True, options={"label": "Calculate local properties", "tooltip": "Check if calculate local properties."}, record=False)
        @magicclass(widget_type="groupbox", name="Parameters")
        class params2:
            """Parameters used in calculation of local properties."""
            interval = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Interval (nm)", "tooltip": "Interval of sampling points of microtubule fragments."}, record=False)
            ft_size = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Local DFT window size (nm)", "tooltip": "Longitudinal length of local discrete Fourier transformation used for structural analysis."}, record=False)
            paint = vfield(True, options={"tooltip": "Check if paint microtubules after local properties are calculated."}, record=False)
        global_props = vfield(True, options={"label": "Calculate global properties", "tooltip": "Check if calculate global properties."}, record=False)

        @all_splines.connect
        def _toggle_spline_list(self):
            self["splines"].visible = not self.all_splines
            
        @dense_mode.connect
        def _toggle_dense_mode_sigma(self):
            self.params1["edge_sigma"].visible = self.dense_mode
        
        @local_props.connect
        def _toggle_localprops_params(self):
            self.params2.visible = self.local_props
        
        def _get_splines_to_run(self, w=None) -> List[int]:
            if self.all_splines:
                return []
            else:
                return self.splines
        
        def _get_edge_sigma(self, w=None) -> Union[float, None]:
            if self.dense_mode:
                return self.params1.edge_sigma
            else:
                return None
        
        def run_mtprops(self): ...
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"run_all.png")
    @do_not_record
    def open_runner(self):
        """Run MTProps with various settings."""
        self._runner.show()
        return None
    
    @_runner.wraps
    @set_design(text="Run")
    @dispatch_worker
    def run_mtprops(
        self,
        splines: Bound[_runner._get_splines_to_run] = (),
        interval: Bound[_runner.params2.interval] = 32.0,
        ft_size: Bound[_runner.params2.ft_size] = 32.0,
        n_refine: Bound[_runner.n_refine] = 1,
        max_shift: Bound[_runner.params1.max_shift] = 5.0,
        edge_sigma: Bound[_runner._get_edge_sigma] = 2.0,
        local_props: Bound[_runner.local_props] = True,
        global_props: Bound[_runner.global_props] = True,
        paint: Bound[_runner.params2.paint] = True,
    ):
        """Run MTProps"""
        self._runner.close()
        
        if self.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if self.tomogram.n_splines == 0:
            raise ValueError("No spline found.")
        elif len(splines) == 0:
            splines = range(self.tomogram.n_splines)
        
        total = (1 + n_refine + int(local_props) + int(global_props)) * len(splines)
        
        worker = create_worker(_iter_run, 
                               tomo=self.tomogram,
                               splines=splines,
                               interval=interval,
                               ft_size=ft_size,
                               n_refine=n_refine,
                               max_shift=max_shift,
                               edge_sigma=edge_sigma,
                               local_props=local_props,
                               global_props=global_props,
                               _progress={"total": total, 
                                          "desc": "Running MTProps"}
                               )
        
        @worker.yielded.connect
        def _on_yield(out):
            if isinstance(out, str):
                self._WorkerControl.info = out
                self._update_splines_in_images()
            
        @worker.returned.connect
        def _on_return(tomo: MtTomogram):
            self._update_splines_in_images()
            if local_props or global_props:
                self.Sample_subtomograms()
            if local_props:
                if paint:
                    self.Paint_MT()
            tomo.metadata["ft_size"] = self._last_ft_size
            if global_props:
                self._update_global_properties_in_widget()
        self._last_ft_size = ft_size
        self._WorkerControl.info = f"[1/{len(splines)}] Spline fitting"
        return worker
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"clear_last.png")
    @do_not_record
    def clear_current(self):
        """Clear current selection."""        
        self.layer_work.data = []
        return None
    
    @toolbar.wraps
    @set_options(_={"widget_type": "Label"})
    @set_design(icon_path=ICON_DIR/"clear_all.png")
    def clear_all(self, _="Are you sure to clear all?"):
        """Clear all the splines and results."""
        self._init_widget_state()
        self._init_layers()
        self.Panels.overview.layers.clear()
        self.tomogram.clear_cache()
        self.tomogram.splines.clear()
        self.reset_choices()
        return None

    @Others.wraps
    @do_not_record
    def Open_help(self):
        """Open a help window."""
        help = build_help(self)
        help.show()
        return None
    
    @Others.wraps
    @do_not_record
    def Create_macro(self):
        """Create Python executable script."""
        self.macro.widget.duplicate().show()
        return None
    
    @Others.wraps
    @set_options(
        yPitchMin={"step": 0.1},
        yPitchMax={"step": 0.1},
        minSkew={"min": -90, "max": 90},
        maxSkew={"min": -90, "max": 90},
        splError={"max": 5.0, "step": 0.1},
        inner={"step": 0.1},
        outer={"step": 0.1},
        daskChunk={"widget_type": TupleEdit, "options": {"min": 16, "max": 2048, "step": 16}},
        GPU={"label": "Use GPU if available"},
    )
    def Global_variables(
        self,
        nPFmin: int = GVar.nPFmin,
        nPFmax: int = GVar.nPFmax,
        splOrder: int = GVar.splOrder,
        yPitchMin: nm = GVar.yPitchMin,
        yPitchMax: nm = GVar.yPitchMax,
        minSkew: float = GVar.minSkew,
        maxSkew: float = GVar.maxSkew,
        splError: nm = GVar.splError,
        inner: float = GVar.inner,
        outer: float = GVar.outer,
        daskChunk: Tuple[int, int, int] = GVar.daskChunk,
        GPU: bool = GVar.GPU,
    ):
        """
        Set global variables.

        Parameters
        ----------
        nPFmin : int
            Minimum protofilament numbers. 
        nPFmax : int
            Maximum protofilament numbers.
        splOrder : int
            Maximum order of spline curve.
        yPitchMin : nm
            Minimum pitch length for estimation.
        yPitchMax : nm
            Maximum pitch length for estimation.
        minSkew : float
            Minimum skew angle for estimation.
        maxSkew : float
            Maximum skew angle for estimation.
        splError : nm
            Average error of spline fitting.
        inner : float
            Radius x inner will be the inner surface of MT.
        outer : float
            Radius x outer will be the outer surface of MT.
        """        
        GVar.set_value(**locals())
        if self.tomogram is not None:
            for spl in self.tomogram.splines:
                spl.localprops = None
                spl.globalprops = None
    
    @Others.wraps
    def Clear_cache(self):
        """Clear cache stored on the current tomogram."""
        if self.tomogram is not None:
            self.tomogram.clear_cache()
    
    @Others.wraps
    @do_not_record
    def MTProps_info(self):
        """Show information of dependencies."""
        import napari
        import magicgui
        from .. import __version__
        import magicclass as mcls
        import dask
        
        value = (
            f"MTProps: {__version__}\n"
            f"impy: {ip.__version__}\n"
            f"magicgui: {magicgui.__version__}\n"
            f"magicclass: {mcls.__version__}\n"
            f"napari: {napari.__version__}\n"
            f"dask: {dask.__version__}\n"
        )
        w = ConsoleTextEdit(value=value)
        w.read_only = True
        w.native.setParent(self.native, w.native.windowFlags())
        w.show()
        return None
    
    @magicclass
    class _loader(MagicTemplate):
        # A loader widget with imread settings.
        path = vfield(Path, record=False, options={"filter": "*.tif;*.tiff;*.mrc;*.rec", "tooltip": "Path to tomogram."})
        scale = vfield(str, record=False, options={"label": "scale (nm)", "tooltip": "Pixel size in nm/pixel."})
        bin_size = vfield(4, record=False, options={"label": "bin size", "min": 1, "max": 8, "tooltip": "Bin size of image for reference. This value does not affect MTProps analysis."})
        subtomo_length = vfield(48.0, record=False, options={"label": "subtomogram length (nm)", "min": 2.0, "max": 100.0, "step": 4.0, "tooltip": "The axial length of subtomogram."})
        subtomo_width = vfield(44.0, record=False, options={"label": "subtomogram width (nm)", "min": 2.0, "max": 100.0, "step": 4.0, "tooltip": "The diameter of subtomogram."})
        light_background = vfield(False, record=False, options={"label": "light background", "tooltip": "Check if background is bright."})
        use_lowpass = vfield(False, record=False, options={"label": "Apply low-pass filter","tooltip": "Check if images need prefilter."})
        cutoff_freq = vfield(0.2, record=False, options={"label": "Cutoff frequency (1/px)", "visible": False, "min": 0.0, "max": 0.85, "step": 0.05, "tooltip": "Relative cutoff frequency of low-pass prefilter. Must be 0.0 < freq < 0.866."})
        
        @use_lowpass.connect
        def _enable_freq_option(self):
            self["cutoff_freq"].visible = self.use_lowpass
        
        def _get_cutoff_freq(self, _=None):
            if self.use_lowpass:
                return self.cutoff_freq
            else:
                return 0.0
        
        @path.connect
        def _read_scale(self):
            img = ip.lazy_imread(self.path, chunks=GVar.daskChunk)
            scale = img.scale.x
            self.scale = f"{scale:.4f}"
            if scale > 0.96:
                self.bin_size = 1
            elif scale > 0.48:
                self.bin_size = 2
            else:
                self.bin_size = 4
        
        def load_tomogram(self): ...
    
    @_loader.wraps
    @set_design(text="OK")
    @dispatch_worker
    def load_tomogram(
        self, 
        path: Bound[_loader.path],
        scale: Bound[_loader.scale] = None,
        bin_size: Bound[_loader.bin_size] = None,
        light_background: Bound[_loader.light_background] = False,
        cutoff: Bound[_loader._get_cutoff_freq] = None,
        subtomo_length: Bound[_loader.subtomo_length] = 48.,
        subtomo_width: Bound[_loader.subtomo_width] = 44.,
    ):
        """Start loading image."""
        img = ip.lazy_imread(path, chunks=GVar.daskChunk)
        if scale is not None:
            scale = float(scale)
            img.scale.x = img.scale.y = img.scale.z = scale
        else:
            scale = img.scale.x
    
        if scale > 0.96:
            bin_size = 1
        elif scale > 0.48:
            bin_size = 2
        else:
            bin_size = 4
        
        if cutoff is None:
            cutoff = 1.0
        
        worker = self._get_process_image_worker(
            img, 
            path=path,
            binsize=bin_size,
            light_bg=light_background,
            cutoff=cutoff, 
            length=subtomo_length,
            width=subtomo_width,
            )
        
        self._loader.close()
        return worker
    
    @File.wraps
    @do_not_record
    def Open_image(self):
        """Open an image and add to viewer."""
        self._loader.show(run=False)
        return None
    
    @File.wraps
    @do_not_record
    def Open_tomogram_list(self):
        """Open the list of loaded tomogram references."""
        self._TomogramList.show()
        return None
        
    @File.wraps
    @set_options(path={"filter": "*.json;*.txt"})
    def Load_json(self, path: Path):
        """Choose a json file and load it."""        
        tomo = self.tomogram
        tomo.load_json(path)

        self._last_ft_size = tomo.metadata.get("ft_size", self._last_ft_size)
            
        self._update_splines_in_images()
        self.reset_choices()
        self.Sample_subtomograms()
        return None
    
    @File.wraps
    @set_design(text="Save results as json")
    @set_options(save_path={"mode": "w", "filter": "*.json;*.txt"})
    def Save_results_as_json(self, save_path: Path):
        """Save the results as json."""
        self.tomogram.save_json(save_path)
        return None
    
    @File.wraps
    @set_options(save_path={"mode": "w", "filter": "*.txt;*.csv;*.dat"})
    def Save_molecules(
        self,
        layer: MonomerLayer, 
        save_path: Path,
    ):
        """
        Save monomer coordinates.

        Parameters
        ----------
        layer : Points
            Select the points layer to save.
        save_path : Path
            Where to save the molecules.
        """
        separator = Sep(separator)
        mole: Molecules = layer.metadata[MOLECULES]
        mole.to_csv(save_path)
        return None
    
    @Image.wraps
    def Show_image_info(self):
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
    @dispatch_worker
    def Apply_lowpass_to_reference_image(self):
        """Apply low-pass filter to enhance contrast of the reference image."""
        cutoff = 0.2
        def func():
            with ip.silent(), set_gpu():
                self.layer_image.data = self.layer_image.data.tiled_lowpass_filter(
                    cutoff, chunks=(32, 128, 128)
                    )
            return np.percentile(self.layer_image.data, [1, 97])
        worker = create_worker(func, _progress={"total": 0, "desc": "Running"})
        self._WorkerControl.info = "Low-pass filtering"

        @worker.returned.connect
        def _on_return(contrast_limits):
            self.layer_image.contrast_limits = contrast_limits
            with ip.silent():
                proj = self.layer_image.data.proj("z")
            self.Panels.overview.image = proj
            self.Panels.overview.contrast_limits = contrast_limits
        
        return worker
    
    @Image.wraps
    @dispatch_worker
    def Invert_tomogram(self):
        """
        Invert intensity of tomogram and the reference image.
        
        This method will update each image but will not overwrite image file itself.
        A temporary memory-mapped file with inverted image is created which will be
        deleted after Python is closed.
        """
        tomo = self.tomogram
        def func():
            with ip.silent(), set_gpu():
                img_inv = -tomo.image
                img_inv.release()
                tomo._set_image(img_inv)
            return -self.layer_image.data
        
        worker = create_worker(func, _progress={"total": 0, "desc": "Running"})
        self._WorkerControl.info = "Inverting tomogram"
        
        @worker.returned.connect
        def _on_return(imgb_inv):
            self.layer_image.data = imgb_inv
            vmin, vmax = self.layer_image.contrast_limits
            clims = [-vmax, -vmin]
            self.layer_image.contrast_limits = clims
            self.Panels.overview.image = -self.Panels.overview.image
            self.Panels.overview.contrast_limits = clims
        
        return worker
    
    @SplineControl.num.connect
    @SplineControl.pos.connect
    @SplineControl.footer.focus.connect
    def _focus_on(self):
        """Change camera focus to the position of current MT fragment."""
        if self.layer_paint is None:
            return None
        if not self.SplineControl.footer.focus:
            self.layer_paint.show_selected_label = False
            return None
        
        viewer = self.parent_viewer
        i: int = self.SplineControl.num
        j: int = self.SplineControl.pos
        
        tomo = self.tomogram
        spl = tomo.splines[i]
        pos = spl.anchors[j]
        next_center = spl(pos) / tomo.scale
        change_viewer_focus(viewer, next_center, tomo.scale)
        
        self.layer_paint.show_selected_label = True
        
        j_offset = sum(spl.anchors.size for spl in tomo.splines[:i])
        self.layer_paint.selected_label = j_offset + j + 1
        return None
    
    @Image.wraps
    def Sample_subtomograms(self):
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
    
    @Image.wraps
    @set_design(text="R-projection")
    def show_r_proj(self, i: Bound[SplineControl.num], j: Bound[SplineControl.pos]):
        """Show radial projection of cylindrical image around the current MT fragment."""
        with ip.silent():
            polar = self._current_cylindrical_img().proj("r")
        
        self.Panels.image2D.image = polar.value
        self.Panels.image2D.text_overlay.update(visible=True, text=f"{i}-{j}", color="lime")
        # move to center
        ly, lx = polar.shape
        self.Panels.image2D.xlim = [lx*0.3, lx*0.7]
        self.Panels.current_index = 1
        return None
    
    @Image.wraps
    @set_design(text="R-projection (Global)")
    def show_global_r_proj(self):
        """Show radial projection of cylindrical image along current MT."""        
        i = self.SplineControl.num
        with ip.silent():
            polar = self.tomogram.straighten_cylindric(i).proj("r")
        self.Panels.image2D.image = polar.value
        self.Panels.image2D.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        # move to center
        ly, lx = polar.shape
        self.Panels.image2D.xlim = [lx*0.3, lx*0.7]
        self.Panels.current_index = 1
        return None
    
    @Image.wraps
    @set_design(text="2D-FT")
    def show_current_ft(self, i: Bound[SplineControl.num], j: Bound[SplineControl.pos]):
        """View Fourier space of local cylindrical coordinate system at current position."""        
        with ip.silent():
            polar = self._current_cylindrical_img(i, j)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw /= pw.max()
        
        if self.Panels.image2D.image is None:
            self.Panels.image2D.contrast_limits = np.percentile(pw, [0, 75])
        self.Panels.image2D.image = pw.value
        self.Panels.image2D.text_overlay.update(visible=True, text=f"{i}-{j}", color="lime")
        # move to center
        ly, lx = pw.shape
        self.Panels.image2D.xlim = [lx*0.3, lx*0.7]
        self.Panels.current_index = 1
        return None
    
    @Image.wraps
    @set_design(text="2D-FT (Global)")
    def show_global_ft(self, i: Bound[SplineControl.num]):
        """View Fourier space along current MT."""  
        with ip.silent():
            polar = self.tomogram.straighten_cylindric(i)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw /= pw.max()
            
        if self.Panels.image2D.image is None:
            self.Panels.image2D.contrast_limits = np.percentile(pw, [0, 75])
        self.Panels.image2D.image = pw.value
        self.Panels.image2D.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        # move to center
        ly, lx = pw.shape
        self.Panels.image2D.xlim = [lx*0.3, lx*0.7]
        self.Panels.current_index = 1
        return None
    
    @Splines.wraps
    def Show_splines(self):
        """Show 3D spline paths of microtubule center axes as a layer."""        
        paths = [r.partition(100) for r in self.tomogram.splines]
        
        self.parent_viewer.add_shapes(
            paths, shape_type="Splines", edge_color="lime", edge_width=1,
        )
        return None

    @Splines.wraps
    def Invert_spline(self, spline: Bound[SplineControl.num] = None):
        """Invert current displayed spline in place."""
        if spline is None:
            return
        spl = self.tomogram.splines[spline]
        self.tomogram.splines[spline] = spl.invert()
        self._update_splines_in_images()
        self.reset_choices()
        
        need_resample = self.SplineControl.canvas[0].image is not None
        self._init_widget_state()
        if need_resample:
            self.Sample_subtomograms()
        return None
    
    @Splines.wraps
    @set_options(orientation={"choices": ["MinusToPlus", "PlusToMinus"]})
    def Align_to_polarity(self, orientation: Ori = "MinusToPlus"):
        """
        Align all the splines in the direction parallel to microtubule polarity.

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
            self.Sample_subtomograms()
    
    @Splines.wraps
    @set_options(
        auto_call=True,
        spline={"choices": _get_splines},
        start={"max": 1.0, "step": 0.01},
        stop={"max": 1.0, "step": 0.01},
    )
    def Clip_spline(self, spline: int, start: float = 0.0, stop: float = 1.0):
        if spline is None:
            return
        spl = self.tomogram.splines[spline]
        self.tomogram.splines[spline] = spl.restore().clip(start, stop)
        self._update_splines_in_images()
        return None
        
    @Splines.wraps
    @set_options(max_interval={"label": "Max interval (nm)"},
                 cutoff={"options": {"max": 1.0, "step": 0.05, "value": 0.2}})
    @dispatch_worker
    def Fit_splines(self, 
                    max_interval: nm = 30,
                    cutoff: Optional[float] = None,
                    degree_precision: float = 0.5,
                    dense_mode: bool = False,
                    ):
        """
        Fit MT with spline curve, using manually selected points.

        Parameters
        ----------
        max_interval : nm, default is 30.0
            Maximum interval of sampling points in nm unit.
        degree_precision : float, default is 0.5
            Precision of MT xy-tilt degree in angular correlation.
        dense_mode : bool, default is False
            Check if microtubules are densely packed. Initial spline position must be "almost" fitted
            in dense mode.
        """        
        worker = create_worker(
            self.tomogram.fit,
            max_interval=max_interval,
            cutoff=cutoff,
            degree_precision=degree_precision,
            dense_mode=dense_mode,
            _progress={"total": 0, "desc": "Running"}
        )
        worker.returned.connect(self._update_splines_in_images)
        self._WorkerControl.info = "Spline Fitting"

        return worker
    
    @Splines.wraps
    @set_options(max_interval={"label": "Max interval (nm)"})
    def Fit_splines_manually(self, max_interval: nm = 50.0):
        """
        Open a spline fitter window and fit MT with spline manually.

        Parameters
        ----------
        max_interval : nm, default is 50.0
            Maximum interval between new anchors.
        """        
        self._SplineFitter._load_parent_state(max_interval=max_interval)
        self._SplineFitter.show()
        return None
    
    @Splines.wraps
    @set_options(interval={"label": "Interval between anchors (nm)"})
    def Add_anchors(self, interval: nm = 25.0):
        """
        Add anchors to splines.

        Parameters
        ----------
        interval : nm, default is 25.0
            Anchor interval.
        """        
        tomo = self.tomogram
        if tomo.n_splines == 0:
            raise ValueError("Cannot add anchors before adding splines.")
        for i in range(tomo.n_splines):
            tomo.make_anchors(i, interval=interval)
        self._update_splines_in_images()
        return None
    
    @Analysis.wraps
    @set_options(radius={"text": "Measure radii by radial profile."})
    @dispatch_worker
    def Set_radius(self, radius: Optional[nm] = None):
        """Measure MT radius for each spline path."""        
        worker = create_worker(self.tomogram.set_radius,
                               radius=radius,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        self._WorkerControl.info = "Measuring Radius"

        return worker
    
    @Splines.wraps
    @set_options(max_interval={"label": "Maximum interval (nm)"},
                 corr_allowed={"label": "Correlation allowed", "max": 1.0, "step": 0.1})
    @dispatch_worker
    def Refine_splines(self, max_interval: nm = 30, projection: bool = True, corr_allowed: float = 0.9):
        """
        Refine splines using the global MT structural parameters.
        
        Parameters
        ----------
        max_interval : nm, default is 30
            Maximum interval between anchors.
        projection : bool, default is True
            Check and Y-projection will be used to align subtomograms.
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        """
        tomo = self.tomogram
        
        worker = create_worker(tomo.refine,
                               max_interval=max_interval,
                               projection=projection,
                               corr_allowed=corr_allowed,
                               _progress={"total": 0, 
                                          "desc": "Running"})
        
        worker.finished.connect(self._update_splines_in_images)

        self._WorkerControl.info = "Refining splines ..."
        
        self._init_widget_state()
        return worker
    
    @Analysis.wraps
    @dispatch_worker
    def Local_FT_analysis(self, interval: nm = 32.0, ft_size: nm = 32.0):
        """
        Determine MT structural parameters by local Fourier transformation.

        Parameters
        ----------
        interval : nm, default is 32.0
            Interval of subtomogram analysis.
        ft_size : nm, default is 32.0
            Longitudinal length of local discrete Fourier transformation used for 
            structural analysis.
        """
        tomo = self.tomogram
        if tomo.splines[0].radius is None:
            self.Set_radius()
        self.Add_anchors(interval=interval)
        worker = create_worker(tomo.local_ft_params,
                               ft_size=ft_size,
                               _progress={"total": 0, "desc": "Running"}
                               )
        @worker.returned.connect
        def _on_return(df):
            self.Sample_subtomograms()
            self._update_local_properties_in_widget()
        self._last_ft_size = ft_size
        self._WorkerControl.info = "Local Fourier transform ..."
        return worker
        
    @Analysis.wraps
    @dispatch_worker
    def Global_FT_analysis(self):
        """Determine MT global structural parameters by Fourier transformation."""        
        tomo = self.tomogram
        worker = create_worker(tomo.global_ft_params,
                               _progress={"total": 0, "desc": "Running"})
        worker.returned.connect(lambda e: self._update_global_properties_in_widget())
        
        self._WorkerControl.info = f"Global Fourier transform ..."
        
        return worker
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Analysis.Mapping.wraps
    @set_options(
        splines={"widget_type": "Select", "choices": _get_splines},
        length={"text": "Use full length"}
    )
    @dispatch_worker
    def Map_monomers(
        self,
        splines: Iterable[int] = (),
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
        worker = create_worker(tomo.map_monomers,
                               i=splines,
                               length=length,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: List[Molecules]):
            for i, mol in enumerate(out):
                spl = tomo.splines[i]
                layer = add_molecules(self.parent_viewer, mol, f"Mono-{i}", source=spl)
                npf = roundint(spl.globalprops[H.nPF])
                update_features(layer, Mole.pf, np.arange(len(mol)) % npf)
                
        self._WorkerControl.info = "Monomer mapping ..."
        return worker

    @Analysis.Mapping.wraps
    @set_options(
        auto_call=True, 
        y_offset={"widget_type": "FloatSlider", "max": 5, "step": 0.1, "label": "y offset (nm)"},
        theta_offset={"widget_type": "FloatSlider", "max": 180, "label": "θ offset (deg)"},
        length={"text": "Use full length"},
    )
    def Map_monomers_manually(
        self, 
        i: Bound[SplineControl.num] = 0,
        y_offset: nm = 0, 
        theta_offset: float = 0,
        length: Optional[nm] = None,
    ):
        """
        Map points to monomer molecules with parameter sweeping.

        Parameters
        ----------
        i : int
            ID of microtubule.
        y_offset : nm, optional
            Offset in y-direction
        theta_offset : float, optional
            Offset of angle.
        length : nm, optional
            Length from the tip where monomers will be mapped.
        """
        theta_offset = np.deg2rad(theta_offset)
        tomo = self.tomogram
        tomo.global_ft_params(i)
        mol: Molecules = tomo.map_monomers(i, offsets=(y_offset, theta_offset), length=length)
        viewer = self.parent_viewer
        layer_name = f"Monomers-{i}"
        spl = tomo.splines[i]
        npf = roundint(spl.globalprops[H.nPF])
        labels = np.arange(len(mol)) % npf
        if layer_name not in viewer.layers:
            
            points_layer = self.parent_viewer.add_points(
                ndim=3, size=3, face_color="lime", edge_color="lime",
                out_of_slice_display=True, name=layer_name, 
                metadata={MOLECULES: mol, Mole.pf: labels, SOURCE: spl}
                )
            
            points_layer.shading = "spherical"
        
        else:
            points_layer: Points = viewer.layers[layer_name]
            points_layer.data = mol.pos
            points_layer.selected_data = set()
            points_layer.metadata[MOLECULES] = mol
            points_layer.metadata[SOURCE] = spl
            update_features(points_layer, Mole.pf, labels)
        
    @Analysis.Mapping.wraps
    @set_options(
        splines={"widget_type": "Select", "choices": _get_splines},
        interval={"text": "Set to dimer length"},
        length={"text": "Use full length"}
    )
    def Map_centers(
        self,
        splines: Iterable[int] = (),
        interval: Optional[nm] = None,
        length: Optional[nm] = None,
    ):
        """
        Map molecules along splines. Each molecule is rotated by skew angle.
        
        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        interval : nm, otional
            Interval between molecules.
        length : nm, optional
            Length from the tip where monomers will be mapped.
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        mols = tomo.map_centers(i=splines, interval=interval, length=length)
        for i, mol in enumerate(mols):
            add_molecules(self.parent_viewer, mol, f"Center-{i}", source=mol)

    @Analysis.Mapping.wraps
    @set_options(
        splines={"widget_type": "Select", "choices": _get_splines},
        interval={"text": "Set to dimer length"},
        angle_offset={"max": 360}
    )
    def Map_along_PF(
        self,
        splines: Iterable[int],
        interval: Optional[nm] = None,
        angle_offset: float = 0.0,
    ):
        """
        Map molecules along splines. Each molecule is rotated by skew angle.
        
        Parameters
        ----------
        splines : iterable of int
            Select splines to map monomers.
        interval : nm, otional
            Interval between molecules.
        angle_offset : float, default is 0.0
            Offset of PF angle in radian.
        """
        tomo = self.tomogram
        mols = tomo.map_pf_line(i=splines, interval=interval, angle_offset=angle_offset)
        for i, mol in enumerate(mols):
            add_molecules(self.parent_viewer, mol, f"PF line-{i}", source=mol)

    @Molecules.wraps
    @set_options(orientation={"choices": ["x", "y", "z"]})
    def Show_orientation(
        self,
        layer: MonomerLayer,
        orientation: str = "z"
    ):
        """
        Show molecule orientations with a vectors layer.

        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        orientation : {"x", "y", "z"}, default is "z"
            Which orientation will be shown. "z" is the spline-to-molecule direction,
            "y" is parallel to the spline and "x" is defined by right-handedness.
        """
        mol: Molecules = layer.metadata[MOLECULES]
        name = f"{layer.name} {orientation.upper()}-axis"
        
        vector_data = np.stack([mol.pos, getattr(mol, orientation)], axis=1)
        
        self.parent_viewer.add_vectors(
            vector_data, edge_width=0.3, edge_color="crimson", length=2.4,
            name=name,
            )
        return None
        
    @Molecules.wraps
    @set_options(spline_precision={"max": 2.0, "step": 0.01, "label": "spline precision (nm)"})
    def Calculate_intervals(
        self,
        layer: MonomerLayer,
        filter_length: int = 1,
        filter_width: int = 1,
        spline_precision: nm = 0.2,
    ):
        ndim = 3
        mole: Molecules = layer.metadata[MOLECULES]
        spl: MtSpline = layer.metadata[SOURCE]
        
        npf = roundint(spl.globalprops[H.nPF])
        try:
            pf_label = layer.features[Mole.pf]
            pos_list: list[np.ndarray] = []  # each shape: (y, ndim)
            for pf in range(pf_label.max() + 1):
                pos_list.append(mole.pos[pf_label == pf])
            pos = np.stack(pos_list, axis=1)  # shape: (y, pf, ndim)
            # pos = mole.pos.reshape(-1, npf, ndim)
        except Exception as e:
            raise TypeError(
                f"Reshaping failed. Molecules represented by layer {layer.name} must be "
                f"correctly labeled at {Mole.pf!r} feature. Original error is\n"
                f"{type(e).__name__}: {e}"
            ) from e
        
        pitch_vec = np.diff(pos, axis=0, append=0)
        u = spl.world_to_y(mole.pos, precision=spline_precision)
        spl_vec = spl(u, der=1)
        spl_vec_norm: np.ndarray = spl_vec / np.sqrt(np.sum(spl_vec**2, axis=1))[:, np.newaxis]
        spl_vec_norm = spl_vec_norm.reshape(-1, npf, ndim)
        y_dist: np.ndarray = np.sum(pitch_vec * spl_vec_norm, axis=2)  # inner product
        
        # apply filter
        if filter_length > 1 or filter_width > 1:
            l_ypad = filter_length // 2
            l_apad = filter_width // 2
            start = roundint(spl.globalprops[H.start])
            input = pad_mt_edges(y_dist, (l_ypad, l_apad), start=start)
            out = ndi.uniform_filter(input, (filter_length, filter_width), mode="constant")
            ly, lx = out.shape
            y_dist = out[l_ypad:ly-l_ypad, l_apad:lx-l_apad]
        
        properties = y_dist.ravel()
        _interval = "interval"
        _clim = [GVar.yPitchMin, GVar.yPitchMax]
        
        update_features(layer, _interval, np.abs(properties))
        
        # Set colormap
        layer.face_color = layer.edge_color = _interval
        layer.face_colormap = layer.edge_colormap = self.label_colormap
        layer.face_contrast_limits = layer.edge_contrast_limits = _clim
        layer.refresh()
        return None
    
    @Molecules.wraps
    @set_options(
        color_0={"widget_type": ColorEdit},
        color_1={"widget_type": ColorEdit},
    )
    def Show_isotypes(
        self,
        layer: MonomerLayer,
        color_0: Union[Iterable[float], str] = "orange",
        color_1: Union[Iterable[float], str] = "cyan",
    ):
        """
        Paint molecules according to the isotypes.

        Parameters
        ----------
        layer : MonomerLayer
            Molecules-bound layer.
        color_0 : Iterable[float] or str, optional
            Color of isotype 0.
        color_1 : Iterable[float] or str, optional
            Color of isotype 1.
        """
        if Mole.isotype not in layer.features.columns:
            raise ValueError("Isotype is not determined yet.")
        nmole = len(layer.data)
        spec = np.reshape(layer.features[Mole.isotype].values == 0, (-1, 1))
        colors = np.where(spec, [color_0]*nmole, [color_1]*nmole)
        layer.face_color = colors
        layer.edge_color = colors
        layer.refresh()
        return None
        
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Subtomogram averaging methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Analysis.wraps
    @do_not_record
    def Open_subtomogram_analyzer(self):
        """Open the subtomogram analyzer dock widget."""
        self._subtomogram_averaging.show()
    
    @magicclass(name="Subtomogram averaging")
    class _subtomogram_averaging(MagicTemplate):
        # Widget for subtomogram averaging
        
        def __post_init__(self):
            self._template = None
            self._viewer: Union[napari.Viewer, None] = None
            self.mask = "No mask"
            
        template_path = vfield(Path, options={"label": "Template", "filter": "*.mrc;*.tif"})
        mask = vfield(RadioButtons, options={"label": "Mask", "choices": ["No mask", "Use blurred template as a mask", "Supply a image"]}, record=False)
        
        @magicclass(layout="horizontal", widget_type="groupbox", name="Parameters")
        class params(MagicTemplate):
            dilate_radius = vfield(1.0, options={"tooltip": "Radius of dilation applied to binarized template (unit: nm).", "step": 0.5, "max": 20}, record=False)
            sigma = vfield(1.0, options={"tooltip": "Standard deviation of Gaussian blur applied to the edge of binary image (unit: nm).", "step": 0.5, "max": 20}, record=False)
            
        @magicclass(layout="horizontal", widget_type="frame")
        class mask_path(MagicTemplate):
            mask_path = vfield(Path, options={"filter": "*.mrc;*.tif"}, record=False)
        
        chunk_size = vfield(200, options={"min": 1, "max": 600, "tooltip": "How many subtomograms will be loaded at the same time."})
        
        @mask.connect
        def _on_switch(self):
            v = self.mask
            self.params.visible = (v == "Use blurred template as a mask")
            self.mask_path.visible = (v == "Supply a image")
        
        def _get_template(self, path: Union[str, None] = None) -> ip.ImgArray:
            if path is None:
                path = self.template_path
            else:
                self.template_path = path
            
            # check path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path!r} does not exist.")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Path {path!r} is not a file.")
            
            img = ip.imread(path)
            if img.ndim != 3:
                raise TypeError(f"Template image must be 3-D, got {img.ndim}-D.")
            parent = self.find_ancestor(MTPropsWidget)
            if parent.tomogram is not None:
                scale_ratio = img.scale.x / parent.tomogram.scale
                if scale_ratio < 0.99 or 1.01 < scale_ratio:
                    with ip.silent():
                        img = img.rescale(scale_ratio)
            self._template = img
            return img
        
        def _get_shape_in_nm(self) -> Tuple[int, ...]:
            if self._template is None:
                self._get_template()
            
            return tuple(s * self._template.scale.x for s in self._template.shape)
        
        def _get_mask_params(self, params=None) -> Union[str, Tuple[nm, nm], None]:
            v = self.mask
            if v == "No mask":
                params = None
            elif v == "Use blurred template as a mask":
                if self._template is None:
                    self._get_template()
                params = (self.params.dilate_radius, self.params.sigma)
            else:
                params = self.mask_path.mask_path
            return params
        
        _sentinel = object()
        
        def _get_mask(self, params: Union[str, Tuple[int, float], None] = _sentinel) -> Union[ip.ImgArray, None]:
            if params is self._sentinel:
                params = self._get_mask_params()
            else:
                if params is None:
                    self.mask = "No mask"
                elif isinstance(params, tuple):
                    self.mask = "Use blurred template as a mask"
                else:
                    self.mask_path.mask_path = params
            
            if params is None:
                return None
            elif isinstance(params, tuple):
                with ip.silent():
                    thr = self._template.threshold()
                    scale: nm = thr.scale.x
                    mask_image = thr.smooth_mask(
                        sigma=params[1]/scale, 
                        dilate_radius=roundint(params[0]/scale),
                    )
            else:
                mask_image = ip.imread(self.mask_path.mask_path)
            
            if mask_image.ndim != 3:
                raise TypeError(f"Mask image must be 3-D, got {mask_image.ndim}-D.")
            scale_ratio = mask_image.scale.x/self.find_ancestor(MTPropsWidget).tomogram.scale
            if scale_ratio < 0.99 or 1.01 < scale_ratio:
                with ip.silent():
                    mask_image = mask_image.rescale(scale_ratio)
            return mask_image
        
        def _show_reconstruction(self, image: ip.ImgArray, name: str) -> napari.Viewer:
            if self._viewer is not None:
                try:
                    # This line will raise RuntimeError if viewer window had been closed by user.
                    self._viewer.window.activate()
                except RuntimeError:
                    self._viewer = None
            if self._viewer is None:
                from .function_menu import Volume
                volume_menu = Volume()
                self._viewer = napari.Viewer(title=name, axis_labels=("z", "y", "x"), ndisplay=3)
                self._viewer.window.main_menu.addMenu(volume_menu.native)
                volume_menu.native.setParent(self._viewer.window.main_menu, volume_menu.native.windowFlags())
                self._viewer.window.resize(10, 10)
                self._viewer.window.activate()
            self._viewer.scale_bar.visible = True
            self._viewer.scale_bar.unit = "nm"
            with ip.silent():
                self._viewer.add_image(image.rescale_intensity(dtype=np.float32), scale=image.scale, name=name)
            
            return self._viewer
        
        @do_not_record
        def Show_template(self):
            """Load and show template image."""
            self._show_reconstruction(self._get_template(), name="Template image")
        
        @do_not_record
        def Show_mask(self):
            """Load and show mask image."""
            self._show_reconstruction(self._get_mask(), name="Mask image")
        
        @magicmenu
        class Subtomogram_analysis(MagicTemplate):
            def Average_all(self): ...
            def Average_subset(self): ...
            # def Calculate_FSC(self): ...
            def Seam_search(self): ...
        
        @magicmenu
        class Refinement(MagicTemplate):
            def Align_averaged(self): ...
            def Align_all(self): ...
            def Align_with_multiple_templates(self): ...
        
        @magicmenu
        class Template(MagicTemplate):
            def Reshape_template(self): ...
        
        @do_not_record
        @set_options(
            new_shape={"widget_type": TupleEdit, "options": {"min": 2, "max": 100}},
            save_as={"mode": "w", "filter": "*.mrc;*.tif"}
        )
        @Template.wraps
        def Reshape_template(
            self, 
            new_shape: Tuple[nm, nm, nm] = (20.0, 20.0, 20.0),
            save_as: Path = "",
            update_template_path: bool = True,
        ):
            template = self._get_template()
            if save_as == "":
                raise ValueError("Set save path.")
            scale = template.scale.x
            shape = tuple(roundint(s/scale) for s in new_shape)
            with ip.silent():
                reshaped = pad_template(template, shape)
            reshaped.imsave(save_as)
            if update_template_path:
                self.template_path = save_as
            return None
    
    def _get_binned_loader(
        self, 
        molecules: Molecules,
        shape: tuple[nm, ...], 
        chunk_size: int
    ) -> SubtomogramLoader:
        """Called when a method has to use subtomogram loader with binned image."""
        imgb: ip.ImgArray = self.layer_image.data
        tomo = self.tomogram
        binsize = roundint(imgb.scale.x/tomo.scale)
        tr = -(binsize - 1)/2*tomo.scale
        mole = molecules.translate([tr, tr, tr])
        shape = tuple(roundint(s/imgb.scale.x) for s in shape)
        return SubtomogramLoader(imgb, mole, shape, chunksize=chunk_size)
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "Subtomogram size (nm)"},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        save_at={"text": "Do not save the result.", "options": {"mode": "w", "filter": "*.mrc;*.tif"}},
    )
    @dispatch_worker
    def Average_all(
        self,
        layer: MonomerLayer,
        size: Optional[nm] = None,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
        interpolation: int = 1,
        use_binned_image: bool = False,
        save_at: Optional[Path] = None,
    ):
        """
        Subtomogram averaging using all the subvolumes.

        .. code-block:: python
        
            loader = ui.tomogram.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
            averaged = ui.tomogram
            
        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        size : nm, optional
            Size of subtomograms. Use template size by default.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        interpolation : int, default is 1
            Interpolation order used in ``ndi.map_coordinates``.
        use_binned_image : bool, default is False
            Check if you want to use binned image (reference image in the viewer) to boost image
            analysis.
        save_at : str, optional
            If given, save the averaged image at the specified location.
        """
        # TODO: save_at not working
        molecules: Molecules = layer.metadata[MOLECULES]
        tomo = self.tomogram
        nmole = len(molecules)
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        if use_binned_image:
            loader = self._get_binned_loader(molecules, shape, chunk_size)
        else:
            loader = tomo.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
        nbatch = 24
        worker = create_worker(
            loader.iter_average,
            order=interpolation,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_returned(img: ip.ImgArray):
            if self.tomogram.light_background:
                img = -img
            self._subtomogram_averaging._show_reconstruction(img, f"[AVG]{layer.name}")
            if save_at is not None:
                with ip.silent():
                    img.imsave(save_at)
        
        self._WorkerControl.info = f"Subtomogram averaging of {layer.name}"
        return worker
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "Subtomogram size (nm)"},
        method={"choices": ["steps", "first", "last", "random"]},
    )
    @dispatch_worker
    def Average_subset(
        self,
        layer: MonomerLayer,
        size: Optional[nm] = None,
        method="steps", 
        number: int = 64,
        use_binned_image: bool = False,
    ):
        """
        Subtomogram averaging using a subset of subvolumes.
        
        This function is equivalent to

        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        size : nm, optional
            Size of subtomograms. Use template size by default.
        method : str, optional
            How to choose subtomogram subset. 
            (1) steps: Each 'steps' subtomograms from the tip of spline. 
            (2) first: First subtomograms.
            (3) last: Last subtomograms.
            (4) random: choose randomly.
        number : int, default is 64
            Number of subtomograms to use.
        use_binned_image : bool, default is False
            Check if you want to use binned image (reference image in the viewer) to boost image
            analysis.
        """
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
        if use_binned_image:
            loader = self._get_binned_loader(mole, shape, chunk_size=1)
        else:
            loader = self.tomogram.get_subtomogram_loader(mole, shape, chunksize=1)
        
        worker = create_worker(loader.iter_average,
                               order = 1,
                               _progress={"total": number, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_returned(img: ip.ImgArray):
            if self.tomogram.light_background:
                img = -img
            self._subtomogram_averaging._show_reconstruction(img, f"Subtomogram average (n={number})")
        
        self._WorkerControl.info = f"Subtomogram Averaging (subset)"

        return worker
    
    def _check_binning_for_alignment(
        self,
        template: Union[ip.ImgArray, List[ip.ImgArray]],
        mask: Union[ip.ImgArray, None],
        use_binned_image: bool,
        molecules: Molecules,
        chunk_size: int
    ) -> Tuple[SubtomogramLoader, ip.ImgArray, Union[ip.ImgArray, None]]:
        shape = self._subtomogram_averaging._get_shape_in_nm()
        if use_binned_image:
            loader = self._get_binned_loader(molecules, shape, chunk_size)
            binsize = roundint(self.layer_image.scale[0]/self.tomogram.scale)
            with ip.silent():
                if isinstance(template, list):
                    template = [tmp.binning(binsize, check_edges=False) for tmp in template]
                else:
                    template = template.binning(binsize, check_edges=False)
                if mask is not None:
                    mask = mask.binning(binsize, check_edges=False)
        else:
            loader = self.tomogram.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
        return loader, template, mask
            
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
    )
    @dispatch_worker
    def Align_averaged(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        cutoff: float = 0.5,
        use_binned_image: bool = False,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        """
        Align the averaged image at current monomers to the template image.
        
        This function creates a new layer with transformed monomers, which should
        align well with template image.

        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        template_path : str
            Template image.
        mask_params : str, (int, float), optional
            Mask image
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied to averaged image.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        use_binned_image : bool, default is False
            Check if you want to use binned image (reference image in the viewer) to boost image
            analysis. Be careful! this may cause unexpected fitting result.
        """
        molecules: Molecules = layer.metadata[MOLECULES]
        template: ip.ImgArray = self._subtomogram_averaging._get_template(path=template_path)
        mask: ip.ImgArray = self._subtomogram_averaging._get_mask(params=mask_params)
        if mask is not None and template.shape != mask.shape:
            raise ValueError("Shape mismatch between template and mask.")
        nmole = len(molecules)
        spl: MtSpline = layer.metadata.get(SOURCE, None)
        
        pitch = spl.globalprops[H.yPitch]
        radius = spl.radius
        npf = spl.globalprops[H.nPF]
        
        loader, template, mask = self._check_binning_for_alignment(
            template, mask, use_binned_image, molecules, chunk_size
        )
        if use_binned_image:
            _scale = self.layer_image.data.scale.x
        else:
            _scale = self.tomogram.scale
            
        max_shifts = tuple(np.array([pitch, pitch/2, 2*np.pi*radius/npf/2])/_scale)
        nbatch = 24
        worker = create_worker(
            loader.iter_average,
            order=1,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
                
        @worker.returned.connect
        def _on_return(image_avg: ip.ImgArray):
            from ..components._align_utils import align_image_to_template, transform_molecules
            with ip.silent():
                img = image_avg.lowpass_filter(cutoff=cutoff)
                if use_binned_image and img.shape != template.shape:
                    sl = tuple(slice(0, s) for s in template.shape)
                    img = img[sl]
                    
                rot, shift = align_image_to_template(img, template, mask, max_shifts=max_shifts)
                cval = np.percentile(image_avg, 1)
                shifted_image = image_avg.affine(
                    translation=shift, cval=cval
                    ).rotate(np.rad2deg(rot), dims="zx", cval=cval)

            points = add_molecules(
                self.parent_viewer, 
                transform_molecules(molecules, shift * self.tomogram.scale, [0, -rot, 0]),
                _coerce_aligned_name(layer.name, self.parent_viewer),
                source=spl
            )
            points.features = layer.features
            self._subtomogram_averaging._show_reconstruction(shifted_image, "Aligned")
            layer.visible = False
                
        self._WorkerControl.info = f"Aligning averaged image (n={nmole}) to template"
        return worker
    
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"widget_type": TupleEdit, "options": {"max": 8.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        y_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        x_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
    )
    @dispatch_worker
    def Align_all(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        max_shifts: Tuple[nm, nm, nm] = (1., 1., 1.),
        z_rotation: Tuple[float, float] = (0., 0.),
        y_rotation: Tuple[float, float] = (0., 0.),
        x_rotation: Tuple[float, float] = (0., 0.),
        cutoff: float = 0.5,
        interpolation: int = 1,
        use_binned_image: bool = False,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
        template_path : ip.ImgArray, optional
            Template image.
        mask_params : ip.ImgArray, optional
            Mask image. Must in the same shape as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template in nm. ZYX order.
        z_rotation : tuple of float, optional
            Rotation in external degree around z-axis.
        y_rotation : tuple of float, optional
            Rotation in external degree around y-axis.
        x_rotation : tuple of float, optional
            Rotation in external degree around x-axis.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        interpolation : int, default is 1
            Interpolation order.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        """
        
        molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        source = layer.metadata.get(SOURCE, None)
        nmole = len(molecules)
        
        loader, template, mask = self._check_binning_for_alignment(
            template, mask, use_binned_image, molecules, chunk_size
        )
        nbatch = 24
        worker = create_worker(
            loader.iter_align,
            template=template, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            order=interpolation,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
                    
        @worker.returned.connect
        def _on_return(aligned_loader: SubtomogramLoader):
            points = add_molecules(
                self.parent_viewer, 
                aligned_loader.molecules,
                _coerce_aligned_name(layer.name, self.parent_viewer),
                source=source
            )
            points.features = layer.features
            layer.visible = False
                
        self._WorkerControl.info = f"Aligning subtomograms (n={nmole})"
        return worker
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        other_templates={"filter": "*.mrc;*.tif"},
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"widget_type": TupleEdit, "options": {"max": 8.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        y_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        x_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
    )
    @dispatch_worker
    def Align_with_multiple_templates(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        other_templates: List[Path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        max_shifts: Tuple[nm, nm, nm] = (1., 1., 1.),
        z_rotation: Tuple[float, float] = (0., 0.),
        y_rotation: Tuple[float, float] = (0., 0.),
        x_rotation: Tuple[float, float] = (0., 0.),
        cutoff: float = 0.5,
        interpolation: int = 1,
        use_binned_image: bool = False,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
        template_path : Path or str
            Template image path.
        other_templates : list of Path or str
            Path to other template images.
        mask_params : ip.ImgArray, optional
            Mask image. Must in the same shape as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template in nm. ZYX order.
        z_rotation : tuple of float, optional
            Rotation in external degree around z-axis.
        y_rotation : tuple of float, optional
            Rotation in external degree around y-axis.
        x_rotation : tuple of float, optional
            Rotation in external degree around x-axis.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        interpolation : int, default is 1
            Interpolation order.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        """
        
        molecules = layer.metadata[MOLECULES]
        templates = [self._subtomogram_averaging._get_template(path=template_path)]
        with ip.silent():
            for path in other_templates:
                img = ip.imread(path)
                scale_ratio = img.scale.x / self.tomogram.scale
                if scale_ratio < 0.99 or 1.01 < scale_ratio:
                    img = img.rescale(scale_ratio)
                templates.append(img)
        
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        source = layer.metadata.get(SOURCE, None)
        nmole = len(molecules)
        loader, templates, mask = self._check_binning_for_alignment(
            templates, mask, use_binned_image, molecules, chunk_size
        )
        nbatch = 24
        worker = create_worker(
            loader.iter_align_multi_templates,
            templates=templates, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            order=interpolation,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
                    
        @worker.returned.connect
        def _on_return(out: Tuple[np.ndarray, SubtomogramLoader]):
            labels, aligned_loader = out
            points = add_molecules(
                self.parent_viewer, 
                aligned_loader.molecules,
                _coerce_aligned_name(layer.name, self.parent_viewer),
                source=source
            )
            points.features = layer.features
            update_features(points, "opt-template", labels)
            layer.visible = False
                
        self._WorkerControl.info = f"Aligning subtomograms (n={nmole})"
        return worker

    # @_subtomogram_averaging.Subtomogram_analysis.wraps
    # @set_options(
    #     interpolation={"choices": [("linear", 1), ("cubic", 3)]},
    #     shape={"widget_type": TupleEdit, "text": "Use template shape"}
    # )
    # @dispatch_worker
    # def Calculate_FSC(
    #     self,
    #     layer: MonomerLayer,
    #     mask_params: Bound[_subtomogram_averaging._get_mask_params],
    #     shape: Optional[tuple[nm, nm, nm]] = None,
    #     seed: Optional[int] = 0,
    #     interpolation: int = 1,
    #     chunk_size: Bound[_subtomogram_averaging.chunk_size] = 64,
    # ):
    #     mole: Molecules = layer.metadata[MOLECULES]
    #     mask = self._subtomogram_averaging._get_mask(params=mask_params)
    #     if shape is None:
    #         shape = self._subtomogram_averaging._get_shape_in_nm()
    #     loader = self.tomogram.get_subtomogram_loader(mole, shape, chunksize=chunk_size)
    #     worker = create_worker(loader.fsc,
    #                            seed=seed,
    #                            mask=mask,
    #                            order=interpolation,
    #                            _progress={"total": 0, "desc": "Running"}
    #                            )
        
    #     @worker.returned.connect
    #     def _on_returned(out: tuple[np.ndarray, np.ndarray]):
    #         freq, fsc = out
    #         ind = (freq <= 0.5)
    #         plt = Figure(style="dark_background")
    #         plt.plot(freq[ind], fsc[ind], color="gold")
    #         plt.xlabel("Frequency")
    #         plt.ylabel("FSC")
    #         plt.title(f"FSC of {layer.name}")
    #         plt.show()
        
    #     self._WorkerControl.info = f"Calculating FSC ..."
    #     return worker
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        npf={"text": "Use global properties"},
        load_all={"label": "Load all the subtomograms in memory for better performance."}
    )
    @dispatch_worker
    def Seam_search(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 64,
        interpolation: int = 1,
        npf: Optional[int] = None,
        load_all: bool = False,
    ):
        """
        Search for the best seam position.
        
        Try all patterns of seam positions.

        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        template_path : ip.ImgArray, optional
            Template image.
        mask_params : ip.ImgArray, optional
            Mask image. Must in the same shape as the template.
        interpolation : int, default is 1
            Interpolation order.
        npf : int, optional
            Number of protofilaments. By default the global properties stored in the corresponding spline
            will be used.
        load_all : bool, default is False
            Load all the subtomograms into memory for better performance
            at the expense of memory usage.
        """
        molecules: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        source: MtSpline = layer.metadata.get(SOURCE, None)
        loader = self.tomogram.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
        if npf is None:
            try:
                npf = roundint(source.globalprops[H.nPF])
            except Exception:
                npf = np.max(layer.features[Mole.pf]) + 1
        
        total = 0 if load_all else 2*npf
            
        worker = create_worker(
            loader.iter_each_seam,
            npf=npf,
            template=template,
            mask=mask,
            load_all=load_all,
            order=interpolation,
            _progress={"total": total, "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_returned(result: Tuple[np.ndarray, ip.ImgArray]):
            corrs, img_ave, all_labels = result
            viewer: napari.Viewer = self._subtomogram_averaging._show_reconstruction(
                img_ave, layer.name
            )
            
            # calculate score and the best PF position
            corr1, corr2 = corrs[:npf], corrs[npf:]
            score = np.empty_like(corrs)
            score[:npf] = corr1 - corr2
            score[npf:] = corr2 - corr1
            imax = np.argmax(score)
                
            # plot all the correlation
            plt1 = Figure(style="dark_background")
            plt1.ax.axvline(imax, color="gray", alpha=0.5)
            plt1.ax.axhline(corrs[imax], color="gray", alpha=0.5)
            plt1.plot(corrs)
            plt1.xlabel("Seam position")
            plt1.ylabel("Correlation")
            plt1.xticks(np.arange(0, 2*npf+1, 4))
            plt1.title("Seam search result")
            
            # plot the score
            plt2 = Figure(style="dark_background")
            plt2.plot(score)
            plt2.xlabel("PF position")
            plt2.ylabel("ΔCorr")
            plt2.xticks(np.arange(0, 2*npf+1, 4))
            plt2.title("Score")

            wdt = Container(widgets=[plt1, plt2], labels=False)
            viewer.window.add_dock_widget(wdt, name=f"Seam search of {layer.name}", area="right")
            
            self.sub_viewer.layers[-1].metadata["Correlation"] = corrs
            self.sub_viewer.layers[-1].metadata["Score"] = score
            
            update_features(layer, Mole.isotype, all_labels[imax].astype(np.uint8))
            
        self._WorkerControl.info = "Seam search ... "

        return worker
        
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"pick_next.png")
    @do_not_record
    def pick_next(self):
        """Automatically pick MT center using previous two points."""        
        stride_nm = self.toolbar.Adjust.stride
        angle_pre = self.toolbar.Adjust.angle_precision
        angle_dev = self.toolbar.Adjust.angle_deviation
        imgb: ip.ImgArray = self.layer_image.data
        try:
            # orientation is point0 -> point1
            point0: np.ndarray = self.layer_work.data[-2]/imgb.scale.x  # unit: pixel
            point1: np.ndarray = self.layer_work.data[-1]/imgb.scale.x
        except IndexError:
            raise IndexError("Auto pick needs at least two points in the working layer.")
        
        tomo = self.tomogram
        binsize = roundint(self.layer_image.scale[0]/tomo.scale)  # scale of binned reference image
        
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        
        shape = (width_px,) + (roundint((width_px+length_px)/1.41),)*2
        
        with ip.silent():
            orientation = point1[1:] - point0[1:]
            img = crop_tomogram(imgb, point1, shape)
            center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
            angle_deg = angle_corr(img, ang_center=center, drot=angle_dev, nrots=ceilint(angle_dev/angle_pre))
            angle_rad = np.deg2rad(angle_deg)
            dr = np.array([0.0, stride_nm*np.cos(angle_rad), -stride_nm*np.sin(angle_rad)])
            if np.dot(orientation, dr[1:]) > np.dot(orientation, -dr[1:]):
                point2 = point1 + dr
            else:
                point2 = point1 - dr
            img_next = crop_tomogram(imgb, point2, shape)
            centering(img_next, point2, angle_deg)
            
        next_data = point2 * imgb.scale.x
        self.layer_work.add(next_data)
        msg = self._check_path()
        if msg:
            self.layer_work.data = self.layer_work.data[:-1]
            raise ValueError(msg)
        change_viewer_focus(self.parent_viewer, point2, imgb.scale.x)
        return None
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"auto_center.png")
    @do_not_record
    def auto_center(self):
        """Auto centering of selected points."""        
        imgb: ip.ImgArray = self.layer_image.data
        tomo = self.tomogram
        binsize = roundint(self.layer_image.scale[0]/tomo.scale)  # scale of binned reference image
        selected = self.layer_work.selected_data
        
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        
        shape = (width_px,) + (roundint((width_px+length_px)/1.41),)*2
        
        points = self.layer_work.data / imgb.scale.x
        last_i = -1
        with ip.silent():
            for i, point in enumerate(points):
                if i not in selected:
                    continue
                img_input = crop_tomogram(imgb, point, shape)
                angle_deg = angle_corr(img_input, ang_center=0, drot=89.5, nrots=19)
                centering(img_input, point, angle_deg, drot=5, nrots=7)
                last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], imgb.scale.x)
        return None
    
    @Image.wraps
    def Paint_MT(self):
        """
        Paint microtubule fragments by its pitch length.
        
        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """
        if self._last_ft_size is None:
            raise ValueError("Local structural parameters have not been determined yet.")
        lbl = np.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, List[float]] = {0: [0, 0, 0, 0]}
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        tomo = self.tomogram
        ft_size = self._last_ft_size
        
        lz, ly, lx = [int(r/bin_scale*1.4)*2 + 1 for r in [15, ft_size/2, 15]]
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        binsize = roundint(bin_scale/tomo.scale)
        with ip.silent():
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
                        
                    ry = max(ceilint(ry), 1)
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
            center = tomo.nm2pixel(crd)//binsize
            sl = []
            outsl = []
            # We should deal with the borders of image.
            for c, l, size in zip(center, [lz, ly, lx], lbl.shape):
                _sl, _pad = make_slice_and_pad(c - l//2, c + l//2 + 1, size)
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
        
        if tomo.light_background:
            thr = np.percentile(ref[lbl>0], 95)
            lbl[ref>thr] = 0
        else:
            thr = np.percentile(ref[lbl>0], 5)
            lbl[ref<thr] = 0
        
        # Labels layer properties
        _id = "ID"
        _structure = "structure"
        columns = [_id, H.riseAngle, H.yPitch, H.skewAngle, _structure]
        df = tomo.collect_localprops()[[H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]]
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
        start={"widget_type": ColorEdit},
        end={"widget_type": ColorEdit},
        limit={"widget_type": TupleEdit, "options": {"min": -20, "max": 20, "step": 0.01}, "label": "limit (nm)"},
        color_by={"choices": [H.yPitch, H.skewAngle, H.nPF, H.riseAngle]},
        auto_call=True
    )
    def Set_colormap(
        self,
        start=(0, 0, 1, 1), 
        end=(1, 0, 0, 1), 
        limit=(4.00, 4.24), 
        color_by: str = H.yPitch
    ):
        """
        Set the color-map for painting microtubules.
        
        Parameters
        ----------
        start : tuple, default is "blue"
            RGB color that corresponds to the most compacted microtubule.
        end : tuple, default is "red"
            RGB color that corresponds to the most expanded microtubule.
        limit : tuple, default is (4.00, 4.24)
            Color limit (nm).
        """        
        self.label_colormap = Colormap([start, end], name="PitchLength")
        self.label_colorlimit = limit
        self._update_colormap(prop=color_by)
        return None
    
    @Image.wraps
    def Show_colorbar(self):
        """Create a colorbar from the current colormap."""
        arr = self.label_colormap.colorbar[:5]  # shape == (5, 28, 4)
        plt = Figure()
        plt.imshow(arr)
        xmin, xmax = self.label_colorlimit
        plt.xticks([0, 27], [f"{xmin:.2f}", f"{xmax:.2f}"])
        plt.yticks([], [])
        plt.show()
        return None
    
    @nogui
    def get_molecules(self, name: str):
        """Retrieve Molecules object from layer list."""
        return self.parent_viewer.layers[name].metadata[MOLECULES]

    @nogui
    def get_loader(self, name: str, chunksize: int = 64):
        mole = self.get_molecules(name)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(mole, shape, chunksize)
        return loader
    
    @nogui
    def get_current_spline(self) -> MtSpline:
        tomo = self.tomogram
        i = self.SplineControl.num
        return tomo.splines[i]
    
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

        
    def _get_process_image_worker(
        self,
        img: ip.LazyImgArray,
        path: str,
        binsize: int,
        light_bg: bool, 
        cutoff: float,
        length: nm,
        width: nm,
        *, 
        new: bool = True
    ):
        """
        When an image is opened, we have to (1) prepare binned image for reference, (2) apply 
        low-pass filter if needed, (3) change existing layer scales if needed, (4) construct
        a new ``MtTomogram`` object if needed (5) make 2D projection. 
        """
        viewer = self.parent_viewer
        img = img.as_float()
        
        def _run(img: ip.LazyImgArray, binsize: int, cutoff: float):
            with ip.silent():
                if 0 < cutoff < 0.866:
                    img.tiled_lowpass_filter(cutoff, update=True)
                    img.release()
                imgb = img.binning(binsize, check_edges=False).compute()
            
            return imgb
        
        worker = create_worker(_run,
                               img=img,
                               binsize=binsize,
                               cutoff=cutoff,
                               _progress={"total": 0, "desc": "Reading Image"})

        self._WorkerControl.info = \
            f"Loading with {binsize}x{binsize} binned size: {tuple(s//binsize for s in img.shape)}"
        
        @worker.returned.connect
        def _on_return(imgb: ip.ImgArray):
            tr = (binsize - 1)/2*img.scale.x
            rendering = "minip" if light_bg else "mip"
            
            # update image layer
            if self.layer_image not in viewer.layers:
                self.layer_image = viewer.add_image(
                    imgb, 
                    scale=imgb.scale, 
                    name=imgb.name, 
                    translate=[tr, tr, tr],
                    contrast_limits=[np.min(imgb), np.max(imgb)],
                    rendering=rendering
                )
            else:
                self.layer_image.data = imgb
                self.layer_image.scale = imgb.scale
                self.layer_image.name = imgb.name
                self.layer_image.translate = [tr, tr, tr]
                self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]
                self.layer_image.rendering = rendering
            
            # update viewer dimensions
            viewer.scale_bar.unit = img.scale_unit
            viewer.dims.axis_labels = ("z", "y", "x")
            change_viewer_focus(viewer, np.asarray(imgb.shape)/2, imgb.scale.x)
            
            # update labels layer
            if self.layer_paint is not None:
                self.layer_paint.data = np.zeros(imgb.shape, dtype=np.uint8)
                self.layer_paint.scale = imgb.scale
                self.layer_paint.translate = [tr, tr, tr]
            
            # update overview
            with ip.silent():
                proj = imgb.proj("z")
            self.Panels.overview.image = proj
            self.Panels.overview.ylim = (0, proj.shape[0])
            
            if new:
                tomo = MtTomogram(subtomogram_length=length, 
                                  subtomogram_width=width, 
                                  light_background=light_bg)
                # metadata for GUI
                tomo._set_image(img)
                tomo.metadata["source"] = str(path)
                tomo.metadata["binsize"] = binsize
                tomo.metadata["cutoff"] = cutoff
                if self._last_ft_size is not None:
                    tomo.metadata["ft_size"] = self._last_ft_size
                self.tomogram = tomo
                tomo_list_widget = self._TomogramList
                tomo_list_widget._tomogram_list.append(tomo)
                tomo_list_widget.reset_choices()  # Next line of code needs updated choices
                tomo_list_widget.tomograms.value = len(tomo_list_widget._tomogram_list) - 1
                self.clear_all()
            
            return None
        
        return worker
    
    def _init_widget_state(self):
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
    
    def _check_path(self) -> str:
        tomo = self.tomogram
        imgshape_nm = np.array(tomo.image.shape) * tomo.image.scale.x
        if self.layer_work.data.shape[0] == 0:
            return ""
        else:
            point0 = self.layer_work.data[-1]
            box_size = (tomo.subtomo_width,) + ((tomo.subtomo_width+tomo.subtomo_length)/1.41,)*2
            
            if not np.all([r/4 <= p < s - r/4
                           for p, s, r in zip(point0, imgshape_nm, box_size)]):
                # outside image
                return "Outside boundary."        
        return ""
    
    def _current_cartesian_img(self, i=None, j=None):
        """
        Return local Cartesian image at the current position
        """        
        i: int = i or self.SplineControl.num
        j: int = j or self.SplineControl.pos
        tomo = self.tomogram
        spl = tomo._splines[i]
        
        l = tomo.nm2pixel(tomo.subtomo_length)
        w = tomo.nm2pixel(tomo.subtomo_width)
        
        coords = spl.local_cartesian((w, w), l, spl.anchors[j])
        img = tomo.image
        out = map_coordinates(img, coords, order=1)
        out = ip.asarray(out, axes="zyx")
        out.set_scale(img)
        out.scale_unit = img.scale_unit
        return out
    
    def _current_cylindrical_img(self, i=None, j=None):
        """
        Return cylindric-transformed image at the current position
        """        
        i: int = i or self.SplineControl.num
        j: int = j or self.SplineControl.pos
        tomo = self.tomogram
        if self._last_ft_size is None:
            raise ValueError("Local structural parameters have not been determined yet.")
        
        ylen = tomo.nm2pixel(self._last_ft_size)
        spl = tomo._splines[i]
        
        rmin = tomo.nm2pixel(spl.radius*GVar.inner)
        rmax = tomo.nm2pixel(spl.radius*GVar.outer)
        
        coords = spl.local_cylindrical((rmin, rmax), ylen, spl.anchors[j])
        img = tomo.image
        polar = map_coordinates(img, coords, order=1)
        polar = ip.asarray(polar, axes="rya") # radius, y, angle
        polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
        polar.scale_unit = img.scale_unit
        return polar
    
    def _init_layers(self):
        viewer: napari.Viewer = self.parent_viewer
        
        common_properties = dict(ndim=3, out_of_slice_display=True, size=8)
        if self.layer_prof in self.parent_viewer.layers:
            viewer.layers.remove(self.layer_prof)
    
        self.layer_prof: Points = viewer.add_points(
            **common_properties,
            name=SELECTION_LAYER_NAME,
            features={SPLINE_ID: []},
            opacity=0.4, 
            edge_color="black",
            face_color="black",
            )
        self.layer_prof.feature_defaults[SPLINE_ID] = 0
        self.layer_prof.editable = False
            
        if self.layer_work in viewer.layers:
            viewer.layers.remove(self.layer_work)
        
        self.layer_work = viewer.add_points(
            **common_properties,
            name=WORKING_LAYER_NAME,
            face_color="yellow"
            )

        self.layer_work.mode = "add"
        
        if self.layer_paint is not None:
            self.layer_paint.data = np.zeros_like(self.layer_paint.data)
            self.layer_paint.scale = self.layer_image.scale
        self.GlobalProperties._init_text()
        return None
    
    @SplineControl.num.connect
    def _highlight_spline(self):
        i: int = self.SplineControl.num
        if i is None:
            return
        
        for layer in self.Panels.overview.layers:
            if f"spline-{i}" in layer.name:
                layer.color = "red"
            else:
                layer.color = "lime"
        
        spec = self.layer_prof.features[SPLINE_ID] == i
        self.layer_prof.face_color = "black"
        self.layer_prof.face_color[spec] = [0.8, 0.0, 0.5, 1]
        self.layer_prof.refresh()
    
    @SplineControl.num.connect
    def _update_global_properties_in_widget(self):
        i: int = self.SplineControl.num
        if i is None:
            return
        spl = self.tomogram.splines[i]
        if spl.globalprops is not None:
            headers = [H.yPitch, H.skewAngle, H.nPF, H.start]
            pitch, skew, npf, start = spl.globalprops[headers]
            radius = spl.radius
            ori = spl.orientation
            self.GlobalProperties._set_text(pitch, skew, npf, start, radius, ori)
    
    @SplineControl.num.connect
    @SplineControl.pos.connect
    def _update_local_properties_in_widget(self):
        i: int = self.SplineControl.num
        tomo = self.tomogram
        if i is None or i >= len(tomo.splines):
            return
        j: int = self.SplineControl.pos
        spl = tomo.splines[i]
        if spl.localprops is not None:
            headers = [H.yPitch, H.skewAngle, H.nPF, H.start]
            pitch, skew, npf, start = spl.localprops[headers].iloc[j]
            self.LocalProperties._set_text(pitch, skew, npf, start)
        return None
    
    def _connect_worker(self, worker: Worker):
        self._WorkerControl._set_worker(worker)
        return None
        
    def _add_spline_to_images(self, spl: MtSpline, i: int):
        interval = 15
        length = spl.length()
        scale = self.layer_image.scale[0]
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        self.layer_prof.feature_defaults[SPLINE_ID] = i
        self.layer_prof.add(fit)
        self.Panels.overview.add_curve(
            fit[:, 2]/scale, fit[:, 1]/scale, color="lime", lw=3, name=f"spline-{i}",)
        return None
    
    def _update_splines_in_images(self):
        self.Panels.overview.layers.clear()
        self.layer_prof.data = []
        scale = self.layer_image.scale[0]
        for i, spl in enumerate(self.tomogram.splines):
            self._add_spline_to_images(spl, i)
            if spl._anchors is None:
                continue
            coords = spl()
            self.Panels.overview.add_scatter(
                coords[:, 2]/scale, 
                coords[:, 1]/scale,
                color="lime", 
                symbol="x",
                lw=1,
                size=10,
                name=f"spline-{i}-anc",
            )
        

def centering(imgb: ip.ImgArray, point: np.ndarray, angle: float, drot: int = 5, 
              nrots: int = 7):
    
    angle_deg2 = angle_corr(imgb, ang_center=angle, drot=drot, nrots=nrots)
    
    img_next_rot = imgb.rotate(-angle_deg2, cval=np.mean(imgb))
    proj = img_next_rot.proj("y")
    shift = mirror_pcc(proj)
    
    shiftz, shiftx = shift/2
    shift = np.array([shiftz, 0, shiftx])
    rad = -np.deg2rad(angle_deg2)
    cos = np.cos(rad)
    sin = np.sin(rad)
    shift = shift @ [[1.,   0.,  0.],
                     [0.,  cos, sin],
                     [0., -sin, cos]]
    point += shift


def _multi_affine(images, matrices, cval: float = 0, order=1):
    out = np.empty_like(images)
    for i, (img, matrix) in enumerate(zip(images, matrices)):
        out[i] = ndi.affine_transform(
            img, matrix, order=order, cval=cval, prefilter=order>1
        )
    return out
    
def _iter_run(
    tomo: MtTomogram, 
    splines: Iterable[int],
    interval: nm,
    ft_size,
    n_refine,
    max_shift,
    edge_sigma,
    local_props,
    global_props
) -> Iterator[str]:
    n_spl = len(splines)
    for i_spl in splines:
        if i_spl > 0:
            yield f"[{i_spl + 1}/{n_spl}] Spline fitting"
        tomo.fit(i=i_spl, edge_sigma=edge_sigma, max_shift=max_shift)
        tomo.set_radius(i=i_spl)
        
        for i in range(n_refine):
            yield f"[{i_spl + 1}/{n_spl}] Spline refinement (iteration {i + 1}/{n_refine})"
            tomo.refine(i=i_spl, max_interval=max(interval, 30))
            tomo.set_radius(i=i_spl)
            
        tomo.make_anchors(i=i_spl, interval=interval)
        if local_props:
            yield f"[{i_spl + 1}/{n_spl}] Local Fourier transformation"
            tomo.local_ft_params(i=i_spl, ft_size=ft_size)
        if global_props:
            yield f"[{i_spl + 1}/{n_spl}] Global Fourier transformation"
            tomo.global_ft_params(i=i_spl)
    yield "Finishing ..."
    return tomo


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
