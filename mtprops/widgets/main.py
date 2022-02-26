import re
from typing import Iterable, Union, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
import napari
from napari.utils import Colormap
from napari.qt import create_worker
from napari.layers import Points, Image, Labels, Vectors

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
    Separator,
    Table,
    RadioButtons,
    ColorEdit,
    ConsoleTextEdit,
    Figure,
    DraggableContainer
    )
from magicclass.ext.pyqtgraph import QtImageCanvas, QtMultiImageCanvas

from ..components import SubtomogramLoader, Molecules, MtSpline, MtTomogram
from ..components.tomogram import angle_corr, dask_affine
from ..utils import (
    Projections,
    crop_tomogram,
    make_slice_and_pad,
    map_coordinates,
    mirror_pcc, 
    roundint,
    ceilint,
    load_rot_subtomograms,
    no_verbose
    )
from ..const import EulerAxes, Unit, nm, H, Ori, GVar, Sep, Order
from ..const import WORKING_LAYER_NAME, SELECTION_LAYER_NAME, SOURCE, ALN_SUFFIX, MOLECULES
from ..types import MonomerLayer

from .localprops import LocalProperties
from .spline_fitter import SplineFitter
from .tomogram_list import TomogramList
from .worker import WorkerControl, dispatch_worker, Worker
from .widget_utils import add_molecules
from ..ext.etomo import PEET

ICON_DIR = Path(__file__).parent / "icons"

### The main widget ###
    
@magicclass(widget_type="scrollable", name="MTProps widget")
class MTPropsWidget(MagicTemplate):
    # Main GUI class.
    
    ### widgets ###
    
    _worker_control = field(WorkerControl)
    _spline_fitter = field(SplineFitter)
    
    @magicmenu
    class File(MagicTemplate):
        """File I/O."""  
        def Open_image(self): ...
        def Open_tomogram_list(self): ...
        def Load_json(self): ...
        sep0 = field(Separator)
        def Save_results_as_json(self): ...
        def Save_results_as_csv(self): ...
        def Save_monomer_coordinates(self): ...
        def Save_monomer_angles(self): ...
        sep1 = field(Separator)
        PEET = PEET

    @magicmenu
    class Image(MagicTemplate):
        """Image processing and visualization"""
        def Apply_lowpass_to_reference_image(self): ...
        sep0 = field(Separator)
        def show_current_ft(self): ...
        def show_global_ft(self): ...
        def show_r_proj(self): ...
        def show_global_r_proj(self): ...
        sep1 = field(Separator)
        def Sample_subtomograms(self): ...
        def Show_results_in_a_table_widget(self): ...
        def Show_straightened_image(self): ...
        def Paint_MT(self): ...
        def Set_colormap(self): ...
        focus = field(False, options={"text": "Focus"}, record=False)
    
    @magicmenu
    class Splines(MagicTemplate):
        """Spline fitting and operations."""
        def Show_splines(self): ...
        def Align_to_polarity(self): ...
        def Add_anchors(self): ...
        sep = field(Separator)
        def Fit_splines(self): ...
        def Fit_splines_manually(self): ...
        def Refine_splines(self): ...

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
        @magicmenu
        class Molecule_features(MagicTemplate):
            def Molecule_orientation(self): ...
            def Monomer_intervals(self): ...
        def Open_subtomogram_analyzer(self): ...
    
        
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
            stride = field(50.0, widget_type="FloatSlider", options={"min": 10, "max": 100, "tooltip": "Stride length (nm) of auto picker"}, record=False)
        sep1 = field(Separator)
        def clear_current(self): ...
        def clear_all(self): ...
    
    _Tomogram_list = TomogramList
    
    @magicclass(widget_type="groupbox")
    class Spline_control(MagicTemplate):
        """MT sub-regions"""
        def _get_splines(self, widget=None) -> list[int]:
            """Get list of spline objects for categorical widgets."""
            try:
                tomo = self.find_ancestor(MTPropsWidget).tomogram
            except Exception:
                return []
            if tomo is None:
                return []
            return [(str(spl), i) for i, spl in enumerate(tomo.splines)]
        
        num = field(int, widget_type="ComboBox", options={"choices": _get_splines}, name="Spline No.", record=False)
        pos = field(int, widget_type="Slider", options={"max": 0, "tooltip": "Position along a MT."}, name="Position", record=False)
        
        @num.connect
        def _num_changed(self):
            i = self.num.value
            tomo = self.find_ancestor(MTPropsWidget).tomogram
            spl = tomo.splines[i]
            if spl._anchors is not None:
                self.pos.max = spl.anchors.size - 1
            else:
                self.pos.value = 0
                self.pos.max = 0
        
    canvas = field(QtMultiImageCanvas, name="Figure", options={"nrows": 1, "ncols": 3, "tooltip": "Projections"})
    
    orientation_choice = vfield(Ori.none, name="Orientation: ", options={"tooltip": "MT polarity."})
    
    Local_Properties = LocalProperties
    
    @magicclass(widget_type="tabbed")
    class Panels(MagicTemplate):
        """Panels for output."""
        overview = field(QtImageCanvas, name="Overview", options={"tooltip": "Overview of splines"})
        image2D = field(QtImageCanvas, options={"tooltip": "2-D image viewer."})
        table = field(Table, name="Table", options={"tooltip": "Result table"}, record=False)
    
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
        self.Spline_control.pos.min_width = 70
        self.min_width = 400
        
        # Initialize multi-image canvas
        self.canvas.min_height = 200
        self.canvas.max_height = 230
        self.canvas[0].lock_contrast_limits = True
        self.canvas[0].title = "XY-Projection"
        self.canvas[1].lock_contrast_limits = True
        self.canvas[1].title = "XZ-Projection"
        self.canvas[2].lock_contrast_limits = True
        self.canvas[2].title = "Rot. average"
        
        self.Local_Properties.collapsed = False
        
        # Initialize multi-plot canvas
        self.Local_Properties.plot.min_height = 240
        self.Local_Properties.plot[0].ylabel = "pitch (nm)"
        self.Local_Properties.plot[0].legend.visible = False
        self.Local_Properties.plot[0].border = [1, 1, 1, 0.2]
        self.Local_Properties.plot[1].xlabel = "position (nm)"
        self.Local_Properties.plot[1].ylabel = "skew (deg)"
        self.Local_Properties.plot[1].legend.visible = False
        self.Local_Properties.plot[1].border = [1, 1, 1, 0.2]
        
        self.Local_Properties.params._init_text()
        self.Panels.min_height = 300

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

        self.tomogram.add_spline(coords)
        spl = self.tomogram.splines[-1]
        
        # draw path
        self._add_spline_to_images(spl)
        self.layer_work.data = []
        
        self.reset_choices()
        return None
    
    @magicclass(name="Run MTProps")
    class _runner(MagicTemplate):
        dense_mode = vfield(True, options={"label": "Use dense-mode", "tooltip": "Check if microtubules are densely packed. Initial spline position must be 'almost' fitted in dense mode."}, record=False)
        @magicclass(widget_type="groupbox", name="Parameters")
        class params1:
            """Parameters used in spline fitting."""
            dense_mode_sigma = vfield(2.0, options={"label": "dense-mode sigma", "tooltip": "Sharpness of dense-mode mask."}, record=False)
        n_refine = vfield(1, options={"label": "Refinement iteration", "max": 4, "tooltip": "Iteration number of spline refinement."}, record=False)
        local_props = vfield(True, options={"label": "Calculate local properties", "tooltip": "Check if calculate local properties."}, record=False)
        @magicclass(widget_type="groupbox", name="Parameters")
        class params2:
            """Parameters used in calculation of local properties."""
            interval = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Interval (nm)", "tooltip": "Interval of sampling points of microtubule fragments."}, record=False)
            ft_size = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Local DFT window size (nm)", "tooltip": "Longitudinal length of local discrete Fourier transformation used for structural analysis."}, record=False)
            paint = vfield(True, options={"tooltip": "Check if paint microtubules after local properties are calculated."}, record=False)
        global_props = vfield(True, options={"label": "Calculate global properties", "tooltip": "Check if calculate global properties."}, record=False)

        @dense_mode.connect
        def _toggle_dense_mode_sigma(self):
            self.params1.visible = self.dense_mode
        
        @local_props.connect
        def _toggle_localprops_params(self):
            self.params2.visible = self.local_props
        
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
        interval: Bound[_runner.params2.interval],
        ft_size: Bound[_runner.params2.ft_size],
        n_refine: Bound[_runner.n_refine],
        dense_mode: Bound[_runner.dense_mode],
        dense_mode_sigma: Bound[_runner.params1.dense_mode_sigma],
        local_props: Bound[_runner.local_props],
        global_props: Bound[_runner.global_props],
        paint: Bound[_runner.params2.paint]
    ):
        """Run MTProps"""
        self._runner.close()
        if self.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        
        total = 1 + n_refine + int(local_props) + int(global_props)
        
        worker = create_worker(_iter_run, 
                               tomo=self.tomogram,
                               interval=interval,
                               ft_size=ft_size,
                               n_refine=n_refine,
                               dense_mode=dense_mode,
                               dense_mode_sigma=dense_mode_sigma,
                               local_props=local_props,
                               global_props=global_props,
                               _progress={"total": total, 
                                          "desc": "Running MTProps"}
                               )
        
        @worker.yielded.connect
        def _on_yield(out):
            if isinstance(out, str):
                self._worker_control.info = out
                self._update_splines_in_images()
            
        @worker.returned.connect
        def _on_return(tomo: MtTomogram):
            self._update_splines_in_images()
            if local_props:
                self.Sample_subtomograms()
                if paint:
                    self.Paint_MT()
            tomo.metadata["ft_size"] = self._last_ft_size
            if global_props:
                self._globalprops_to_table(tomo.global_ft_params())
        self._last_ft_size = ft_size
        self._worker_control.info = "Spline fitting"
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
        self._init_widget_params()
        self._init_layers()
        self.Panels.overview.layers.clear()
        self._init_figures()
        self.tomogram.clear_cache()
        self.tomogram.splines.clear()
        self.reset_choices()
        return None
    
    @magicmenu
    class Others(MagicTemplate):
        """Other menus."""
        def Open_help(self): ...
        def Create_macro(self): ...
        def Global_variables(self): ...
        def Clear_cache(self): ...
        def MTProps_info(self): ...

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
        daskChunk={"widget_type": TupleEdit, "options": {"min": 16, "max": 2048, "step": 16}}
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
        daskChunk: Tuple[int, int, int] = GVar.daskChunk
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
        
        value = f"MTProps: {__version__}\n"\
                f"impy: {ip.__version__}\n"\
                f"magicgui: {magicgui.__version__}\n"\
                f"magicclass: {mcls.__version__}\n"\
                f"napari: {napari.__version__}\n"\
                f"dask: {dask.__version__}\n"
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
            self[7].visible = self.use_lowpass
        
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
        scale: Bound[_loader.scale],
        bin_size: Bound[_loader.bin_size],
        light_background: Bound[_loader.light_background],
        cutoff: Bound[_loader._get_cutoff_freq],
        subtomo_length: Bound[_loader.subtomo_length],
        subtomo_width: Bound[_loader.subtomo_width]
    ):
        """Start loading image."""
        try:
            scale = float(scale)
        except Exception as e:
            raise type(e)(f"Invalid input: {scale}")
        
        img = ip.lazy_imread(path, chunks=GVar.daskChunk)
        img.scale.x = img.scale.y = img.scale.z = scale
        
        worker = self._get_process_image_worker(
            img, 
            path,
            bin_size,
            light_background,
            cutoff, 
            subtomo_length,
            subtomo_width
            )
        
        self._loader.close()
        return worker
    
    @File.wraps
    @do_not_record
    def Open_image(self):
        """Open an image and add to viewer."""
        self._loader.show()
        return None
    
    @File.wraps
    @do_not_record
    def Open_tomogram_list(self):
        self._Tomogram_list.show()
        return None
        
    @File.wraps
    @set_options(path={"filter": "*.json;*.txt"})
    def Load_json(self, path: Path):
        """Choose a json file and load it."""        
        tomo = self.tomogram
        tomo.load_json(path)

        self._last_ft_size = tomo.metadata.get("ft_size", self._last_ft_size)
            
        self._update_splines_in_images()
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
    def Save_monomer_coordinates(self,
                                 save_path: Path,
                                 layer: MonomerLayer, 
                                 separator = Sep.Comma,
                                 unit = Unit.pixel,
                                 order = Order.xyz):
        """
        Save monomer coordinates.

        Parameters
        ----------
        save_path : Path
            Saving path.
        layer : Points
            Select the Vectors layer to save.
        separator : str, optional
            Select the separator.
        unit : Unit
            Unit of length.
        order : Order
            The order of output array.
        """        
        unit = Unit(unit)
        order = Order(order)
        separator = Sep(separator)
        if unit == Unit.pixel:
            arr = layer.data / self.tomogram.scale
        elif unit == Unit.nm:
            arr = layer.data
        elif unit == Unit.angstrom:
            arr = layer.data * 10
        if order == Order.xyz:
            arr = arr[:, ::-1]
        elif not order == Order.zyx:
            raise RuntimeError
        np.savetxt(save_path, arr, delimiter=str(separator))
        return None
    
    @File.wraps
    @set_options(save_path={"mode": "w", "filter": "*.txt;*.csv;*.dat"})
    def Save_monomer_angles(self, 
                            save_path: Path,
                            layer: MonomerLayer, 
                            rotation_axes = EulerAxes.ZXZ,
                            in_degree: bool = True,
                            separator = Sep.Comma):
        """
        Save monomer angles in Euler angles.

        Parameters
        ----------
        save_path : Path
            Saving path.
        layer : Points
            Select the Vectors layer to save.
        rotation_axes : str, default is "ZXZ"
            Select the rotation axes. {"X", "Y", "Z"} for intrinsic rotations, or
            {"x", "y", "z"} for extrinsic rotations.
        in_degree : bool, default is True
            Check to save angles in degrres.
        separator : str, optional
            Select the separator.
        """        
        separator = Sep(separator)
        mol: Molecules = layer.metadata[MOLECULES]
        arr = mol.euler_angle(rotation_axes, degrees=in_degree)
        np.savetxt(save_path, arr, delimiter=str(separator))
        return None
    
    @Image.wraps
    @dispatch_worker
    def Apply_lowpass_to_reference_image(self):
        """Apply low-pass filter to enhance contrast of the reference image."""
        cutoff = 0.2
        def func():
            with no_verbose():
                self.layer_image.data = self.layer_image.data.tiled_lowpass_filter(
                    cutoff, chunks=(32, 128, 128)
                    )
                return np.percentile(self.layer_image.data, [1, 97])
        worker = create_worker(func, _progress={"total": 0, "desc": "Running"})
        self._worker_control.info = "Low-pass filtering"

        @worker.returned.connect
        def _on_return(contrast_limits):
            self.layer_image.contrast_limits = contrast_limits
            with no_verbose():
                proj = self.layer_image.data.proj("z")
            self.Panels.overview.image = proj
            self.Panels.overview.contrast_limits = contrast_limits
        
        return worker
                    
    @Spline_control.num.connect
    @Spline_control.pos.connect
    @Image.focus.connect
    def _focus_on(self):
        """Change camera focus to the position of current MT fragment."""
        if self.layer_paint is None:
            return None
        if not self.Image.focus.value:
            self.layer_paint.show_selected_label = False
            return None
        
        viewer = self.parent_viewer
        i = self.Spline_control.num.value
        j = self.Spline_control.pos.value
        
        tomo = self.tomogram
        spl = tomo.splines[i]
        pos = spl.anchors[j]
        next_center = spl(pos) / tomo.scale
        viewer.dims.current_step = list(next_center.astype(np.int64))
        
        viewer.camera.center = next_center
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        viewer.camera.zoom = zoom
        
        self.layer_paint.show_selected_label = True
        
        j_offset = sum(spl.anchors.size for spl in tomo.splines[:i])
        self.layer_paint.selected_label = j_offset + j + 1
        return None
    
    @Image.wraps
    def Sample_subtomograms(self):
        """Sample subtomograms at the anchor points on splines"""
        self._spline_fitter.close()
        tomo = self.tomogram
        spl = tomo.splines[0]
        ori = spl.orientation
        
        # initialize GUI
        self._init_widget_params()
        self._init_layers()
        self.layer_work.mode = "pan_zoom"
        
        if spl.localprops is not None:
            n_anc = len(spl.localprops)
        elif spl._anchors is not None:
            n_anc = len(spl._anchors)
        else:
            return
        
        self.Spline_control.pos.max = n_anc - 1
        
        self.orientation_choice = ori
        self._update_spline()
        return None
    
    @Image.wraps
    def Show_results_in_a_table_widget(self):
        """Show result table."""
        self.Panels.table.value = self.tomogram.collect_localprops()
        self.Panels.current_index = 2
        return None
    
    @Image.wraps
    @dispatch_worker
    def Show_straightened_image(self, i: Bound[Spline_control.num]):
        """Send straightened image of the current MT to the viewer."""        
        tomo = self.tomogram
        
        worker = create_worker(tomo.straighten, 
                               i=i, 
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.ImgArray):
            self.parent_viewer.add_image(out, scale=out.scale)
        
        self._worker_control.info = f"Straightening spline No. {i}"
        
        return worker
    
    @Image.wraps
    @set_design(text="R-projection")
    def show_r_proj(self, i: Bound[Spline_control.num], j: Bound[Spline_control.pos]):
        """Show radial projection of cylindrical image around the current MT fragment."""
        with no_verbose():
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
        i = self.Spline_control.num.value
        with no_verbose():
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
    def show_current_ft(self, i: Bound[Spline_control.num], j: Bound[Spline_control.pos]):
        """View Fourier space of local cylindrical coordinate system at current position."""        
        with no_verbose():
            polar = self._current_cylindrical_img()
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
    def show_global_ft(self, i: Bound[Spline_control.num]):
        """View Fourier space along current MT."""  
        with no_verbose():
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
        
        self.parent_viewer.add_shapes(paths, shape_type="path", edge_color="lime", edge_width=1,
                                      translate=self.layer_image.translate)
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
        need_resample = self.canvas[0].image is not None
        self.tomogram.align_to_polarity(orientation=orientation)
        self._update_splines_in_images()
        self._init_widget_params()
        self._init_figures()
        if need_resample:
            self.Sample_subtomograms()
        
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
        worker = create_worker(self.tomogram.fit,
                               max_interval=max_interval,
                               cutoff=cutoff,
                               degree_precision=degree_precision,
                               dense_mode=dense_mode,
                               _progress={"total": 0, "desc": "Running"}
                               )
        worker.returned.connect(self._init_layers)
        worker.returned.connect(self._update_splines_in_images)
        self._worker_control.info = "Spline Fitting"

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
        self._spline_fitter._load_parent_state(max_interval=max_interval)
        self._spline_fitter.show()
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
        
        self._worker_control.info = "Measuring Radius"

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

        self._worker_control.info = "Refining splines ..."
        
        self._init_widget_params()
        self._init_figures()
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
        self._last_ft_size = ft_size
        self._worker_control.info = "Local Fourier transform ..."
        return worker
        
    @Analysis.wraps
    @dispatch_worker
    def Global_FT_analysis(self):
        """Determine MT global structural parameters by Fourier transformation."""        
        tomo = self.tomogram
        worker = create_worker(tomo.global_ft_params,
                               _progress={"total": 0, "desc": "Running"})
        worker.returned.connect(self._globalprops_to_table)
        
        self._worker_control.info = f"Global Fourier transform ..."
        
        return worker
    
    def _globalprops_to_table(self, out: List[pd.Series]):
        df = pd.DataFrame({f"MT-{k}": v for k, v in enumerate(out)})
        self.Panels.table.value = df
        self.Panels.current_index = 2
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Analysis.Mapping.wraps
    @set_options(splines={"widget_type": "Select", "choices": _get_splines},
                 length={"text": "Use full length"})
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
        worker = create_worker(tomo.map_monomers,
                               i=splines,
                               length=length,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: List[Molecules]):
            for i, mol in enumerate(out):
                spl = tomo.splines[i]
                add_molecules(self.parent_viewer, mol, f"Monomers-{i}", source=spl)
                
        self._worker_control.info = "Monomer mapping ..."
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
        i: Bound[Spline_control.num],
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
        mol = tomo.map_monomers(i, offsets=(y_offset, theta_offset), length=length)
        
        viewer = self.parent_viewer
        layer_name = f"Monomers-{i}"
        if layer_name not in viewer.layers:
            points_layer = self.parent_viewer.add_points(
                ndim=3, size=3, face_color="lime", edge_color="lime",
                out_of_slice_display=True, name=layer_name, metadata={MOLECULES: mol}
                )
            
            points_layer.shading = "spherical"
            
            self.parent_viewer.add_vectors(
                ndim=3, edge_width=0.3, edge_color="crimson", length=2.4,
                name=layer_name + " Z-axis",
                )
        
        points_layer: Points = viewer.layers[layer_name]
        points_layer.data = mol.pos
        points_layer.selected_data = set()
        points_layer.metadata[SOURCE] = mol
        vector_layer: Vectors = viewer.layers[layer_name + " Z-axis"]
        vector_layer.data = np.stack([mol.pos, mol.z], axis=1)

    @Analysis.Mapping.wraps
    @set_options(
        splines={"widget_type": "Select", "choices": _get_splines},
        interval={"text": "Set to dimer length"},
        length={"text": "Use full length"}
    )
    def Map_centers(
        self,
        splines: Iterable[int],
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

    @Analysis.Molecule_features.wraps
    @set_options(orientation={"choices": ["x", "y", "z"]})
    def Molecule_orientation(
        self,
        layer: MonomerLayer,
        orientation: str = "z"
    ):
        mol: Molecules = layer.metadata[MOLECULES]
        name = f"{layer.name} {orientation.upper()}-axis"
        
        vector_data = np.stack([mol.pos, getattr(mol, orientation)], axis=1)
        
        self.parent_viewer.add_vectors(
            vector_data, edge_width=0.3, edge_color="crimson", length=2.4,
            name=name,
            )
        return None
        
    @Analysis.Molecule_features.wraps
    @set_options(spline_precision={"max": 2.0, "step": 0.01, "label": "spline precision (nm)"})
    def Monomer_intervals(
        self,
        layer: MonomerLayer,
        spline_precision: nm = 0.2,
    ):
        ndim = 3
        mole: Molecules = layer.metadata[MOLECULES]
        spl: MtSpline = layer.metadata[SOURCE]
        
        npf = roundint(spl.globalprops[H.nPF])
        try:
            pos = mole.pos.reshape(-1, npf, ndim)
        except ValueError as e:
            msg = (
                f"Reshaping failed. Molecules represented by layer {layer.name} may not "
                f"be a tubular shaped."
            )
            e.args = (msg,)
            raise e
        
        pitch_vec = np.diff(pos, axis=0, append=0)
        u = spl.world_to_y(mole.pos, precision=spline_precision)
        spl_vec = spl(u, der=1)
        spl_vec_norm = spl_vec / np.sqrt(np.sum(spl_vec**2, axis=1))[:, np.newaxis]
        spl_vec_norm = spl_vec_norm.reshape(-1, npf, ndim)
        y_dist = np.sum(pitch_vec * spl_vec_norm, axis=2)  # inner product
        
        properties = y_dist.ravel()
        _interval = "interval"
        _clim = [GVar.yPitchMin, GVar.yPitchMax]
        
        # Update features
        features = layer.features
        features[_interval] = properties
        layer.features = features
        
        # Set colormap
        layer.face_color = layer.edge_color = _interval
        layer.face_colormap = layer.edge_colormap = self.label_colormap
        layer.face_contrast_limits = layer.edge_contrast_limits = _clim
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
            dilate_radius = vfield(4, options={"tooltip": "Radius of dilation applied to template (unit: pixel).", "max": 100}, record=False)
            sigma = vfield(4.0, options={"tooltip": "Standard deviation of Gaussian blur applied to the edge of binary image (unit: pixel).", "max": 100}, record=False)
            
        @magicclass(layout="horizontal", widget_type="frame")
        class mask_path(MagicTemplate):
            mask_path = vfield(Path, options={"filter": "*.mrc;*.tif"}, record=False)
        
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
            img = ip.imread(path)
            if img.ndim != 3:
                raise TypeError(f"Template image must be 3-D, got {img.ndim}-D.")
            scale_ratio = img.scale.x/self.find_ancestor(MTPropsWidget).tomogram.scale
            if scale_ratio < 0.99 or 1.01 < scale_ratio:
                with no_verbose():
                    img = img.rescale(scale_ratio)
            self._template = img
            return img
        
        def _get_shape_in_nm(self) -> Tuple[int, ...]:
            if self._template is None:
                self._get_template()
            
            return tuple(s * self._template.scale.x for s in self._template.shape)
        
        def _get_mask_params(self, params=None) -> Union[str, Tuple[int, float], None]:
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
                with no_verbose():
                    thr = self._template.threshold()
                    mask_image = thr.smooth_mask(
                        sigma=params[1], 
                        dilate_radius=params[0]
                    )
            else:
                mask_image = ip.imread(self.mask_path.mask_path)
            
            if mask_image.ndim != 3:
                raise TypeError(f"Mask image must be 3-D, got {mask_image.ndim}-D.")
            scale_ratio = mask_image.scale.x/self.find_ancestor(MTPropsWidget).tomogram.scale
            if scale_ratio < 0.99 or 1.01 < scale_ratio:
                with no_verbose():
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
                self._viewer = napari.Viewer(title=name, axis_labels=("z", "y", "x"), ndisplay=3)
            self._viewer.scale_bar.visible = True
            self._viewer.scale_bar.unit = "nm"
            with no_verbose():
                self._viewer.add_image(image.rescale_intensity(), scale=image.scale, name=name)
            
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
            def Calculate_FSC(self): ...
            def Seam_search(self): ...
        
        @magicmenu
        class Refinement(MagicTemplate):
            def Align_averaged(self): ...
            def Align_all(self): ...
    
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
        chunk_size={"min": 1, "max": 512},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        save_at={"text": "Do not save the result.", "options": {"mode": "w", "filter": "*.mrc;*.tif"}},
    )
    @dispatch_worker
    def Average_all(
        self,
        layer: MonomerLayer,
        size: Optional[nm] = None,
        chunk_size: int = 64,
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
        
        worker = create_worker(loader.iter_average,
                               order=interpolation,
                               _progress={"total": nmole, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_returned(img: ip.ImgArray):
            if self.tomogram.light_background:
                img = -img
            self._subtomogram_averaging._show_reconstruction(img, f"Subtomogram average (n={nmole})")
            if save_at is not None:
                with no_verbose():
                    img.imsave(save_at)
        
        self._worker_control.info = f"Subtomogram averaging of {layer.name} ..."
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
        
        self._worker_control.info = f"Subtomogram Averaging (subset) ..."

        return worker
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        chunk_size={"min": 1, "max": 512},
    )
    @dispatch_worker
    def Align_averaged(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        cutoff: float = 0.5,
        use_binned_image: bool = False,
        chunk_size: int = 64,
    ):
        """
        Align the averaged image at current monomers to the template image.

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
        """        
        molecules: Molecules = layer.metadata[MOLECULES]
        template: ip.ImgArray = self._subtomogram_averaging._get_template(path=template_path)
        mask: ip.ImgArray = self._subtomogram_averaging._get_mask(params=mask_params)
        if template.shape != mask.shape:
            raise ValueError("Shape mismatch between template and mask.")
        nmole = len(molecules)
        spl: MtSpline = layer.metadata.get(SOURCE, None)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        
        pitch = spl.globalprops[H.yPitch]
        radius = spl.radius
        npf = spl.globalprops[H.nPF]
        
        if use_binned_image:
            loader = self._get_binned_loader(molecules, shape, chunk_size)
            _scale = self.layer_image.data.scale.x
            binsize = roundint(self.layer_image.scale[0]/self.tomogram.scale)
            with no_verbose():
                template = template.binning(binsize, check_edges=False)
                mask = mask.binning(binsize, check_edges=False)
        else:
            loader = self.tomogram.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
            _scale = self.tomogram.scale
            
        max_shifts = tuple(np.array([pitch, pitch/2, 2*np.pi*radius/npf/2])/_scale)
        # max_shifts = ceilint(pitch/_scale)
        worker = create_worker(loader.iter_average,
                               order = 1,
                               _progress={"total": nmole, "desc": "Running"}
                               )
                
        @worker.returned.connect
        def _on_return(image_avg: ip.ImgArray):
            from ..components._align_utils import align_image_to_template
            from scipy.spatial.transform import Rotation
            with no_verbose():
                img = image_avg.lowpass_filter(cutoff=cutoff)
                if use_binned_image and img.shape != template.shape:
                    sl = tuple(slice(0, s) for s in template.shape)
                    img = img[sl]
                    
                rot, shift = align_image_to_template(img, template, mask, max_shifts=max_shifts)
                shifted_image = image_avg.affine(translation=shift, cval=np.min(image_avg))
            dx = shift[-1]
            dtheta = dx/radius
            skew_rotator = Rotation.from_rotvec(molecules.y * dtheta)
            shift = molecules.rotator.apply(-shift * self.tomogram.scale)  # Verified
            internal_rotator = Rotation.from_rotvec([0, rot, 0])
            mole = molecules.rotate_by(skew_rotator).translate(internal_rotator.apply(shift))
            
            add_molecules(self.parent_viewer, 
                           mole,
                           _coerce_aligned_name(layer.name),
                           source=spl
                           )
            self._subtomogram_averaging._show_reconstruction(shifted_image, "Aligned average image")
                
        self._worker_control.info = f"Aligning averaged image (n={nmole}) to template"
        return worker
    
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"widget_type": TupleEdit, "options": {"max": 8.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        y_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        x_rotation={"widget_type": TupleEdit, "options": {"max": 5.0, "step": 0.1}},
        chunk_size={"min": 1, "max": 512},
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
        chunk_size: int = 64,
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
        shape = self._subtomogram_averaging._get_shape_in_nm()
        nmole = len(molecules)
        
        if use_binned_image:
            loader = self._get_binned_loader(molecules, shape, chunk_size)
        else:
            loader = self.tomogram.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
            
        worker = create_worker(loader.iter_align,
                               template=template, 
                               mask=mask,
                               max_shifts=max_shifts,
                               rotations=(z_rotation, y_rotation, x_rotation),
                               cutoff=cutoff,
                               order=interpolation,
                               _progress={"total": nmole, "desc": "Running"}
                               )
                    
        @worker.returned.connect
        def _on_return(aligned_loader: SubtomogramLoader):
            add_molecules(self.parent_viewer, 
                           aligned_loader.molecules,
                           _coerce_aligned_name(layer.name),
                           source=source
                           )            
                
        self._worker_control.info = f"Aligning subtomograms (n={nmole})"
        return worker
    

    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        shape={"widget_type": TupleEdit, "text": "Use template shape"}
    )
    @dispatch_worker
    def Calculate_FSC(
        self,
        layer: MonomerLayer,
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        shape: Optional[tuple[nm, nm, nm]] = None,
        seed: Optional[int] = 0,
        interpolation: int = 1,
    ):
        mole: Molecules = layer.metadata[MOLECULES]
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        if shape is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(mole, shape)
        worker = create_worker(loader.fsc,
                               seed=seed,
                               mask=mask,
                               order=interpolation,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_returned(out: tuple[np.ndarray, np.ndarray]):
            freq, fsc = out
            plt = Figure(style="dark_background")
            plt.plot(freq, fsc, color="darkblue")
            plt.xlabel("Frequency")
            plt.ylabel("FSC")
            plt.title(f"Fourier Shell Correlation of {layer.name}")
            plt.show()
        
        self._worker_control.info = f"Calculating FSC ..."
        return worker
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        load_all={"label": "Load all the subtomograms in memory for better performance."}
    )
    @dispatch_worker
    def Seam_search(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        interpolation: int = 1,
        load_all: bool = False,
    ):
        molecules: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        source: MtSpline = layer.metadata[SOURCE]
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(molecules, shape)
        npf = roundint(source.globalprops[H.nPF])
        
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
        def _on_returned(result: Tuple[np.ndarray, ip.ImgArray, list[Molecules]]):
            corrs, img_ave, moles = result
            iopt = np.argmax(corrs)
            viewer: napari.Viewer = self._subtomogram_averaging._show_reconstruction(
                img_ave, "All reconstructions"
            )
            # plot all the correlation
            plt1 = Figure(style="dark_background")
            plt1.plot(corrs)
            plt1.xlabel("Seam position")
            plt1.ylabel("Correlation")
            plt1.xticks(np.arange(0, 2*npf+1, 4))
            plt1.title("Seam search result")
            
            # plot the score
            corr1, corr2 = corrs[:npf], corrs[npf:]
            if corr1.max() < corr2.max():
                score = corr2 - corr1
            else:
                score = corr1 - corr2
            plt2 = Figure(style="dark_background")
            plt2.plot(score)
            plt2.xlabel("PF position")
            plt2.ylabel("ΔCorr")
            plt1.xticks(np.arange(0, npf+1, 2))
            plt2.title("Score")
            wdt = DraggableContainer(widgets=[plt1, plt2], labels=False)
            viewer.window.add_dock_widget(wdt, name="Seam search", area="right")
            add_molecules(self.parent_viewer, moles[iopt], layer.name + "-OPT", source=source)
            
        self._worker_control.info = "Seam search ... "

        return worker
        
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"pick_next.png")
    @do_not_record
    def pick_next(self):
        """Automatically pick MT center using previous two points."""        
        stride_nm = self.toolbar.Adjust.stride.value
        imgb = self.layer_image.data
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
        
        with no_verbose():
            orientation = point1[1:] - point0[1:]
            img = crop_tomogram(imgb, point1, shape)
            center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
            angle_deg = angle_corr(img, ang_center=center, drot=25, nrots=25)
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
        change_viewer_focus(self.parent_viewer, point2, next_data)
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
        with no_verbose():
            for i, point in enumerate(points):
                if i not in selected:
                    continue
                img_input = crop_tomogram(imgb, point, shape)
                angle_deg = angle_corr(img_input, ang_center=0, drot=89.5, nrots=19)
                centering(img_input, point, angle_deg, drot=5, nrots=7)
                last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], self.layer_work.data[last_i])
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
        with no_verbose():
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
                matrices.append(spl.affine_matrix(center=center))
            
            cylinders = np.concatenate(cylinders, axis=0)
            matrices = np.concatenate(matrices, axis=0)
            out = dask_affine(cylinders, matrices) > 0.3
            
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
        _type = "type"
        columns = [_id, H.riseAngle, H.yPitch, H.skewAngle, _type]
        df = tomo.collect_localprops()[[H.riseAngle, H.yPitch, H.skewAngle, H.nPF, H.start]]
        df_reset = df.reset_index()
        df_reset[_id] = df_reset.apply(
            lambda x: "{}-{}".format(int(x["SplineID"]), int(x["PosID"])), 
            axis=1
            )
        df_reset[_type] = df_reset.apply(
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


    def _plot_properties(self):
        i = self.Spline_control.num.value
        props = self.tomogram.splines[i].localprops
        if props is None:
            return None
        x = np.asarray(props[H.splDistance])
        pitch_color = "lime"
        skew_color = "gold"
        
        self.Local_Properties.plot[0].layers.clear()
        self.Local_Properties.plot[0].add_curve(x, props[H.yPitch], color=pitch_color)
        
        self.Local_Properties.plot[1].layers.clear()
        self.Local_Properties.plot[1].add_curve(x, props[H.skewAngle], color=skew_color)

        self.Local_Properties.plot.xlim = (x[0] - 2, x[-1] + 2)
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
            with no_verbose():
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

        self._worker_control.info = \
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
            viewer.dims.set_current_step(0, imgb.shape[0]//2)
            
            # update labels layer
            if self.layer_paint is not None:
                self.layer_paint.data = np.zeros(imgb.shape, dtype=np.uint8)
                self.layer_paint.scale = imgb.scale
                self.layer_paint.translate = [tr, tr, tr]
            
            # update overview
            with no_verbose():
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
                tomo_list_widget = self._Tomogram_list
                tomo_list_widget._tomogram_list.append(tomo)
                tomo_list_widget.reset_choices()  # Next line of code needs updated choices
                tomo_list_widget.tomograms.value = len(tomo_list_widget._tomogram_list) - 1
                self.clear_all()
            
            return None
        
        return worker
    
    def _init_widget_params(self):
        self.Spline_control.pos.value = 0
        self.Spline_control.pos.max = 0
        self.projections: List[Projections] = []
        self.Local_Properties.params._init_text()
        return None
    
    def _init_figures(self):
        for i in range(3):
            del self.canvas[i].image
            self.canvas[i].layers.clear()
            self.canvas[i].text_overlay.text = ""
        for i in range(2):
            self.Local_Properties.plot[i].layers.clear()
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
        i = i or self.Spline_control.num.value
        j = j or self.Spline_control.pos.value
        tomo = self.tomogram
        spl = tomo._splines[i]
        
        l = tomo.nm2pixel(tomo.subtomo_length)
        w = tomo.nm2pixel(tomo.subtomo_width)
        
        coords = spl.local_cartesian((w, w), l, spl.anchors[j])
        coords = np.moveaxis(coords, -1, 0)
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
        i = i or self.Spline_control.num.value
        j = j or self.Spline_control.pos.value
        tomo = self.tomogram
        if self._last_ft_size is None:
            raise ValueError("Local structural parameters have not been determined yet.")
        
        ylen = tomo.nm2pixel(self._last_ft_size)
        spl = tomo._splines[i]
        
        rmin = tomo.nm2pixel(spl.radius*GVar.inner)
        rmax = tomo.nm2pixel(spl.radius*GVar.outer)
        
        coords = spl.local_cylindrical((rmin, rmax), ylen, spl.anchors[j])
        coords = np.moveaxis(coords, -1, 0)
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
    
        self.layer_prof = viewer.add_points(
            **common_properties,
            name=SELECTION_LAYER_NAME,
            opacity=0.4, 
            edge_color="black",
            face_color="black",
            )
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
        self.orientation_choice = Ori.none
        return None
    
    @Spline_control.pos.connect
    def _imshow_all(self):
        tomo = self.tomogram
        i = self.Spline_control.num.value
        j = self.Spline_control.pos.value
        npaths = len(tomo.splines)
        if 0 == npaths:
            return
        if 0 < npaths <= i:
            i = 0
        spl = tomo.splines[i]
        
        if spl.localprops is not None:
            headers = [H.yPitch, H.skewAngle, H.nPF, H.start]
            pitch, skew, npf, start = spl.localprops[headers].iloc[j]
            self.Local_Properties.params._set_text(pitch, skew, npf, start)
        
        binsize = self.tomogram.metadata["binsize"]
        with no_verbose():
            proj = self.projections[j]
            for ic in range(3):
                self.canvas[ic].layers.clear()
            self.canvas[0].image = proj.yx
            self.canvas[1].image = proj.zx
            self.canvas[2].image = proj.zx_ave
        
        # Update text overlay
        self.canvas[0].text_overlay.text = f"{i}-{j}"
        self.canvas[0].text_overlay.color = "lime"
        
        if spl.radius is None:
            return None
        lz, ly, lx = np.array(proj.shape)
        
        if self._last_ft_size is None:
            ylen = 25/binsize/tomo.scale
        else:
            ylen = self._last_ft_size/2/binsize/tomo.scale
        
        # draw a square in YX-view
        ymin, ymax = ly/2 - ylen - 0.5, ly/2 + ylen + 0.5
        r_px = spl.radius/tomo.scale/binsize
        r = r_px * GVar.outer
        xmin, xmax = -r + lx/2 - 0.5, r + lx/2 + 0.5
        self.canvas[0].add_curve([xmin, xmin, xmax, xmax, xmin], 
                                 [ymin, ymax, ymax, ymin, ymin], color="lime")

        # draw two circles in ZX-view
        theta = np.linspace(0, 2*np.pi, 360)
        r = r_px * GVar.inner
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = r_px * GVar.outer
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        return None
    
    @orientation_choice.connect
    def _update_note(self):
        i = self.Spline_control.num.value
        self.tomogram.splines[i].orientation = self.orientation_choice
        return None
    
    @Spline_control.num.connect
    def _update_spline(self):
        i = self.Spline_control.num.value
        tomo = self.tomogram
        spl = tomo.splines[i]
        if spl._anchors is None:
            return
        
        # calculate projection
        binsize = tomo.metadata["binsize"]
        imgb = self.layer_image.data
        
        # Rotational average should be calculated using local nPF if possible.
        # If not available, use global nPF
        projections: List[Projections] = []
        if spl.localprops is not None:
            npf_list = spl.localprops[H.nPF]
        elif spl.globalprops is not None:
            npf_list = [spl.globalprops[H.nPF]] * tomo.splines[i].anchors.size
        else:
            return None
        
        spl.scale *= binsize
        
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        out = load_rot_subtomograms(imgb, length_px, width_px, spl)
        
        spl.scale /= binsize
        
        for img, npf in zip(out, npf_list):    
            proj = Projections(img)
            proj.rotational_average(npf)
            projections.append(proj)
        
        self.projections = projections
        
        self.orientation_choice = Ori(tomo.splines[i].orientation)
        self._plot_properties()
        self._imshow_all()
        return None
    
    def _connect_worker(self, worker: Worker):
        self._worker_control._set_worker(worker)
        viewer: napari.Viewer = self.parent_viewer
        viewer.window._status_bar._toggle_activity_dock(True)
        dialog = viewer.window._qt_window._activity_dialog
        
        @worker.finished.connect
        def _on_finish(*args):
            viewer.window._status_bar._toggle_activity_dock(False)
            dialog.layout().removeWidget(self._worker_control.native)

        dialog.layout().addWidget(self._worker_control.native)
        return None
        
    def _add_spline_to_images(self, spl: MtSpline):
        interval = 15
        length = spl.length()
        scale = self.layer_image.scale[0]
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        self.layer_prof.add(fit)
        self.Panels.overview.add_curve(fit[:, 2]/scale, fit[:, 1]/scale, color="lime", lw=3)
    
    def _update_splines_in_images(self):
        self.Panels.overview.layers.clear()
        self.layer_prof.data = []
        scale = self.layer_image.scale[0]
        for spl in self.tomogram.splines:
            self._add_spline_to_images(spl)
            if spl._anchors is None:
                continue
            coords = spl()
            self.Panels.overview.add_scatter(coords[:, 2]/scale, coords[:, 1]/scale,
                                             color="lime", symbol="x", lw=1, size=10)
        

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

def change_viewer_focus(viewer: "napari.Viewer", next_center: Iterable[float], 
                        next_coord: np.ndarray):
    viewer.camera.center = next_center
    zoom = viewer.camera.zoom
    viewer.camera.events.zoom()
    viewer.camera.zoom = zoom
    viewer.dims.current_step = list(next_coord.astype(np.int64))

def _iter_run(tomo: MtTomogram, 
              interval: nm,
              ft_size,
              n_refine,
              dense_mode,
              dense_mode_sigma,
              local_props,
              global_props):
    
    tomo.fit(dense_mode=dense_mode, dense_mode_sigma=dense_mode_sigma)
    tomo.set_radius()
    
    for i in range(n_refine):
        if n_refine == 1:
            yield "Spline refinement ..."
        else:
            yield f"Spline refinement (iteration {i+1}/{n_refine}) ..."
        tomo.refine(max_interval=max(interval, 30))
        tomo.set_radius()
        
    tomo.make_anchors(interval=interval)
    if local_props:
        yield "Local Fourier transformation ..."
        tomo.local_ft_params(ft_size=ft_size)
    if global_props:
        yield "Global Fourier transformation ..."
        tomo.global_ft_params()
    yield "Finishing ..."
    return tomo


def _coerce_aligned_name(name: str):
    num = 1
    if re.match(fr".*-{ALN_SUFFIX}(\d)+", name):
        try:
            *pre, suf = name.split(f"-{ALN_SUFFIX}")
            num = int(suf) + 1
            name = "".join(pre)
        except Exception:
            num = 1
    return name + f"-{ALN_SUFFIX}{num}"