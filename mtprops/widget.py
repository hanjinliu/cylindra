from functools import wraps
import pandas as pd
from typing import Any, Callable, Iterable, NewType, Union
import os
import numpy as np
import warnings
import napari
from napari.utils import Colormap
from napari.qt import create_worker
from napari._qt.qthreading import GeneratorWorker, FunctionWorker
from napari.layers import Points, Layer, Image, Labels
from pathlib import Path

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
    MagicTemplate,
    bind_key,
    build_help,
    )
from magicclass.widgets import TupleEdit, Separator, ListWidget, Table
from magicclass.ext.pyqtgraph import QtImageCanvas, QtMultiPlotCanvas, QtMultiImageCanvas
from magicclass.utils import show_messagebox, to_clipboard

from mtprops.molecules import Molecules

from .tomogram import Coordinates, MtSpline, MtTomogram, cachemap, angle_corr, dask_affine, centroid
from .utils import (
    Projections,
    load_a_subtomogram,
    make_slice_and_pad,
    map_coordinates,
    mirror_pcc, 
    roundint,
    ceilint,
    load_rot_subtomograms,
    no_verbose
    )
from .const import EulerAxes, Unit, nm, H, Ori, GVar, Sep

# TODO: when anchor is updated (especially, "Fit splines manually" is clicked), spinbox and slider
# should also be initialized.

WORKING_LAYER_NAME = "Working Layer"
SELECTION_LAYER_NAME = "Selected MTs"
ICON_DIR = Path(__file__).parent / "icons"
MOLECULES = "Molecules"

import macrokit as mkit
import magicgui
from magicgui.widgets._bases import CategoricalWidget
from napari.utils._magicgui import find_viewer_ancestor

mkit.register_type(np.ndarray, lambda arr: str(arr.tolist()))
_mkit_viewer = mkit.Symbol.var("viewer")

@mkit.register_type(Layer)
def _get_layer_macro(layer: Layer):
    expr = mkit.Expr("getitem", 
                     [mkit.Expr("getattr", [_mkit_viewer, "layers"]),
                      layer.name]
                     )
    return expr

MonomerLayer = NewType("MonomerLayer", Points)
Worker = Union[FunctionWorker, GeneratorWorker]

def get_monomer_layers(gui: CategoricalWidget) -> list[Points]:
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, Points) and MOLECULES in x.metadata]

magicgui.register_type(MonomerLayer, choices=get_monomer_layers)

def run_worker_function(worker: Worker):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("always")
            warnings.showwarning = lambda *w: worker.warned.emit(w)
            result = worker.work()
        if isinstance(result, Exception):
            raise result
        worker.returned.emit(result)
    except Exception as exc:
        worker.errored.emit(exc)
    worker._running = False
    worker.finished.emit()
    worker._finished.emit(worker)


def dispatch_worker(f: Callable[[Any], Worker]) -> Callable[[Any], None]:
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        worker = f(self, *args, **kwargs)
        if self[f.__name__].running:
            self._connect_worker(worker)
            worker.start()
        else:
            run_worker_function(worker)
        return None
    return wrapper
    
### Child widgets ###

@magicclass(layout="horizontal", labels=False, error_mode="stderr")
class WorkerControl(MagicTemplate):
    # A widget that has a napari worker object and appears as buttons in the activity dock 
    # while running.
    
    info = field(str, record=False)
    
    def __post_init__(self):
        self.paused = False
        self.worker: Worker = None
        self._last_info = ""
    
    def _set_worker(self, worker):
        self.worker = worker
        @worker.errored.connect
        def _(e):
            # In some environments, errors raised in workers are completely hidden.
            # We have to re-raise it here.
            show_messagebox("error", title=e.__class__.__name__, text=str(e), parent=self.native)
        
    def Pause(self):
        """Pause/Resume thread."""        
        if not isinstance(self.worker, GeneratorWorker):
            return
        if self.paused:
            self.worker.resume()
            self["Pause"].text = "Pause"
            self.info.value = self._last_info
        else:
            self.worker.pause()
            self["Pause"].text = "Resume"
            self._last_info = self.info.value
            self.info.value = "Pausing"
        self.paused = not self.paused
        
    def Interrupt(self):
        """Interrupt thread."""
        self.worker.quit()


@magicclass
class SplineFitter(MagicTemplate):
    # Manually fit MT with spline curve using longitudinal projections
    
    canvas = field(QtImageCanvas, options={"lock_contrast_limits": True})
        
    @magicclass(layout="horizontal")
    class mt(MagicTemplate):
        """MT sub-regions"""
        mtlabel = field(int, options={"max": 0, "tooltip": "Number of MT"}, 
                        name="MTLabel", record=False)
        pos = field(int, options={"max": 0, "tooltip": "Position in a MT"},
                    name="Pos", record=False)
        def Fit(self): ...
        
        @bind_key("Up")
        @do_not_record
        def _next_pos(self):
            self.pos.value = min(self.pos.value + 1, self.pos.max)
        
        @bind_key("Down")
        @do_not_record
        def _prev_pos(self):
            self.pos.value = max(self.pos.value - 1, self.pos.min)
            
    
    @magicclass(widget_type="collapsible")
    class Rotational_averaging(MagicTemplate):
        canvas_rot = field(QtImageCanvas, options={"lock_contrast_limits": True})

        @magicclass(layout="horizontal")
        class frame:
            nPF = field(10, options={"min": 1, "max": 48, "tooltip": "Number of protofilament (if nPF=12, rotational average will be calculated by summing up every 30° rotated images)."}, record=False)
            cutoff = field(0.2, options={"min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Relative cutoff frequency of low-pass filter."}, record=False)
            def Average(self): ...
    
    def _get_shifts(self, _=None):
        i = self.mt.mtlabel.value
        return self.shifts[i]
    
    @mt.wraps
    def Fit(self, shifts: Bound(_get_shifts), i: Bound(mt.mtlabel)):
        """Fit current spline."""        
        shifts = np.asarray(shifts)
        spl = self.splines[i]
        sqsum = GVar.splError**2 * shifts.shape[0]
        spl.shift_fit(shifts=shifts*self.binsize, s=sqsum)
        spl.make_anchors(max_interval=self.max_interval)
        self.fit_done = True
        self._mt_changed()
        self.__magicclass_parent__._update_splines_in_images()
    
    @Rotational_averaging.frame.wraps
    @do_not_record
    def Average(self):
        """Show rotatinal averaged image."""        
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        parent: MTPropsWidget = self.__magicclass_parent__
        
        with no_verbose():
            img = parent._current_cartesian_img(i, j)
            cutoff = self.Rotational_averaging.frame.cutoff.value
            if 0 < cutoff < 0.5:
                img = img.lowpass_filter(cutoff=cutoff)
            proj = Projections(img)
            proj.rotational_average(self.Rotational_averaging.frame.nPF.value)
        self.Rotational_averaging.canvas_rot.image = proj.zx_ave
    
    
    def __post_init__(self):
        self.shifts: list[np.ndarray] = None
        self.canvas.min_height = 160
        self.fit_done = True
        self.canvas.add_infline(pos=[0, 0], angle=90, color="lime", lw=2)
        self.canvas.add_infline(pos=[0, 0], angle=0, color="lime", lw=2)
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        cos = np.cos(theta)
        sin = np.sin(theta)
        self.canvas.add_curve(cos, sin, color="lime", lw=2, ls="--")
        self.canvas.add_curve(2*cos, 2*sin, color="lime", lw=2, ls="--")
        self.mt.max_height = 50
        self.mt.height = 50
        
        @self.canvas.mouse_click_callbacks.append
        def _(e):
            if "left" not in e.buttons():
                return
            self.fit_done = False
            x, z = e.pos()
            self._update_cross(x, z)
    
    def _update_cross(self, x: float, z: float):
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        
        itemv = self.canvas.layers[0]
        itemh = self.canvas.layers[1]
        item_circ_inner = self.canvas.layers[2]
        item_circ_outer = self.canvas.layers[3]
        itemv.pos = [x, z]
        itemh.pos = [x, z]
        
        tomo: MtTomogram = self.__magicclass_parent__.active_tomogram
        r_max: nm = tomo.subtomo_width/2
        nbin = max(roundint(r_max/tomo.scale/self.binsize/2), 8)
        prof = self.subtomograms[j].radial_profile(center=[z, x], nbin=nbin, r_max=r_max)
        if tomo.light_background:
            prof = -prof
        imax = np.argmax(prof)
        imax_sub = centroid(prof, imax-5, imax+5)
        r_peak = (imax_sub+0.5)/nbin*r_max/tomo.scale/self.binsize
        
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        item_circ_inner.xdata = r_peak * GVar.inner * np.cos(theta) + x
        item_circ_inner.ydata = r_peak * GVar.inner * np.sin(theta) + z
        item_circ_outer.xdata = r_peak * GVar.outer * np.cos(theta) + x
        item_circ_outer.ydata = r_peak * GVar.outer * np.sin(theta) + z
        
        lz, lx = self.subtomograms.sizesof("zx")
        self.shifts[i][j, :] = z - lz/2 + 0.5, x - lx/2 + 0.5
        return None
    
    def _load_parent_state(self, max_interval: nm):
        self.max_interval = max_interval
        tomo: MtTomogram = self.__magicclass_parent__.active_tomogram
        for i in range(tomo.n_splines):
            spl = tomo.splines[i]
            spl.make_anchors(max_interval=self.max_interval)
            
        self.shifts = [None] * tomo.n_splines
        self.binsize = tomo.metadata["binsize"]
        self.mt.mtlabel.max = tomo.n_splines - 1
        self.mt.mtlabel.value = 0
        self._mt_changed()
        
    @mt.mtlabel.connect
    def _mt_changed(self):
        i = self.mt.mtlabel.value
        self.mt.pos.value = 0
        imgb = self.__magicclass_parent__.layer_image.data
        tomo: MtTomogram = self.__magicclass_parent__.active_tomogram
        
        spl = tomo.splines[i]
        self.splines = tomo.splines
        npos = spl.anchors.size
        self.shifts[i] = np.zeros((npos, 2))
        
        spl.scale *= self.binsize
        length_px = tomo.nm2pixel(tomo.subtomo_length/self.binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/self.binsize)
        
        with no_verbose():
            out = load_rot_subtomograms(imgb, length_px, width_px, spl)
            self.subtomograms = out.proj("y")["x=::-1"]
            
        # Restore spline scale.
        spl.scale /= self.binsize
        self.canvas.image = self.subtomograms[0]
        self.mt.pos.max = npos - 1
        self.canvas.xlim = (0, self.canvas.image.shape[1])
        self.canvas.ylim = (0, self.canvas.image.shape[0])
        lz, lx = self.subtomograms.sizesof("zx")
        self._update_cross(lx/2 - 0.5, lz/2 - 0.5)
        
        del self.Rotational_averaging.canvas_rot.image # to avoid confusion
        
        return None
    
    @mt.pos.connect
    def _position_changed(self):
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        self.canvas.image = self.subtomograms[j]
        if self.shifts is not None and self.shifts[i] is not None:
            y, x = self.shifts[i][j]
        else:
            y = x = 0
        lz, lx = self.subtomograms.shape[-2:]
        self._update_cross(x + lx/2 - 0.5, y + lz/2 - 0.5)


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
        def Load_json(self): ...
        def Save_results_as_json(self): ...
        def Save_results_as_csv(self): ...
        def Save_monomer_coordinates(self): ...
        def Save_monomer_angles(self): ...
    
    @magicmenu
    class View(MagicTemplate):
        """Visualization."""
        def Apply_lowpass_to_reference_image(self): ...
        sep0 = field(Separator)
        def show_current_ft(self): ...
        def show_global_ft(self): ...
        def show_r_proj(self): ...
        def show_global_r_proj(self): ...
        sep1 = field(Separator)
        def Show_splines(self): ...
        def Show_results_in_a_table_widget(self): ...
        def Show_straightened_image(self): ...
        def Paint_MT(self): ...
        def Set_colormap(self): ...
        focus = field(False, options={"text": "Focus"}, record=False)
    
    @magicmenu
    class Analysis(MagicTemplate):
        """Analysis of tomograms."""        
        def Fit_splines(self): ...
        def Fit_splines_manually(self): ...                
        def Add_anchors(self): ...
        def Measure_radius(self): ...
        def Refine_splines(self): ...
        def Refine_splines_with_MAO(self): ...
        sep0 = field(Separator)
        def Local_FT_analysis(self): ...
        def Global_FT_analysis(self): ...
        sep1 = field(Separator)
        @magicmenu
        class Reconstruction(MagicTemplate):
            def Reconstruct_MT(self): ...
            def cylindric_reconstruction(self): ...
        def Map_monomers(self): ...
        def Map_monomers_manually(self): ...
    
    @magicmenu
    class Others(MagicTemplate):
        """Other menus."""
        def Create_macro(self): ...
        def Global_variables(self, **kwargs): ...
        def MTProps_info(self): ...
        def Open_help(self): ...
        
    @magictoolbar(labels=False)
    class toolbar(MagicTemplate):
        """Frequently used operations."""        
        def register_path(self): ...
        def open_runner(self): ...
        def pick_next(self): ...
        def auto_center(self): ...
        def clear_current(self): ...
        def clear_all(self): ...
        stride = field(50.0, widget_type="FloatSlider", options={"min": 10, "max": 100, "tooltip": "Stride length (nm) of auto picker"}, record=False)
        
    @magicclass(widget_type="collapsible")
    class Tomogram_List(MagicTemplate):
        """List of tomograms that have loaded to the widget."""        
        tomograms = ListWidget(name="Tomogram List")
    
    @magicclass(layout="horizontal")
    class mt(MagicTemplate):
        """MT sub-regions"""
        mtlabel = field(int, options={"max": 0, "tooltip": "Number of MT."}, name="MTLabel", record=False)
        pos = field(int, widget_type="Slider", options={"max": 0, "tooltip": "Position along a MT."}, name="Pos", record=False)
    
    canvas = field(QtMultiImageCanvas, name="Figure", options={"nrows": 1, "ncols": 3, "tooltip": "Projections"})
    
    @magicclass(widget_type="collapsible")
    class Profiles(MagicTemplate):
        """Local profiles."""
        txt = field(str, options={"enabled": False, "tooltip": "Structural parameters at current MT position."}, name="result")    
        orientation_choice = field(Ori.none, name="Orientation: ", options={"tooltip": "MT polarity."})
        plot = field(QtMultiPlotCanvas, name="Plot", options={"nrows": 2, "ncols": 1, "sharex": True, "tooltip": "Plot of local properties"})

    @magicclass(widget_type="tabbed")
    class Panels(MagicTemplate):
        """Panels for output."""
        overview = field(QtImageCanvas, name="Overview", options={"tooltip": "Overview of splines"})
        image2D = field(QtImageCanvas, options={"tooltip": "2-D image viewer."})
        table = field(Table, name="Table", options={"tooltip": "Result table"})
    
    ### methods ###
        
    def __post_init__(self):
        self.active_tomogram: MtTomogram = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        
        self.Set_colormap()
        self.mt.pos.min_width = 70
        
        tomograms = self.Tomogram_List.tomograms
        
        @tomograms.register_callback(MtTomogram)
        def open_tomogram(tomo: MtTomogram, i: int):
            if tomo is self.active_tomogram:
                return None
            self.active_tomogram = tomo
            
            # Load dask again. Here, lowpass filter is already applied so that cutoff frequency
            # should be set to 0.
            worker = self._get_process_image_worker(
                tomo.image, tomo.metadata["binsize"], 
                tomo.light_background, tomo.metadata["cutoff"],
                tomo.subtomo_length, tomo.subtomo_width,
                new=False
                )
            
            self._connect_worker(worker)
            worker.start()
            
            if tomo.splines:
                worker.finished.connect(self._load_tomogram_results)
            else:
                worker.finished.connect(self._init_layers)
                worker.finished.connect(self._init_widget_params)
        
        @tomograms.register_contextmenu(MtTomogram)
        def Load_tomogram(tomo: MtTomogram, i: int):
            open_tomogram(tomo, i)
        
        @tomograms.register_contextmenu(MtTomogram)
        def Remove_tomogram_from_list(tomo: MtTomogram, i: int):
            tomograms.pop(i)
            
        @tomograms.register_contextmenu(MtTomogram)
        def Copy_path(tomo: MtTomogram, i: int):
            if "source" in tomo.metadata:
                to_clipboard(tomo.metadata["source"])
        
        @tomograms.register_tooltip(MtTomogram)
        def _tooltip(tomo: MtTomogram):
            gb = tomo.image.gb
            return f"{gb:.3g} GB"
            
        tomograms.height = 160
        tomograms.max_height = 160
        self.min_width = 395
        
        # Initialize multi-image canvas
        self.canvas.min_height = 200
        self.canvas.max_height = 230
        self.canvas[0].lock_contrast_limits = True
        self.canvas[0].title = "XY-Projection"
        self.canvas[1].lock_contrast_limits = True
        self.canvas[1].title = "XZ-Projection"
        self.canvas[2].lock_contrast_limits = True
        self.canvas[2].title = "Rot. average"
        
        self.Profiles.collapsed = False
        
        # Initialize multi-plot canvas
        self.Profiles.plot.min_height = 240
        self.Profiles.plot[0].ylabel = "pitch (nm)"
        self.Profiles.plot[0].legend.visible = False
        self.Profiles.plot[0].border = [1, 1, 1, 0.2]
        self.Profiles.plot[1].xlabel = "position (nm)"
        self.Profiles.plot[1].ylabel = "skew (deg)"
        self.Profiles.plot[1].legend.visible = False
        self.Profiles.plot[1].border = [1, 1, 1, 0.2]
        
        self.Panels.min_height = 300


    def _get_spline_coordinates(self, widget=None) -> np.ndarray:
        """Get coordinates of the manually picked spline."""
        coords = self.layer_work.data
        return coords
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"add_spline.png")
    @bind_key("F1")
    def register_path(self, coords: Bound(_get_spline_coordinates) = None):
        """Register current selected points as a MT path."""        
        if coords is None:
            coords = self.layer_work.data
        else:
            coords = np.asarray(coords)
        
        if coords.size == 0:
            return None

        self.active_tomogram.add_spline(coords)
        spl = self.active_tomogram.splines[-1]
        
        # draw path
        self._add_spline_to_images(spl)
        self.layer_work.data = []
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
    def run_mtprops(self,
                    interval: Bound(_runner.params2.interval),
                    ft_size: Bound(_runner.params2.ft_size),
                    n_refine: Bound(_runner.n_refine),
                    dense_mode: Bound(_runner.dense_mode),
                    dense_mode_sigma: Bound(_runner.params1.dense_mode_sigma),
                    local_props: Bound(_runner.local_props),
                    global_props: Bound(_runner.global_props),
                    paint: Bound(_runner.params2.paint)):
        """Run MTProps"""
        self._runner.close()
        if self.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        
        total = 1 + n_refine + int(local_props) + int(global_props)
        
        worker = create_worker(_iter_run, 
                               tomo=self.active_tomogram,
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
                self._worker_control.info.value = out
                self._update_splines_in_images()
            
        @worker.returned.connect
        def _on_return(tomo: MtTomogram):
            self._update_splines_in_images()
            if local_props:
                self._load_tomogram_results()
                if paint:
                    self.Paint_MT()
            if global_props:
                self._globalprops_to_table(tomo.global_ft_params())
            
        self._worker_control.info.value = "Spline fitting"
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
        
        cachemap.clear()
        self.active_tomogram._splines.clear()
        
        return None
    
    @Others.wraps
    @do_not_record
    def Create_macro(self):
        """Create Python executable script."""
        self.macro.widget.duplicate().show()
        return None
    
    @Others.wraps
    @set_options(yPitchMin={"step": 0.1},
                 yPitchMax={"step": 0.1},
                 minSkew={"min": -90, "max": 90},
                 maxSkew={"min": -90, "max": 90},
                 splError={"max": 5.0, "step": 0.1},
                 inner={"step": 0.1},
                 outer={"step": 0.1},
                 daskChunk={"widget_type": TupleEdit, "options": {"min": 16, "max": 2048, "step": 16}})
    def Global_variables(self, 
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
                         daskChunk: int = GVar.daskChunk):
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
        for spl in self.active_tomogram.splines:
            spl.localprops = None
            spl.globalprops = None
        
    
    @Others.wraps
    @do_not_record
    def MTProps_info(self):
        """Show information of dependencies."""
        import napari
        import magicgui
        from .__init__ import __version__
        import magicclass as mcls
        import dask
        
        value = f"MTProps: {__version__}\n"\
                f"impy: {ip.__version__}\n"\
                f"magicgui: {magicgui.__version__}\n"\
                f"magicclass: {mcls.__version__}\n"\
                f"napari: {napari.__version__}\n"\
                f"dask: {dask.__version__}\n"
        show_messagebox(title="MTProps info", text=value, parent=self.native)
        return None
    
    @Others.wraps
    @do_not_record
    def Open_help(self):
        """Open a help window."""
        help = build_help(self)
        help.show()
        return None
    
    @magicclass
    class _loader(MagicTemplate):
        # A loader widget with imread settings.
        path = vfield(Path, record=False, options={"filter": "*.tif;*.tiff;*.mrc;*.rec", "tooltip": "Path to tomogram."})
        scale = vfield(str, record=False, options={"label": "scale (nm)", "tooltip": "Pixel size in nm/pixel."})
        bin_size = vfield(4, record=False, options={"label": "bin size", "min": 1, "max": 8, "tooltip": "Bin size of image for reference. This value does not affect MTProps analysis."})
        subtomo_length = vfield(48.0, record=False, options={"label": "subtomogram length (nm)", "min": 2.0, "max": 100.0, "step": 4.0, "tooltip": "The axial length of subtomogram."})
        subtomo_width = vfield(44.0, record=False, options={"label": "subtomogram width (nm)", "min": 2.0, "max": 100.0, "step": 4.0, "tooltip": "The diameter of subtomogram."})
        light_background = vfield(True, record=False, options={"label": "light background", "tooltip": "Check if background is bright."})
        use_lowpass = vfield(False, record=False, options={"label": "Apply low-pass filter","tooltip": "Check if images need prefilter."})
        cutoff_freq = vfield(0.2, record=False, options={"label": "Cutoff frequency (1/px)", "visible": False, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Relative cutoff frequency of low-pass prefilter. Must be 0.0 < freq < 0.5."})
        
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
            self.scale = f"{img.scale.x:.3f}"
        
        def load_tomogram(self): ...
    
    @_loader.wraps
    @set_design(text="OK")
    @dispatch_worker
    def load_tomogram(self, 
                      path: Bound(_loader.path),
                      scale: Bound(_loader.scale),
                      bin_size: Bound(_loader.bin_size),
                      light_background: Bound(_loader.light_background),
                      cutoff: Bound(_loader._get_cutoff_freq),
                      subtomo_length: Bound(_loader.subtomo_length),
                      subtomo_width: Bound(_loader.subtomo_width)
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
    @set_options(path={"filter": "*.json;*.txt"})
    @dispatch_worker
    def Load_json(self, path: Path):
        """Choose a json file and load it."""        
        tomo = self.active_tomogram
        worker = create_worker(tomo.load_json, path, _progress={"total": 0, "desc": "Running"})
        worker.returned.connect(self._load_tomogram_results)
        self._worker_control.info.value = f"Loading {os.path.basename(path)}"
        return worker
    
    @File.wraps
    @set_design(text="Save results as json")
    @set_options(save_path={"mode": "w", "filter": "*.json;*.txt"})
    def Save_results_as_json(self, save_path: Path):
        """Save the results as json."""
        self.active_tomogram.save_json(save_path)
        return None
    
    @File.wraps
    @set_options(save_path={"mode": "w", "filter": "*.txt;*.csv;*.dat"})
    def Save_monomer_coordinates(self,
                                 save_path: Path,
                                 layer: MonomerLayer, 
                                 separator = Sep.Comma,
                                 unit = Unit.pixel):
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
        """        
        unit = Unit(unit)
        if unit == Unit.pixel:
            arr = layer.data / self.active_tomogram.scale
        elif unit == Unit.nm:
            arr = layer.data
        elif unit == Unit.angstrom:
            arr = layer.data * 10
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
        mol: Molecules = layer.metadata[MOLECULES]
        arr = mol.euler_angle(rotation_axes, degrees=in_degree)
        np.savetxt(save_path, arr, delimiter=str(separator))
        return None
    
    @View.wraps
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
        self._worker_control.info.value = "Low-pass filtering"

        @worker.returned.connect
        def _on_return(contrast_limits):
            self.layer_image.contrast_limits = contrast_limits
            proj = self.layer_image.data.proj("z")
            self.Panels.overview.image = proj
            self.Panels.overview.contrast_limits = contrast_limits
        
        return worker
                    
    @mt.mtlabel.connect
    @mt.pos.connect
    @View.focus.connect
    def _focus_on(self):
        """Change camera focus to the position of current MT fragment."""
        if self.layer_paint is None:
            return None
        if not self.View.focus.value:
            self.layer_paint.show_selected_label = False
            return None
        
        viewer = self.parent_viewer
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        
        tomo = self.active_tomogram
        spl = tomo.splines[i]
        pos = spl.anchors[j]
        next_center = spl(pos)
        viewer.dims.current_step = list(next_center.astype(np.int64))
        
        viewer.camera.center = next_center
        zoom = viewer.camera.zoom
        viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        viewer.camera.zoom = zoom
        
        self.layer_paint.show_selected_label = True
        
        j_offset = sum(spl.anchors.size for spl in tomo.splines[:i])
        self.layer_paint.selected_label = j_offset + j + 1
        return None
    
    @View.wraps
    def Show_results_in_a_table_widget(self):
        """Show result table."""
        self.Panels.table.value = self.active_tomogram.collect_localprops()
        self.Panels.current_index = 2
        return None
    
    @View.wraps
    @dispatch_worker
    def Show_straightened_image(self, i: Bound(mt.mtlabel)):
        """Send straightened image of the current MT to the viewer."""        
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.straighten, 
                               i=i, 
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.ImgArray):
            self.parent_viewer.add_image(out, scale=out.scale)
        
        self._worker_control.info.value = f"Straightening spline No. {i}"
        
        return worker
    
    @View.wraps
    @set_design(text="R-projection")
    def show_r_proj(self, i: Bound(mt.mtlabel), j: Bound(mt.pos)):
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
    
    @View.wraps
    @set_design(text="R-projection (Global)")
    def show_global_r_proj(self):
        """Show radial projection of cylindrical image along current MT."""        
        i = self.mt.mtlabel.value
        with no_verbose():
            polar = self.active_tomogram.cylindric_straighten(i).proj("r")
        self.Panels.image2D.image = polar.value
        self.Panels.image2D.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        # move to center
        ly, lx = polar.shape
        self.Panels.image2D.xlim = [lx*0.3, lx*0.7]
        self.Panels.current_index = 1
        return None
    
    @View.wraps
    @set_design(text="2D-FT")
    def show_current_ft(self, i: Bound(mt.mtlabel), j: Bound(mt.pos)):
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
    
    @View.wraps
    @set_design(text="2D-FT (Global)")
    def show_global_ft(self, i: Bound(mt.mtlabel)):
        """View Fourier space along current MT."""  
        with no_verbose():
            polar = self.active_tomogram.cylindric_straighten(i)
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
    
    @View.wraps
    def Show_splines(self):
        """Show 3D spline paths of microtubule center axes as a layer."""        
        paths = [r.partition(100) for r in self.active_tomogram.splines]
        
        self.parent_viewer.add_shapes(paths, shape_type="path", edge_color="lime", edge_width=1,
                                      translate=self.layer_image.translate)
        return None
    
    @Analysis.wraps
    @set_options(max_interval={"label": "Max interval (nm)"})
    @dispatch_worker
    def Fit_splines(self, 
                    max_interval: nm = 30,
                    degree_precision: float = 0.2,
                    dense_mode: bool = False,
                    ):
        """
        Fit MT with spline curve, using manually selected points.

        Parameters
        ----------
        max_interval : nm, default is 30.0
            Maximum interval of sampling points in nm unit.
        degree_precision : float, default is 0.2
            Precision of MT xy-tilt degree in angular correlation.
        dense_mode : bool, default is False
            Check if microtubules are densely packed. Initial spline position must be "almost" fitted
            in dense mode.
        """        
        worker = create_worker(self.active_tomogram.fit,
                               max_interval=max_interval,
                               degree_precision=degree_precision,
                               dense_mode=dense_mode,
                               _progress={"total": 0, "desc": "Running"}
                               )
        worker.returned.connect(self._init_layers)
        worker.returned.connect(self._update_splines_in_images)
        self._worker_control.info.value = "Spline Fitting"

        return worker
    
    @Analysis.wraps
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
    
    @Analysis.wraps
    @set_options(interval={"label": "Interval between anchors (nm)"})
    def Add_anchors(self, interval: nm = 25.0):
        """
        Add anchors to splines.

        Parameters
        ----------
        interval : nm, default is 25.0
            Anchor interval.
        """        
        tomo = self.active_tomogram
        if tomo.n_splines == 0:
            raise ValueError("Cannot add anchors before adding splines.")
        for i in range(tomo.n_splines):
            tomo.make_anchors(i, interval=interval)
        self._update_splines_in_images()
        return None
    
    @Analysis.wraps
    @dispatch_worker
    def Measure_radius(self):
        """Measure MT radius for each spline path."""        
        worker = create_worker(self.active_tomogram.measure_radius,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        self._worker_control.info.value = "Measuring Radius"

        return worker
    
    @Analysis.wraps
    @set_options(max_interval={"label": "Maximum interval (nm)"},
                 mask_8nm={"label": "Mask 8-nm peak"})
    @dispatch_worker
    def Refine_splines(self, max_interval: nm = 30, projection: bool = True, 
                       mask_8nm: bool = False):
        """
        Refine splines using the global MT structural parameters.
        
        Parameters
        ----------
        max_interval : nm, default is 30
            Maximum interval between anchors.
        projection : bool, default is True
            Check and Y-projection will be used to align subtomograms.
        mask_8nm : bool, default is False
            Chech if mask 8-nm peak to prevent phase of binding proteins affects
            refinement results.
        """
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.refine,
                               max_interval=max_interval,
                               projection=projection,
                               mask_8nm=mask_8nm,
                               _progress={"total": 0, 
                                          "desc": "Running"})
        
        worker.finished.connect(self._update_splines_in_images)

        self._worker_control.info.value = "Refining splines ..."
        
        self._init_widget_params()
        self._init_figures()
        return worker

    
    @Analysis.wraps
    @set_options(max_interval={"label": "Maximum interval (nm)"})
    @dispatch_worker
    def Refine_splines_with_MAO(self, max_interval: nm = 30):
        """
        Refine splines using Minimum Angular Oscillation.
        
        Parameters
        ----------
        max_interval : nm, default is 30
            Maximum interval between anchors.
        """
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.fit_mao,
                               max_interval=max_interval,
                               _progress={"total": 0, 
                                          "desc": "Running"})
        
        worker.finished.connect(self._update_splines_in_images)

        self._worker_control.info.value = "Refining splines with MAO..."
        
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
        tomo = self.active_tomogram
        tomo.ft_size = ft_size
        if tomo.splines[0].radius is None:
            self.Measure_radius()
        self.Add_anchors(interval=interval)
        worker = create_worker(tomo.local_ft_params,
                               _progress={"total": 0, "desc": "Running"}
                               )
        @worker.returned.connect
        def _on_return(df):
            self._load_tomogram_results()
        
        self._worker_control.info.value = "Local Fourier transform ..."
        return worker
        
    @Analysis.wraps
    @dispatch_worker
    def Global_FT_analysis(self):
        """Determine MT global structural parameters by Fourier transformation."""        
        tomo = self.active_tomogram
        worker = create_worker(tomo.global_ft_params,
                               _progress={"total": 0, "desc": "Running"})
        worker.returned.connect(self._globalprops_to_table)
        
        self._worker_control.info.value = f"Global Fourier transform ..."
        
        return worker
    
    def _globalprops_to_table(self, out: list[pd.Series]):
        df = pd.DataFrame({f"MT-{k}": v for k, v in enumerate(out)})
        self.Panels.table.value = df
        self.Panels.current_index = 2
        return None
        
    @Analysis.Reconstruction.wraps
    @set_options(rot_ave={"label": "Rotational averaging"},
                 find_seam={"label": "Find seam position"},
                 niter={"label": "Iteration", "max": 3},
                 y_length={"label": "Longitudinal length (nm)"})
    @dispatch_worker
    def Reconstruct_MT(self, i: Bound(mt.mtlabel), rot_ave=False, find_seam=False, niter=1, y_length=50.0):
        """
        Coarse reconstruction of MT.

        Parameters
        ----------
        rot_ave : bool, default is False
            Check to run rotational averaging after reconstruction.
        find_seam : bool, default is False
            Check to find seam position while rotational averaging.
        niter : int, default is 1
            Number of iteration
        y_length : nm, default is 50.0
            Longitudinal length (nm) of reconstructed image.
        """        
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.reconstruct, 
                               i=i,
                               rot_ave=rot_ave, 
                               seam_offset="find" if find_seam else None,
                               niter=niter,
                               y_length=y_length,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.ImgArray):
            if tomo.light_background:
                out = -out
            _show_reconstruction(out, name=f"MT-{i} reconstruction")
        
        self._worker_control.info.value = f"Reconstruction ..."
        return worker
    
    @Analysis.Reconstruction.wraps
    @set_options(rot_ave={"label": "Rotational averaging"},
                 find_seam={"label": "Find seam position"},
                 niter={"label": "Iteration", "max": 3},
                 y_length={"label": "Longitudinal length (nm)"})
    @set_design(text="Reconstruct MT (cylindric)")
    @dispatch_worker
    def cylindric_reconstruction(self, i: Bound(mt.mtlabel), rot_ave=False, find_seam=False, niter=1, 
                                 y_length=50.0):
        """
        Cylindric reconstruction of MT.

        Parameters
        ----------
        rot_ave : bool, default is False
            Check to run rotational averaging after reconstruction.
        find_seam : bool, default is False
            Check to find seam position while rotational averaging.
        niter : int, default is 1
            Number of iteration
        y_length : nm, default is 48.0
            Longitudinal length (nm) of reconstructed image.
        """        
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.cylindric_reconstruct, 
                               i=i,
                               rot_ave=rot_ave, 
                               seam_offset="find" if find_seam else None,
                               niter=niter,
                               y_length=y_length,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: ip.ImgArray):
            if tomo.light_background:
                out = -out
            _show_reconstruction(out, name=f"MT-{i} cylindric reconstruction")
            
        self._worker_control.info.value = f"Cylindric reconstruction ..."
        return worker
    
    @Analysis.wraps
    @dispatch_worker
    def Map_monomers(self):
        """
        Map points to tubulin molecules using the results of global Fourier transformation.
        """        
        tomo = self.active_tomogram
        
        worker = create_worker(tomo.map_monomers,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: list[Coordinates]):
            for i, coords in enumerate(out):
                spl = tomo.splines[i]
                mol = spl.cylindrical_to_world_vector(coords.spline)
                points_layer = self.parent_viewer.add_points(
                    coords.world, size=3, face_color="lime", edge_color="lime",
                    n_dimensional=True, name=f"Monomers-{i}", metadata={MOLECULES: mol}
                    )
                
                points_layer.shading = "spherical"
                
                vector_data = np.stack([mol.pos, mol.z], axis=1)
                self.parent_viewer.add_vectors(
                    vector_data, edge_width=0.8, edge_color="crimson", length=2.4,
                    name=f"Monomer-{i} Z-axis",
                    )
    
        self._worker_control.info.value = "Monomer mapping ..."
        return worker

    @Analysis.wraps
    @set_options(auto_call=True, 
                 y_offset={"widget_type": "FloatSlider", "max": 5, "step": 0.1, "label": "y offset (nm)"},
                 theta_offset={"widget_type": "FloatSlider", "max": 180, "label": "θ offset (deg)"})
    def Map_monomers_manually(self, i: Bound(mt.mtlabel), y_offset: nm = 0, theta_offset: float = 0):
        theta_offset = np.deg2rad(theta_offset)
        tomo = self.active_tomogram
        tomo.global_ft_params(i)
        coords = tomo.map_monomers(i, offsets=(y_offset, theta_offset))
        spl = tomo.splines[i]
        mol = spl.cylindrical_to_world_vector(coords.spline)
        viewer = self.parent_viewer
        layer_name = f"Monomers-{i}"
        if layer_name not in viewer.layers:
            points_layer = self.parent_viewer.add_points(
                ndim=3, size=3, face_color="lime", edge_color="lime",
                n_dimensional=True, name=layer_name, metadata={MOLECULES: mol}
                )
            
            points_layer.shading = "spherical"
            
            self.parent_viewer.add_vectors(
                ndim=3, edge_width=0.8, edge_color="crimson", length=2.4,
                name=layer_name + " Z-axis",
                )
        
        points_layer = viewer.layers[layer_name]
        points_layer.data = coords.world
        vector_layer = viewer.layers[layer_name + " Z-axis"]
        vector_layer.data = np.stack([mol.pos, mol.z], axis=1)
        
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"pick_next.png")
    @do_not_record
    def pick_next(self):
        """Automatically pick MT center using previous two points."""        
        stride_nm = self.toolbar.stride.value
        imgb = self.layer_image.data
        try:
            # orientation is point0 -> point1
            point0: np.ndarray = self.layer_work.data[-2]/imgb.scale.x # unit: pixel
            point1: np.ndarray = self.layer_work.data[-1]/imgb.scale.x
        except IndexError:
            raise IndexError("Auto pick needs at least two points in the working layer.")
        
        tomo = self.active_tomogram
        binsize = roundint(self.layer_image.scale[0]/tomo.scale) # scale of binned reference image
        
        # shape = tomo.nm2pixel(np.array(tomo.box_size)/binsize)
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        
        shape = (width_px,) + (roundint((width_px+length_px)/1.41),)*2
        
        with no_verbose():
            orientation = point1[1:] - point0[1:]
            img = load_a_subtomogram(imgb, point1, shape)
            center = np.rad2deg(np.arctan2(*orientation)) % 180 - 90
            angle_deg = angle_corr(img, ang_center=center, drot=25, nrots=25)
            angle_rad = np.deg2rad(angle_deg)
            dr = np.array([0.0, stride_nm*np.cos(angle_rad), -stride_nm*np.sin(angle_rad)])
            if np.dot(orientation, dr[1:]) > np.dot(orientation, -dr[1:]):
                point2 = point1 + dr
            else:
                point2 = point1 - dr
            img_next = load_a_subtomogram(imgb, point2, shape)
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
        tomo = self.active_tomogram
        binsize = roundint(self.layer_image.scale[0]/tomo.scale) # scale of binned reference image
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
                img_input = load_a_subtomogram(imgb, point, shape)
                angle_deg = angle_corr(img_input, ang_center=0, drot=89.5, nrots=19)
                centering(img_input, point, angle_deg, drot=5, nrots=7)
                last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], self.layer_work.data[last_i])
        return None
    
    @View.wraps
    def Paint_MT(self):
        """
        Paint microtubule fragments by its pitch length.
        
        1. Prepare small boxes and make masks inside them.
        2. Map the masks to the reference image.
        3. Erase masks using reference image, based on intensity.
        """        
        lbl = np.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, list[float]] = {0: [0, 0, 0, 0]}
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        tomo = self.active_tomogram
        
        lz, ly, lx = [int(r/bin_scale*1.4)*2 + 1 for r in [15, tomo.ft_size/2, 15]]
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
                              tomo.ft_size/2) / bin_scale + 0.5 
                        
                    ry = max(ceilint(ry), 1)
                    domain[:, :ly//2-ry] = 0
                    domain[:, ly//2+ry+1:] = 0
                    domain = domain.astype(np.float32)
                    domains.append(domain)
                    
                cylinders.append(domains)
                matrices.append(spl.rotation_matrix(center=center))
            
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
    
    @View.wraps
    @set_options(start={"widget_type": TupleEdit, "options": {"step": 0.1}}, 
                 end={"widget_type": TupleEdit, "options": {"step": 0.1}},
                 limit={"widget_type": TupleEdit, "options": {"step": 0.02}, "label": "limit (nm)"})
    def Set_colormap(self, start=(0.0, 0.0, 1.0), end=(1.0, 0.0, 0.0), limit=(4.00, 4.24)):
        """
        Set the color-map for painting microtubules.
        
        Parameters
        ----------
        start : tuple, default is (0.0, 0.0, 1.0)
            RGB color that corresponds to the most compacted microtubule.
        end : tuple, default is (1.0, 0.0, 0.0)
            RGB color that corresponds to the most expanded microtubule.
        limit : tuple, default is (4.00, 4.24)
            Color limit (nm).
        """        
        self.label_colormap = Colormap([start+(1,), end+(1,)], name="PitchLength")
        self.label_colorlimit = limit
        self._update_colormap()
        return None
    
    
    def _update_colormap(self, prop: str = H.yPitch):
        # TODO: color by other properties
        if self.layer_paint is None:
            return None
        color = {0: np.array([0., 0., 0., 0.], dtype=np.float32),
                 None: np.array([0., 0., 0., 1.], dtype=np.float32)}
        lim0, lim1 = self.label_colorlimit
        df = self.active_tomogram.collect_localprops()[prop]
        for i, value in enumerate(df):
            color[i+1] = self.label_colormap.map((value - lim0)/(lim1 - lim0))
        self.layer_paint.color = color
        return None


    def _plot_properties(self):
        i = self.mt.mtlabel.value
        props = self.active_tomogram.splines[i].localprops
        if props is None:
            return None
        x = np.asarray(props[H.splDistance])
        pitch_color = "lime"
        skew_color = "gold"
        
        self.Profiles.plot[0].layers.clear()
        self.Profiles.plot[0].add_curve(x, props[H.yPitch], color=pitch_color)
        
        self.Profiles.plot[1].layers.clear()
        self.Profiles.plot[1].add_curve(x, props[H.skewAngle], color=skew_color)

        self.Profiles.plot.xlim = (x[0] - 2, x[-1] + 2)
        return None
        
    def _get_process_image_worker(self, img: ip.LazyImgArray, binsize: int, light_bg: bool, 
                                  cutoff: float, length: nm, width: nm, *, new: bool = True):
        """
        When an image is opened, we have to (1) prepare binned image for reference, (2) apply 
        low-pass filter if needed, (3) change existing layer scales if needed, (4) construct
        a new ``MtTomogram`` object if needed (5) make 2D projection. 
        """
        viewer = self.parent_viewer
        img = img.as_float()
        
        def _run(img: ip.LazyImgArray, binsize: int, cutoff: float):
            with no_verbose():
                if 0 < cutoff < 0.5:
                    img.tiled_lowpass_filter(cutoff, update=True)
                    img.release()
                imgb = img.binning(binsize, check_edges=False).data
            
            return imgb
        
        worker = create_worker(_run,
                               img=img,
                               binsize=binsize,
                               cutoff=cutoff,
                               _progress={"total": 0, "desc": "Reading Image"})

        self._worker_control.info.value = \
            f"Loading with {binsize}x{binsize} binned size: {tuple(s//binsize for s in img.shape)}"
        
        @worker.returned.connect
        def _on_return(imgb: ip.ImgArray):
            tr = (binsize - 1)/2*img.scale.x
            rendering = "minip" if light_bg else "mip"
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
                
            viewer.scale_bar.unit = img.scale_unit
            viewer.dims.axis_labels = ("z", "y", "x")
            
            if self.layer_paint is not None:
                self.layer_paint.scale = imgb.scale
                self.layer_paint.translate = [tr, tr, tr]
            
            with no_verbose():
                proj = imgb.proj("z")
            self.Panels.overview.image = proj
            self.Panels.overview.ylim = (0, proj.shape[0])
            
            if new:
                tomo = MtTomogram(subtomogram_length=length, 
                                  subtomogram_width=width, 
                                  light_background=light_bg)
                # metadata for GUI
                tomo.metadata["source"] = str(self._loader.path)
                tomo.metadata["binsize"] = binsize
                tomo.metadata["cutoff"] = cutoff
                
                tomo._set_image(img)
                self.active_tomogram = tomo
                self.Tomogram_List.tomograms.append(tomo)
                
                self.clear_all()
            
            return None
        
        return worker
    
    def _load_tomogram_results(self):
        self._spline_fitter.close()
        tomo = self.active_tomogram
        # initialize GUI
        self._init_widget_params()
        self.mt.mtlabel.max = tomo.n_splines - 1
        self.mt.pos.max = len(tomo.splines[0].anchors) - 1
        
        self._init_layers()
                        
        self.layer_work.mode = "pan_zoom"
        
        self._update_mtpath()
        
        return None
    
    def _init_widget_params(self):
        self.mt.mtlabel.value = 0
        self.mt.mtlabel.min = 0
        self.mt.mtlabel.max = 0
        self.mt.pos.value = 0
        self.mt.pos.min = 0
        self.mt.pos.max = 0
        self.Profiles.txt.value = ""
        return None
    
    def _init_figures(self):
        for i in range(3):
            del self.canvas[i].image
            self.canvas[i].layers.clear()
            self.canvas[i].text_overlay.text = ""
        for i in range(2):
            self.Profiles.plot[i].layers.clear()
        return None
    
    def _check_path(self) -> str:
        tomo = self.active_tomogram
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
            elif self.layer_work.data.shape[0] >= 3:
                point2, point1, point0 = self.layer_work.data[-3:]
                vec2 = point2 - point1
                vec0 = point0 - point1
                len0 = np.sqrt(vec0.dot(vec0))
                len2 = np.sqrt(vec2.dot(vec2))
                cos1 = vec0.dot(vec2)/(len0*len2)
                curvature = 2 * np.sqrt((1 - cos1**2) / sum((point2 - point0)**2))
                if curvature > 0.02:
                    # curvature is too large
                    return f"Curvature {curvature} is too large for a MT."
        
        return ""
    
    def _current_cartesian_img(self, i=None, j=None):
        """
        Return local Cartesian image at the current position
        """        
        i = i or self.mt.mtlabel.value
        j = j or self.mt.pos.value
        tomo = self.active_tomogram
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
        i = i or self.mt.mtlabel.value
        j = j or self.mt.pos.value
        tomo = self.active_tomogram
        ylen = tomo.nm2pixel(tomo.ft_size)
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
        
        common_properties = dict(ndim=3, n_dimensional=True, size=8)
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
        self.Profiles.orientation_choice.value = Ori.none
        return None
    
    @mt.pos.connect
    def _imshow_all(self):
        tomo = self.active_tomogram
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
        npaths = len(tomo.splines)
        if 0 == npaths:
            return
        if 0 < npaths <= i:
            i = 0
        spl = tomo.splines[i]
        
        if spl.localprops is not None:
            headers = [H.yPitch, H.skewAngle, H.nPF, H.start]
            pitch, skew, npf, start = spl.localprops[headers].iloc[j]
            self.Profiles.txt.value = f"{pitch:.2f} nm / {skew:.2f}°/ {int(npf)}_{start:.1f}"

        binsize = self.active_tomogram.metadata["binsize"]
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
        ylen = tomo.ft_size/2/binsize/tomo.scale
        ymin, ymax = ly/2 - ylen, ly/2 + ylen
        r_px = spl.radius/tomo.scale/binsize
        r = r_px*GVar.outer
        xmin, xmax = -r + lx/2, r + lx/2
        self.canvas[0].add_curve([xmin, xmin, xmax, xmax, xmin], 
                                 [ymin, ymax, ymax, ymin, ymin], color="lime")
    
        theta = np.linspace(0, 2*np.pi, 360)
        r = r_px * GVar.inner
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = r_px * GVar.outer
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
                
    
    @Profiles.orientation_choice.connect
    def _update_note(self):
        i = self.mt.mtlabel.value
        self.active_tomogram.splines[i].orientation = self.Profiles.orientation_choice.value
        return None
    
    @mt.mtlabel.connect
    def _update_mtpath(self):
        self.mt.mtlabel.enabled = False
        i = self.mt.mtlabel.value
        tomo = self.active_tomogram
        
        # calculate projection
        binsize = tomo.metadata["binsize"]
        imgb = self.layer_image.data
        
        spl = tomo.splines[i]
        spl.scale *= binsize
        
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        out = load_rot_subtomograms(imgb, length_px, width_px, spl)
        
        spl.scale /= binsize
        
        projections: list[Projections] = []
        for img, npf in zip(out, spl.localprops[H.nPF]):    
            proj = Projections(img)
            proj.rotational_average(npf)
            projections.append(proj)
        
        self.projections = projections
        
        self.mt.pos.max = len(tomo.splines[i].localprops) - 1
        note = tomo.splines[i].orientation
        self.Profiles.orientation_choice.value = Ori(note)
        self._plot_properties()
        self._imshow_all()
        self.mt.mtlabel.enabled = True
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
        for spl in self.active_tomogram.splines:
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

def _show_reconstruction(img: ip.ImgArray, name):
    viewer = napari.Viewer(title=name, axis_labels=("z", "y", "x"), ndisplay=3)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "nm"
    viewer.add_image(img, scale=img.scale, name=name)

def _iter_run(tomo: MtTomogram, 
              interval: nm,
              ft_size,
              n_refine,
              dense_mode,
              dense_mode_sigma,
              local_props,
              global_props):
    tomo.ft_size = ft_size
    tomo.fit(dense_mode=dense_mode, dense_mode_sigma=dense_mode_sigma)
    tomo.measure_radius()
    
    for i in range(n_refine):
        if n_refine == 1:
            yield "Spline refinement ..."
        else:
            yield f"Spline refinement (iteration {i+1}/{n_refine}) ..."
        tomo.refine(max_interval=max(interval, 30))
        tomo.measure_radius()
        
    tomo.make_anchors(interval=interval)
    if local_props:
        yield "Local Fourier transformation ..."
        tomo.local_ft_params()
    if global_props:
        yield "Local Fourier transformation ..."
        tomo.global_ft_params()
    yield "Finishing ..."
    return tomo
