from functools import wraps
import pandas as pd
from typing import Any, Callable, Iterable, NewType, Union, Tuple, List
import numpy as np
import warnings
import napari
from napari.utils import Colormap
from napari.qt import create_worker
from napari._qt.qthreading import GeneratorWorker, FunctionWorker
from napari.layers import Points, Image, Labels, Vectors
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
    build_help
    )
from magicclass.widgets import TupleEdit, Separator, ListWidget, Table, ColorEdit
from magicclass.ext.pyqtgraph import QtImageCanvas, QtMultiPlotCanvas, QtMultiImageCanvas
from magicclass.utils import show_messagebox, to_clipboard

from .molecules import Molecules
from .tomogram import MtSpline, MtTomogram, angle_corr, dask_affine, centroid
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
from .const import EulerAxes, Unit, nm, H, Ori, GVar, Sep, Order, Coordinates

# TODO: when anchor is updated (especially, "Fit splines manually" is clicked), spinbox and slider
# should also be initialized.

WORKING_LAYER_NAME = "Working Layer"
SELECTION_LAYER_NAME = "Selected MTs"
ICON_DIR = Path(__file__).parent / "icons"
MOLECULES = "Molecules"
SOURCE = "Source"

import magicgui
from magicgui.widgets._bases import CategoricalWidget
from napari.utils._magicgui import find_viewer_ancestor

MonomerLayer = NewType("MonomerLayer", Points)
Worker = Union[FunctionWorker, GeneratorWorker]

def get_monomer_layers(gui: CategoricalWidget) -> List[Points]:
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, Points) and MOLECULES in x.metadata]

magicgui.register_type(MonomerLayer, choices=get_monomer_layers)

from macrokit import register_type
register_type(np.ndarray, lambda arr: str(arr.tolist()))

def run_worker_function(worker: Worker):
    """Emulate worker execution."""
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
    """
    Open a progress bar and start worker in a parallel thread if function is called from GUI.
    Otherwise (if function is called from script), the worker will be executed as if the 
    function is directly called.
    """
    @wraps(f)
    def wrapper(self: "MTPropsWidget", *args, **kwargs):
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
    
    info = vfield(str, record=False)
    
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
            self.info = self._last_info
        else:
            self.worker.pause()
            self["Pause"].text = "Resume"
            self._last_info = self.info
            self.info = "Pausing"
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
                        name="Spline No.", record=False)
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
            cutoff = field(0.2, options={"min": 0.0, "max": 0.85, "step": 0.05, "tooltip": "Relative cutoff frequency of low-pass filter."}, record=False)
            def Average(self): ...
    
    def _get_shifts(self, _=None):
        i = self.mt.mtlabel.value
        return self.shifts[i]
    
    @mt.wraps
    def Fit(self, shifts: Bound[_get_shifts], i: Bound[mt.mtlabel]):
        """Fit current spline."""        
        shifts = np.asarray(shifts)
        spl = self.splines[i]
        sqsum = GVar.splError**2 * shifts.shape[0]
        spl.shift_fit(shifts=shifts*self.binsize, s=sqsum)
        spl.make_anchors(max_interval=self.max_interval)
        self.fit_done = True
        self._mt_changed()
        self.find_ancestor(MTPropsWidget)._update_splines_in_images()
    
    @Rotational_averaging.frame.wraps
    @do_not_record
    def Average(self):
        """Show rotatinal averaged image."""        
        i = self.mt.mtlabel.value
        j = self.mt.pos.value
                
        with no_verbose():
            img = self.find_ancestor(MTPropsWidget)._current_cartesian_img(i, j)
            cutoff = self.Rotational_averaging.frame.cutoff.value
            if 0 < cutoff < 0.866:
                img = img.lowpass_filter(cutoff=cutoff)
            proj = Projections(img)
            proj.rotational_average(self.Rotational_averaging.frame.nPF.value)
        self.Rotational_averaging.canvas_rot.image = proj.zx_ave
    
    
    def __post_init__(self):
        self.shifts: List[np.ndarray] = None
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
        
        tomo = self.find_ancestor(MTPropsWidget).active_tomogram
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
        tomo = self.find_ancestor(MTPropsWidget).active_tomogram
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
        parent = self.find_ancestor(MTPropsWidget)
        imgb = parent.layer_image.data
        tomo: MtTomogram = parent.active_tomogram
        
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


@magicmenu
class PEET(MagicTemplate):
    """PEET extension."""
    @set_options(mod_path={"label": "Path to MOD file", "mode": "r", "filter": "Model files (*.mod);;All files (*.txt;*.csv)"},
                 ang_path={"label": "Path to csv file", "mode": "r", "filter": "*.csv;*.txt"},
                 shift_mol={"label": "Apply shifts to monomers if offsets are available."})
    def Read_monomers(self, mod_path: Path, ang_path: Path, shift_mol: bool = True):
        """
        Read monomer coordinates and angles from PEET-format files.

        Parameters
        ----------
        mod_path : Path
            Path to the mod file that contains monomer coordinates.
        ang_path : Path
            Path to the text file that contains monomer angles in Euler angles.
        shift_mol : bool, default is True
            In PEET output csv there may be xOffset, yOffset, zOffset columns that can be directly applied to
            the molecule coordinates.
        """        
        from .ext.etomo import read_mod
        mod = read_mod(mod_path).values
        shifts, angs = _read_shift_and_angle(ang_path)
        mol = Molecules.from_euler(pos=mod*self.scale, angles=angs, degrees=True)
        if shift_mol:
            mol.translate(shifts*self.scale, copy=False)
        
        _add_molecules(self.parent_viewer, mol, "Molecules from PEET", source=None)
    
    @set_options(save_dir={"label": "Save at", "mode": "d"})
    def Save_monomers(self, 
                      save_dir: Path,
                      layer: MonomerLayer,
                      save_protofilaments_separately: bool = False):
        """
        Save monomer angles in PEET format.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        layer : Points
            Select the Vectors layer to save.
        save_protofilaments_separately : bool, default is False
            Check if you want to save monomers on each protofilament in separate files.
        """        
        save_dir = Path(save_dir)
        mol: Molecules = layer.metadata[MOLECULES]
        from .ext.etomo import save_mod, save_angles
        if save_protofilaments_separately:
            spl: MtSpline = layer.metadata[SOURCE]
            npf = roundint(spl.globalprops[H.nPF])
            
            for pf in range(npf):
                sl = slice(pf, None, npf)
                save_mod(save_dir/f"coordinates-PF{pf:0>2}.mod", mol.pos[sl, ::-1]/self.scale)
                save_angles(save_dir/f"angles-PF{pf:0>2}.csv", mol.euler_angle(EulerAxes.ZXZ, degrees=True)[sl])
        else:
            save_mod(save_dir/"coordinates.mod", mol.pos[:, ::-1]/self.scale)
            save_angles(save_dir/"angles.csv", mol.euler_angle(EulerAxes.ZXZ, degrees=True))
        return None
    
    @set_options(save_dir={"label": "Save at", "mode": "d"})
    def Save_all_monomers(self, 
                          save_dir: Path):
        """
        Save monomer angles in PEET format.

        Parameters
        ----------
        save_dir : Path
            Saving path.
        """        
        save_dir = Path(save_dir)
        layers = get_monomer_layers(self)
        if len(layers) == 0:
            raise ValueError("No monomer found.")
        mol = Molecules.concat([l.metadata[MOLECULES] for l in layers])
        from .ext.etomo import save_mod, save_angles
        save_mod(save_dir/"coordinates.mod", mol.pos[:, ::-1]/self.scale)
        save_angles(save_dir/"angles.csv", mol.euler_angle(EulerAxes.ZXZ, degrees=True))
        return None
    
    @set_options(ang_path={"label": "Path to csv file", "mode": "r", "filter": "*.csv;*.txt"})
    def Shift_monomers(self, ang_path: Path, layer: MonomerLayer, update: bool = False):
        """
        Shift monomer coordinates in PEET format.

        Parameters
        ----------
        ang_path : Path
            Path of offset file.
        layer : MonomerLayer
            Points layer of target monomers.
        update : bool, default is False
            Check if update monomer coordinates in place.
        """       
        mol: Molecules = layer.metadata[MOLECULES]
        shifts, angs = _read_shift_and_angle(ang_path)
        mol_shifted = mol.translate(shifts*self.scale)
        mol_shifted = Molecules.from_euler(pos=mol_shifted.pos, angles=angs, degrees=True)
        
        vector_data = np.stack([mol_shifted.pos, mol_shifted.z], axis=1)
        if update:
            layer.data = mol_shifted.pos
            vector_layer = None
            vector_layer_name = layer.name + " Z-axis"
            for l in self.parent_viewer.layers:
                if l.name == vector_layer_name:
                    vector_layer = l
                    break
            if vector_layer is not None:
                vector_layer.data = vector_data
            else:
                self.parent_viewer.add_vectors(
                    vector_data, edge_width=0.3, edge_color="crimson", length=2.4,
                    name=vector_layer_name,
                    )
            layer.metadata[MOLECULES] = mol_shifted
        else:
            _add_molecules(self.parent_viewer, mol_shifted, 
                           "Molecules from PEET", source=layer.metadata.get(SOURCE, None))
    
    @property
    def scale(self) -> float:
        return self.find_ancestor(MTPropsWidget).active_tomogram.scale

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
        def Measure_radius(self): ...
        def Local_FT_analysis(self): ...
        def Global_FT_analysis(self): ...
        sep0 = field(Separator)
        @magicmenu
        class Reconstruction(MagicTemplate):
            def Reconstruct_MT(self): ...
            def cylindric_reconstruction(self): ...
        def Map_monomers(self): ...
        def Map_monomers_manually(self): ...
        @magicmenu
        class Subtomogram_averaging(MagicTemplate):
            def Average_all(self): ...
            def Average_subset(self): ...
    
    @magicmenu
    class Others(MagicTemplate):
        """Other menus."""
        def Create_macro(self): ...
        def Global_variables(self): ...
        def Clear_cache(self): ...
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
        mtlabel = field(int, options={"max": 0, "tooltip": "Number of MT."}, name="Spline No.")
        pos = field(int, widget_type="Slider", options={"max": 0, "tooltip": "Position along a MT."}, name="Pos", record=False)
    
    canvas = field(QtMultiImageCanvas, name="Figure", options={"nrows": 1, "ncols": 3, "tooltip": "Projections"})
    
    @magicclass(widget_type="collapsible")
    class Profiles(MagicTemplate):
        """Local profiles."""
        txt = vfield(str, options={"enabled": False, "tooltip": "Structural parameters at current MT position."}, name="result")    
        orientation_choice = vfield(Ori.none, name="Orientation: ", options={"tooltip": "MT polarity."})
        plot = field(QtMultiPlotCanvas, name="Plot", options={"nrows": 2, "ncols": 1, "sharex": True, "tooltip": "Plot of local properties"})

    @magicclass(widget_type="tabbed")
    class Panels(MagicTemplate):
        """Panels for output."""
        overview = field(QtImageCanvas, name="Overview", options={"tooltip": "Overview of splines"})
        image2D = field(QtImageCanvas, options={"tooltip": "2-D image viewer."})
        table = field(Table, name="Table", options={"tooltip": "Result table"}, record=False)
    
    ### methods ###
        
    def __post_init__(self):
        self.active_tomogram: MtTomogram = None
        self._last_ft_size: nm = None
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
            self._last_ft_size = tomo.metadata.get("ft_size", None)
            self._connect_worker(worker)
            worker.start()
            
            if tomo.splines:
                worker.finished.connect(self.Sample_subtomograms)
            else:
                worker.finished.connect(self._init_layers)
                worker.finished.connect(self._init_widget_params)
                worker.finished.connect(self.Panels.overview.layers.clear)
        
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
    def register_path(self, coords: Bound[_get_spline_coordinates] = None):
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
        self.active_tomogram.clear_cache()
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
                         daskChunk: Tuple[int, int, int] = GVar.daskChunk):
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
    def Clear_cache(self):
        """Clear cache stored on the current tomogram."""
        if self.active_tomogram is not None:
            self.active_tomogram.clear_cache()
    
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
    def load_tomogram(self, 
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
    def Load_json(self, path: Path):
        """Choose a json file and load it."""        
        tomo = self.active_tomogram
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
        self.active_tomogram.save_json(save_path)
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
            arr = layer.data / self.active_tomogram.scale
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
                    
    @mt.mtlabel.connect
    @mt.pos.connect
    @Image.focus.connect
    def _focus_on(self):
        """Change camera focus to the position of current MT fragment."""
        if self.layer_paint is None:
            return None
        if not self.Image.focus.value:
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
    
    @Image.wraps
    def Sample_subtomograms(self):
        """Sample subtomograms at the anchor points on splines"""
        self._spline_fitter.close()
        tomo = self.active_tomogram
        spl = tomo.splines[0]
        ori = spl.orientation
        
        # initialize GUI
        self._init_widget_params()
        self._init_layers()
        self.layer_work.mode = "pan_zoom"
        self.mt.mtlabel.max = tomo.n_splines - 1
        
        if spl.localprops is not None:
            n_anc = len(spl.localprops)
        elif spl._anchors is not None:
            n_anc = len(spl._anchors)
        else:
            return
        
        self.mt.pos.max = n_anc - 1
        
        self.Profiles.orientation_choice = ori
        self._update_mtpath()
        return None
    
    @Image.wraps
    def Show_results_in_a_table_widget(self):
        """Show result table."""
        self.Panels.table.value = self.active_tomogram.collect_localprops()
        self.Panels.current_index = 2
        return None
    
    @Image.wraps
    @dispatch_worker
    def Show_straightened_image(self, i: Bound[mt.mtlabel]):
        """Send straightened image of the current MT to the viewer."""        
        tomo = self.active_tomogram
        
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
    def show_r_proj(self, i: Bound[mt.mtlabel], j: Bound[mt.pos]):
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
        i = self.mt.mtlabel.value
        with no_verbose():
            polar = self.active_tomogram.straighten_cylindric(i).proj("r")
        self.Panels.image2D.image = polar.value
        self.Panels.image2D.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        # move to center
        ly, lx = polar.shape
        self.Panels.image2D.xlim = [lx*0.3, lx*0.7]
        self.Panels.current_index = 1
        return None
    
    @Image.wraps
    @set_design(text="2D-FT")
    def show_current_ft(self, i: Bound[mt.mtlabel], j: Bound[mt.pos]):
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
    def show_global_ft(self, i: Bound[mt.mtlabel]):
        """View Fourier space along current MT."""  
        with no_verbose():
            polar = self.active_tomogram.straighten_cylindric(i)
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
        paths = [r.partition(100) for r in self.active_tomogram.splines]
        
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
        self.active_tomogram.align_to_polarity(orientation=orientation)
        self._update_splines_in_images()
        self._init_widget_params()
        self._init_figures()
        if need_resample:
            self.Sample_subtomograms()
        
    @Splines.wraps
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
        tomo = self.active_tomogram
        
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
        tomo = self.active_tomogram
        if tomo.splines[0].radius is None:
            self.Measure_radius()
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
        tomo = self.active_tomogram
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
        
    @Analysis.Reconstruction.wraps
    @set_options(rot_ave={"label": "Rotational averaging"},
                 find_seam={"label": "Find seam position"},
                 niter={"label": "Iteration", "max": 3},
                 y_length={"label": "Longitudinal length (nm)"})
    @dispatch_worker
    def Reconstruct_MT(self, i: Bound[mt.mtlabel], rot_ave=False, find_seam=False, niter=1, y_length=50.0):
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
        
        self._worker_control.info = f"Reconstruction ..."
        return worker
    
    @Analysis.Reconstruction.wraps
    @set_options(rot_ave={"label": "Rotational averaging"},
                 find_seam={"label": "Find seam position"},
                 niter={"label": "Iteration", "max": 3},
                 y_length={"label": "Longitudinal length (nm)"})
    @set_design(text="Reconstruct MT (cylindric)")
    @dispatch_worker
    def cylindric_reconstruction(self, i: Bound[mt.mtlabel], rot_ave=False, find_seam=False, niter=1, 
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
        
        worker = create_worker(tomo.reconstruct_cylindric, 
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
            
        self._worker_control.info = f"Cylindric reconstruction ..."
        return worker
    
    @Analysis.wraps
    @set_options(step={"min": 1, "max": 10})
    @dispatch_worker
    def Map_monomers(self, length: nm = 0.0, step: int = 1):
        """
        Map points to tubulin molecules using the results of global Fourier transformation.
        """
        tomo = self.active_tomogram
        worker = create_worker(tomo.map_monomers,
                               length=length,
                               _progress={"total": 0, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_return(out: List[Coordinates]):
            for i, coords in enumerate(out):
                spl = tomo.splines[i]
                mol = spl.cylindrical_to_molecules(coords.spline)
                if step > 1:
                    npf = roundint(spl.globalprops[H.nPF])
                    _mol_spec = ((np.arange(len(mol)) // npf) % step) == 0
                    mol = mol.subset(_mol_spec)
                _add_molecules(self.parent_viewer, mol, f"Monomers-{i}", source=spl)
                
        self._worker_control.info = "Monomer mapping ..."
        return worker

    @Analysis.wraps
    @set_options(auto_call=True, 
                 y_offset={"widget_type": "FloatSlider", "max": 5, "step": 0.1, "label": "y offset (nm)"},
                 theta_offset={"widget_type": "FloatSlider", "max": 180, "label": "θ offset (deg)"},
                 step={"min": 1, "max": 10})
    def Map_monomers_manually(
        self, 
        i: Bound[mt.mtlabel],
        y_offset: nm = 0, 
        theta_offset: float = 0,
        length: nm = 0.0, 
        step: int = 1
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
        """
        theta_offset = np.deg2rad(theta_offset)
        tomo = self.active_tomogram
        tomo.global_ft_params(i)
        coords = tomo.map_monomers(i, offsets=(y_offset, theta_offset), length=length)
        spl = tomo.splines[i]
        mol = spl.cylindrical_to_molecules(coords.spline)
        
        if step > 1:
            npf = roundint(spl.globalprops[H.nPF])
            _mol_spec = ((np.arange(len(mol)) // npf) % step) == 0
            mol = mol.subset(_mol_spec)
        
        viewer = self.parent_viewer
        layer_name = f"Monomers-{i}"
        if layer_name not in viewer.layers:
            points_layer = self.parent_viewer.add_points(
                ndim=3, size=3, face_color="lime", edge_color="lime",
                n_dimensional=True, name=layer_name, metadata={MOLECULES: mol}
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
    
    @Analysis.Subtomogram_averaging.wraps
    @set_options(
        shape={"widget_type": TupleEdit, "options": {"min": -20, "max": 20, "step": 0.01}, "label": "Subtomogram shape (nm)"},
        chunk_size={"min": 1, "max": 3600},
    )
    def Average_all(
        self,
        layer: MonomerLayer,
        shape: tuple[nm, nm, nm] = (18., 18., 18.),
        chunk_size: int = 546,
    ):
        """
        Subtomogram averaging using all the subvolumes.

        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        shape : tuple[nm, nm, nm], default is (18., 18., 18.)
            Shape of subtomograms.
        chunk_size : int, default is 546
            How many subtomograms will be loaded at the same time.
        """
        molecules = layer.metadata[MOLECULES]
        loader = self.active_tomogram.get_subtomogram_loader(molecules, shape, chunksize=chunk_size)
        ave = loader.average()
        if self.active_tomogram.light_background:
            ave = -ave
        _show_reconstruction(ave, f"Subtomogram average (n={len(molecules)})")
        return None
    
    @Analysis.Subtomogram_averaging.wraps
    @set_options(
        shape={"widget_type": TupleEdit, "options": {"min": -20, "max": 20, "step": 0.01}, "label": "Subtomogram shape (nm)"},
        method={"choices": ["steps", "first", "last", "random"]},
    )
    def Average_subset(
        self,
        layer: MonomerLayer,
        shape: tuple[nm, nm, nm] = (18., 18., 18.),
        method="steps", 
        number: int = 64
    ):
        """
        Subtomogram averaging using a subset of subvolumes.

        Parameters
        ----------
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        shape : tuple[nm, nm, nm], default is (18., 18., 18.)
            Shape of subtomograms.
        method : str, optional
            How to choose subtomogram subset. 
            (1) steps: Each 'steps' subtomograms from the tip of spline. 
            (2) first: First subtomograms.
            (3) last: Last subtomograms.
            (4) random: choose randomly.
        number : int, default is 64
            Number of subtomograms to use.
            
        """
        molecules = layer.metadata[MOLECULES]
        nmole = len(molecules)
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
            sl = np.random.shuffle(sl_all)[:number]
        else:
            raise NotImplementedError(method)
        mole = molecules.subset(sl)
        loader = self.active_tomogram.get_subtomogram_loader(mole, shape)
        ave = loader.average()
        if self.active_tomogram.light_background:
            ave = -ave
        _show_reconstruction(ave, f"Subtomogram average (n={len(mole)})")
        return None
        
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
        tomo = self.active_tomogram
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
    @set_options(start={"widget_type": ColorEdit},
                 end={"widget_type": ColorEdit},
                 limit={"widget_type": TupleEdit, "options": {"min": -20, "max": 20, "step": 0.01}, "label": "limit (nm)"},
                 color_by={"choices": [H.yPitch, H.skewAngle, H.nPF, H.riseAngle]},
                 auto_call=True)
    def Set_colormap(self,
                     start=(0, 0, 1, 1), 
                     end=(1, 0, 0, 1), 
                     limit=(4.00, 4.24), 
                     color_by: str = H.yPitch):
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
                if self._last_ft_size is not None:
                    tomo.metadata["ft_size"] = self._last_ft_size
                
                tomo._set_image(img)
                self.active_tomogram = tomo
                self.Tomogram_List.tomograms.append(tomo)
                
                self.clear_all()
            
            return None
        
        return worker
    
    def _init_widget_params(self):
        self.mt.mtlabel.value = 0
        self.mt.mtlabel.min = 0
        self.mt.mtlabel.max = 0
        self.mt.pos.value = 0
        self.mt.pos.min = 0
        self.mt.pos.max = 0
        self.Profiles.txt = ""
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
        self.Profiles.orientation_choice = Ori.none
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
            self.Profiles.txt = f"{pitch:.2f} nm / {skew:.2f}°/ {int(npf)}_{start:.1f}"

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
        
        if self._last_ft_size is None:
            ylen = 25/binsize/tomo.scale
        else:
            ylen = self._last_ft_size/2/binsize/tomo.scale
        
        # draw a square in YX-view
        ymin, ymax = ly/2 - ylen - 0.5, ly/2 + ylen + 0.5
        r_px = spl.radius/tomo.scale/binsize
        r = r_px*GVar.outer
        xmin, xmax = -r + lx/2 - 0.5, r + lx/2 + 0.5
        self.canvas[0].add_curve([xmin, xmin, xmax, xmax, xmin], 
                                 [ymin, ymax, ymax, ymin, ymin], color="lime")

        # draw two circles in ZX-view
        theta = np.linspace(0, 2*np.pi, 360)
        r = r_px * GVar.inner
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = r_px * GVar.outer
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
                
    
    @Profiles.orientation_choice.connect
    def _update_note(self):
        i = self.mt.mtlabel.value
        self.active_tomogram.splines[i].orientation = self.Profiles.orientation_choice
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
        
        # Rotational average should be calculated using local nPF if possible.
        # If not available, use global nPF
        projections: List[Projections] = []
        if spl.localprops is not None:
            npf_list = spl.localprops[H.nPF]
        elif spl.globalprops is not None:
            npf_list = [spl.globalprops[H.nPF]] * tomo.splines[i].anchors.size
        else:
            return None
        
        for img, npf in zip(out, npf_list):    
            proj = Projections(img)
            proj.rotational_average(npf)
            projections.append(proj)
        
        self.projections = projections
        
        self.mt.pos.max = tomo.splines[i].anchors.size - 1
        self.Profiles.orientation_choice = Ori(tomo.splines[i].orientation)
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
    with no_verbose():
        viewer.add_image(img.rescale_intensity(), scale=img.scale, name=name)

def _iter_run(tomo: MtTomogram, 
              interval: nm,
              ft_size,
              n_refine,
              dense_mode,
              dense_mode_sigma,
              local_props,
              global_props):
    
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
        tomo.local_ft_params(ft_size=ft_size)
    if global_props:
        yield "Global Fourier transformation ..."
        tomo.global_ft_params()
    yield "Finishing ..."
    return tomo

def _add_molecules(viewer: "napari.Viewer", mol: Molecules, name, source: MtSpline = None):
    metadata ={MOLECULES: mol}
    if source is not None:
        metadata.update({SOURCE: source})
    points_layer = viewer.add_points(
        mol.pos, size=3, face_color="lime", edge_color="lime",
        n_dimensional=True, name=name, metadata=metadata
        )
    
    points_layer.shading = "spherical"
    
    vector_data = np.stack([mol.pos, mol.z], axis=1)
    viewer.add_vectors(
        vector_data, edge_width=0.3, edge_color="crimson", length=2.4,
        name=name + " Z-axis",
        )


def _read_angle(ang_path: str) -> np.ndarray:
    line1 = str(pd.read_csv(ang_path, nrows=1).values[0, 0])  # determine sep
    if "\t" in line1:
        sep = "\t"
    else:
        sep = ","
    
    csv = pd.read_csv(ang_path, sep=sep)
    
    if csv.shape[1] == 3:
        try:
            header = np.array(csv.columns).astype(np.float64)
            csv_data = np.concatenate([header.reshape(1, 3), csv.values], axis=0)
        except ValueError:
            csv_data = csv.values
    elif "CCC" in csv.columns:
        csv_data = -csv[["EulerZ(1)", "EulerX(2)", "EulerZ(3)"]].values
    else:
        raise ValueError(f"Could not interpret data format of {ang_path}:\n{csv.head(5)}")
    return csv_data


def _read_shift_and_angle(path: str) -> Tuple[Union[np.ndarray, None], np.ndarray]:
    """Read offsets and angles from PEET project"""
    csv: pd.DataFrame = pd.read_csv(path)
    if "CCC" in csv.columns:
        ang_data = -csv[["EulerZ(1)", "EulerX(2)", "EulerZ(3)"]].values
        shifts_data = csv[["zOffset", "yOffset", "xOffset"]].values
    else:
        ang_data = _read_angle(path)
        shifts_data = None
    return shifts_data, ang_data