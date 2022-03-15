import os
import re
import json
from typing import Iterable, Iterator, Union, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
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
    MagicTemplate,
    bind_key,
    build_help,
    get_function_gui,
    confirm,
    nogui
    )
from magicclass.types import Color, Bound, Optional, Tuple as _Tuple
from magicclass.widgets import (
    Logger,
    Separator,
    RadioButtons,
    ConsoleTextEdit,
    Select,
    FloatRangeSlider,
    )
from magicclass.ext.pyqtgraph import QtImageCanvas

from ..components import SubtomogramLoader, Molecules, MtSpline, MtTomogram
from ..components.microtubule import angle_corr
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
from ..const import nm, H, Ori, GVar, Mole
from ..const import WORKING_LAYER_NAME, SELECTION_LAYER_NAME, ALN_SUFFIX, MOLECULES
from ..types import MonomerLayer

from .global_variables import GlobalVariables
from .properties import GlobalPropertiesWidget, LocalPropertiesWidget
from .spline_control import SplineControl
from .spline_fitter import SplineFitter
from .tomogram_list import TomogramList
from .feature_control import FeatureControl
from .image_processor import ImageProcessor
from .worker import WorkerControl, dispatch_worker, Worker
from .widget_utils import (
    add_molecules,
    change_viewer_focus,
    update_features,
    molecules_to_spline,
    y_coords_to_start_number,
)
from ..ext.etomo import PEET

ICON_DIR = Path(__file__).parent / "icons"
SPLINE_ID = "spline-id"
MASK_CHOICES = ("No mask", "Use blurred template as a mask", "Supply a image")

### The main widget ###
    
@magicclass(widget_type="scrollable", name="MTProps widget")
class MTPropsWidget(MagicTemplate):
    # Main GUI class.
    
    ### widgets ###
    
    _WorkerControl = field(WorkerControl, name="Worker control")
    _SplineFitter = field(SplineFitter, name="Spline fitter")
    _TomogramList = field(TomogramList, name="Tomogram list")
    _FeatureControl = field(FeatureControl, name="Feature Control")
    
    @magicmenu
    class File(MagicTemplate):
        """File I/O."""  
        def Open_image(self): ...
        def Open_tomogram_list(self): ...
        def Load_molecules(self): ...
        sep0 = field(Separator)
        def Load_project(self): ...
        def Save_project(self): ...
        def Save_molecules(self): ...
        sep1 = field(Separator)
        Process_image_files = ImageProcessor
        PEET = PEET

    @magicmenu
    class Image(MagicTemplate):
        """Image processing and visualization"""
        def Show_image_info(self): ...
        def Filter_reference_image(self): ...
        def Invert_tomogram(self): ...
        def Add_multiscale(self): ...
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
        def Molecules_to_spline(self): ...

    @magicmenu
    class Molecules_(MagicTemplate):
        @magicmenu
        class Mapping(MagicTemplate):
            def Map_monomers(self): ...
            def Map_monomers_manually(self): ...
            def Map_centers(self): ...
            def Map_along_PF(self): ...
        def Show_orientation(self): ...
        def Calculate_intervals(self): ...
        def Open_feature_control(self): ...
        sep0 = field(Separator)
        def Split(self): ...
        
    @magicmenu
    class Analysis(MagicTemplate):
        """Analysis of tomograms."""
        def Set_radius(self): ...
        def Local_FT_analysis(self): ...
        def Global_FT_analysis(self): ...
        sep0 = field(Separator)
        def Open_subtomogram_analyzer(self): ...
    
    @magicmenu
    class Others(MagicTemplate):
        """Other menus."""
        def Create_macro(self): ...
        Global_variables = GlobalVariables
        def Clear_cache(self): ...
        def Open_help(self): ...
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
            angle_deviation = vfield(12.0, widget_type="FloatSlider", options={"min": 1.0, "max": 40.0, "step": 0.5, "tooltip": "Angle deviation (degree) of auto picker"}, record=False)
            angle_precision = vfield(1.0, widget_type="FloatSlider", options={"min": 0.5, "max": 5.0, "step": 0.1, "tooltip": "Angle precision (degree) of auto picker"}, record=False)
        sep1 = field(Separator)
        def clear_current(self): ...
        def clear_all(self): ...
    
    SplineControl = SplineControl
    LocalProperties = field(LocalPropertiesWidget, name="Local Properties")
    GlobalProperties = field(GlobalPropertiesWidget, name="Global Properties")
    
    @magicclass(widget_type="tabbed", labels=False)
    class Panels(MagicTemplate):
        """Panels for output."""
        overview = field(QtImageCanvas, name="Overview", options={"tooltip": "Overview of splines"})
        image2D = field(QtImageCanvas, options={"tooltip": "2-D image viewer."})
        log = field(Logger, name="Log")
    
    ### methods ###
    
    def __init__(self):
        self.tomogram: MtTomogram = None
        self._current_ft_size: nm = None
        self.layer_image: Image = None
        self.layer_prof: Points = None
        self.layer_work: Points = None
        self.layer_paint: Labels = None
        self.objectName()  # load napari types
        
    def __post_init__(self):
        self.Set_colormap()
        self.min_width = 400
        self.LocalProperties.collapsed = False
        self.GlobalProperties.collapsed = False
        self.Panels.min_height = 300
        
        mgui = get_function_gui(self, "Open_image")
        @mgui.path.changed.connect
        def _read_scale():
            img = ip.lazy_imread(mgui.path.value, chunks=GVar.daskChunk)
            scale = img.scale.x
            mgui.scale.value = f"{scale:.4f}"
            mgui.bin_size.value = ceilint(0.96 / scale)

    @property
    def sub_viewer(self) -> napari.Viewer:
        """Return the sub-viewer that is used for subtomogram averaging."""
        return self._subtomogram_averaging._viewer

    def _get_splines(self, widget=None) -> List[Tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        tomo = self.tomogram
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]
        
    def _get_spline_coordinates(self, widget=None) -> np.ndarray:
        """Get coordinates of the manually picked spline."""
        coords = self.layer_work.data
        return np.round(coords, 3)
    
    def _get_available_binsize(self, _=None) -> List[int]:
        if self.tomogram is None:
            return [1]
        out = [x[0] for x in self.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return out
    
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
            return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]
        
        def _get_available_binsize(self, _=None) -> List[int]:
            try:
                parent = self.find_ancestor(MTPropsWidget)
            except Exception:
                return [1]
            if parent.tomogram is None:
                return [1]
            out = [x[0] for x in parent.tomogram.multiscaled]
            if 1 not in out:
                out = [1] + out
            return out
        
        all_splines = vfield(True, options={"text": "Run for all the splines.", "tooltip": "Uncheck to select along which spline algorithms will be executed."}, record=False)
        splines = vfield(Select, options={"choices": _get_splines, "visible": False}, record=False)
        bin_size = vfield(1, options={"choices": _get_available_binsize, "tooltip": "Set to >1 to use binned image for fitting."}, record=False)
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
        self._runner.show(run=False)
        return None
    
    @_runner.wraps
    @set_design(text="Run")
    @dispatch_worker
    def run_mtprops(
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
        """Run MTProps"""
        self._runner.close()
        
        if self.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if self.tomogram.n_splines == 0:
            raise ValueError("No spline found.")
        elif len(splines) == 0:
            splines = range(self.tomogram.n_splines)
        
        total = (1 + n_refine + int(local_props) + int(global_props)) * len(splines)
        
        worker = create_worker(
            _iter_run, 
            tomo=self.tomogram,
            splines=splines,
            bin_size=bin_size,
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
            with self.macro.blocked():
                self._update_splines_in_images()
                if local_props or global_props:
                    self.Sample_subtomograms()
                    if global_props:
                        df = self.tomogram.collect_globalprops(i=splines).transpose()
                        df.columns = [f"Spline-{i}" for i in splines]
                        self.Panels.log.print_table(df, precision=3)
                if local_props and paint:
                    self.Paint_MT()
                tomo.metadata["ft_size"] = self._current_ft_size
                if global_props:
                    self._update_global_properties_in_widget()
        self._current_ft_size = ft_size
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
    @set_design(icon_path=ICON_DIR/"clear_all.png")
    @confirm("Are you sure to clear all?\nYou cannot undo this.")
    def clear_all(self):
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
        import macrokit as mk
        v = mk.Expr("getattr", [mk.symbol(self), "parent_viewer"])
        new = self.macro.widget.new()
        new.value = str(self.macro.format([(mk.symbol(self.parent_viewer), v)]))
        new.show()
        return None
        
    @Others.wraps
    @confirm("Are you sure to clear cache?\nYou cannot undo this.")
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
    
    def _send_tomogram_to_viewer(self, tomo: MtTomogram):
        viewer = self.parent_viewer
        self.tomogram = tomo
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
        with ip.silent():
            proj = imgb.proj("z")
        self.Panels.overview.image = proj
        self.Panels.overview.ylim = (0, proj.shape[0])
        
        try:
            parts = tomo.source.parts
            _name = os.path.join(*parts[-2:])
        except Exception:
            _name = f"Tomogram<{hex(id(tomo))}>"
        self.Panels.log.print_html(f"<h2>{_name}</h2>")
        return None

    @File.wraps
    @set_options(
        path={"filter": "*.mrc;*.rec;*.tif;*.tiff;*.map;;*.*"},
        scale={"min": 0.001, "step": 0.0001, "max": 10.0, "label": "scale (nm)"},
        bin_size={"min": 1, "max": 8}
    )
    @dispatch_worker
    def Open_image(
        self, 
        path: Path,
        scale: float = 1.0,
        bin_size: int = 1,
    ):
        """
        Load an image file and process it before sending it to the viewer.

        Parameters
        ----------
        path : Path
            Path to the tomogram. Must be 3-D image.
        scale : float, defaul is 1.0
            Pixel size in nm/pixel unit.
        bin_size : int, default is 1
            Initial bin size of image. Binned image will be used for visualization in the viewer.
            You can use both binned and non-binned image for analysis.
        """
        img = ip.lazy_imread(path, chunks=GVar.daskChunk)
        if scale is not None:
            scale = float(scale)
            img.scale.x = img.scale.y = img.scale.z = scale
        else:
            scale = img.scale.x
        
        worker = create_worker(
            self._imread,
            path=path,
            scale=scale,
            binsize=bin_size,
            _progress={"total": 0, "desc": "Running"})

        self._WorkerControl.info = f"Reading image (bin size = {bin_size})"
        
        @worker.returned.connect
        def _on_return(tomo: MtTomogram):
            self._send_tomogram_to_viewer(tomo)
            if self._current_ft_size is not None:
                tomo.metadata["ft_size"] = self._current_ft_size
            tomo_list_widget = self._TomogramList
            tomo_list_widget._tomogram_list.append(tomo)
            tomo_list_widget.reset_choices()  # Next line of code needs updated choices
            try:
                tomo_list_widget.tomograms.value = len(tomo_list_widget._tomogram_list) - 1
            except ValueError:
                pass
            with self.macro.blocked():
                self.clear_all()
    
        return worker
    
    
    @File.wraps
    @set_options(path={"filter": "*.json;*.txt"})
    @dispatch_worker
    def Load_project(self, path: Path):
        """Load a project json file."""
        path = str(path)
    
        with open(path, mode="r") as f:
            js: dict = json.load(f)
        
        # load image and multiscales
        multiscales: List[int] = js["multiscales"]
        binsize = multiscales.pop(-1)
        
        worker = create_worker(
            self._imread,
            path=js["image"], 
            scale=js["scale"], 
            binsize=binsize, 
            _progress={"total": 0, "desc": "Running"}
        )

        self._WorkerControl.info = f"Reading project {path!r}"
        
        @worker.returned.connect
        def _on_return(tomo: MtTomogram):
            self._send_tomogram_to_viewer(tomo)
            if self._current_ft_size is not None:
                tomo.metadata["ft_size"] = self._current_ft_size
            tomo_list_widget = self._TomogramList
            tomo_list_widget._tomogram_list.append(tomo)
            tomo_list_widget.reset_choices()  # Next line of code needs updated choices
            try:
                tomo_list_widget.tomograms.value = len(tomo_list_widget._tomogram_list) - 1
            except ValueError:
                pass
            with self.macro.blocked():
                self.clear_all()
    
            for size in multiscales:
                self.tomogram.add_multiscale(size)
            
            self._current_ft_size = js["current_ft_size"]
            
            # load splines
            splines = [MtSpline.from_dict(d) for d in js["splines"]]
            localprops = dict(iter(pd.read_csv(js["localprops"]).groupby("SplineID")))
            globalprops = dict(pd.read_csv(js["globalprops"]).iterrows())
            
            for i, spl in enumerate(splines):
                spl.localprops = localprops.get(i, None)
                if spl.localprops is not None:
                    spl._anchors = np.asarray(spl.localprops.get(H.splPosition))
                    spl.localprops.pop("SplineID")
                    spl.localprops.pop("PosID")
                    spl.localprops.index = range(len(spl.localprops))
                spl.globalprops = globalprops.get(i, None)
                if spl.globalprops is not None:
                    try:
                        spl.radius = spl.globalprops.pop("radius")
                    except KeyError:
                        pass
                    try:
                        spl.orientation = spl.globalprops.pop("orientation")
                    except KeyError:
                        pass
            
            if splines:
                self.tomogram._splines = splines
                self._update_splines_in_images()
                self.Sample_subtomograms()
            
            # load molecules
            from scipy.spatial.transform import Rotation
            for path in js["molecules"]:
                df = pd.read_csv(path)
                features = df.iloc[:, 6:]
                mole = Molecules(df.values[:, :3], Rotation.from_rotvec(df.values[:, 3:6]))
                layer = add_molecules(self.parent_viewer, mole, name=Path(path).stem)
                layer.features = features
            
            # load subtomogram analyzer state
            self._subtomogram_averaging.template_path = js.get("template-image", "")
            self._subtomogram_averaging._set_mask_params(js.get("mask-parameters", None))
            self.reset_choices()
        
        return worker
    
    @File.wraps
    @set_options(
        json_path={"mode": "w", "filter": "*.json;*.txt"},
        results_dir={"text": "Save at the same directory", "options": {"mode": "d"}}
    )
    def Save_project(self, json_path: Path, results_dir: Optional[Path] = None):
        """
        Save current project state as a json file and the results in a directory.
        
        The json file contains paths of images and results, parameters of splines,
        scales and version. Local and global properties, molecule coordinates and
        features will be exported as csv files.

        Parameters
        ----------
        json_path : Path
            Path of json file.
        results_dir : Path, optional
            Optionally you can specify the directory to save csv files.
        """
        from .. import __version__
        tomo = self.tomogram
        localprops = tomo.collect_localprops()    
        globalprops = tomo.collect_globalprops()
        
        _json_path = Path(json_path)
        if results_dir is None:
            results_dir = _json_path.parent / (_json_path.stem + "_results")
        localprops_path = None if localprops is None else results_dir / "localprops.csv"
        globalprops_path = None if globalprops is None else results_dir / "globalprops.csv"
        
        molecule_dataframes: List[pd.DataFrame] = []
        molecules_paths = []
        for layer in filter(
            lambda x: isinstance(x, Points) and MOLECULES in x.metadata,
            self.parent_viewer.layers
        ):
            layer: Points
            mole: Molecules = layer.metadata[MOLECULES]
            features = layer.features
            molecule_dataframes.append(pd.concat([mole.to_dataframe(), features], axis=1))
            molecules_paths.append((results_dir/layer.name).with_suffix(".csv"))
            
        js = {
            "version": __version__,
            "image": tomo.source,
            "scale": tomo.scale,
            "multiscales": [x[0] for x in tomo.multiscaled],
            "current_ft_size": self._current_ft_size,
            "splines": [spl.to_dict() for spl in tomo.splines],
            "localprops": localprops_path,
            "globalprops": globalprops_path,
            "molecules": molecules_paths,
            "template-image": self._subtomogram_averaging.template_path,
            "mask-parameters": self._subtomogram_averaging._get_mask_params(),
        }
        
        with open(_json_path, mode="w") as f:
            json.dump(js, f, indent=4, separators=(",", ": "), default=json_encoder)
        
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        if localprops_path:
            localprops.to_csv(localprops_path)
        if globalprops_path:
            globalprops.to_csv(globalprops_path)
        if molecules_paths:
            for df, fp in zip(molecule_dataframes, molecules_paths):
                df.to_csv(fp, index=False)
        return
    
    @File.wraps
    @do_not_record
    def Open_tomogram_list(self):
        """Open the list of loaded tomogram references."""
        self._TomogramList.show()
        return None
    
    @File.wraps
    @set_options(paths={"filter": "*.csv;*.txt"})
    def Load_molecules(self, paths: List[Path]):
        """Load molecules from a csv file."""
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        for path in paths:
            df: pd.DataFrame = pd.read_csv(path)
            if df.shape[1] < 6:
                raise ValueError(f"CSV must have more than or equal six columns but got shape {df.shape}")
            from scipy.spatial.transform import Rotation
            mole = Molecules(df.values[:, :3], Rotation.from_rotvec(df.values[:, 3:6]))
            name = Path(path).stem
            points = add_molecules(self.parent_viewer, mole, name)
            if df.shape[1] > 6:
                points.features = df.iloc[:, 6:]
        return None
    
    @File.wraps
    @set_options(save_path={"mode": "w", "filter": "*.txt;*.csv;*.dat"})
    def Save_molecules(
        self,
        layer: MonomerLayer, 
        save_path: Path,
        save_features: bool = True,
    ):
        """
        Save monomer coordinates.

        Parameters
        ----------
        layer : Points
            Select the points layer to save.
        save_path : Path
            Where to save the molecules.
        save_features : bool, default is True
            Check if save molecule features.
        """
        mole: Molecules = layer.metadata[MOLECULES]
        if save_features:
            props = layer.features
        else:
            props = None
        mole.to_csv(save_path, properties=props)
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
    def Filter_reference_image(self):
        """Apply low-pass filter to enhance contrast of the reference image."""
        cutoff = 0.2
        def func():
            with ip.silent(), set_gpu():
                img: ip.ImgArray = self.layer_image.data
                overlap = [min(s, 32) for s in img.shape]
                self.layer_image.data = img.tiled_lowpass_filter(
                    cutoff, chunks=(96, 96, 96), overlap=overlap,
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
        worker = create_worker(tomo.invert, _progress={"total": 0, "desc": "Running"})
        self._WorkerControl.info = "Inverting tomogram"
        
        @worker.returned.connect
        def _on_return(tomo_inv: MtTomogram):
            imgb_inv = tomo_inv.multiscaled[-1]
            self.layer_image.data = imgb_inv
            vmin, vmax = self.layer_image.contrast_limits
            clims = [-vmax, -vmin]
            self.layer_image.contrast_limits = clims
            self.Panels.overview.image = -self.Panels.overview.image
            self.Panels.overview.contrast_limits = clims
        
        return worker
    
    @Image.wraps
    @set_options(bin_size={"min": 2, "max": 64})
    @dispatch_worker
    def Add_multiscale(self, bin_size: int = 2, update_layer: bool = False):
        tomo = self.tomogram        
        worker = create_worker(
            tomo.get_multiscale,
            binsize=bin_size,
            add=True,
            _progress={"total": 0, "desc": "Running"}
        )
        self._WorkerControl.info = f"Adding multiscale (bin size = {bin_size})"
        
        @worker.returned.connect
        def _on_return(imgb: ip.ImgArray):
            if not update_layer:
                return
            self.layer_image.data = imgb
            self.layer_image.scale = imgb.scale
            self.layer_image.name = imgb.name + f"(bin {bin_size})"
            self.layer_image.translate = [tomo.multiscale_translation(bin_size)] * 3
            self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]
            with ip.silent():
                self.Panels.overview.image = imgb.proj("z")
        
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
            polar: ip.ImgArray = self.tomogram.straighten_cylindric(i)
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
        return None
    
    @Splines.wraps
    @set_options(
        auto_call=True,
        spline={"choices": _get_splines},
        limits = {"min": 0.0, "max": 1.0, "widget_type": FloatRangeSlider},
    )
    def Clip_spline(self, spline: int, limits: Tuple[float, float] = (0., 1.)):
        # BUG: properties may be inherited in a wrong way
        if spline is None:
            return
        start, stop = limits
        spl = self.tomogram.splines[spline]
        self.tomogram.splines[spline] = spl.restore().clip(start, stop)
        self._update_splines_in_images()
        return None
        
    @Splines.wraps
    @set_options(
        max_interval={"label": "Max interval (nm)"},
        bin_size={"choices": _get_available_binsize},
        edge_sigma={"text": "Do not mask image"},
    )
    @dispatch_worker
    def Fit_splines(
        self, 
        max_interval: nm = 30,
        bin_size: int = 1,
        degree_precision: float = 0.5,
        edge_sigma: Optional[nm] = 2.0,
        max_shift: nm = 5.0,
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
            binsize=bin_size,
            degree_precision=degree_precision,
            edge_sigma=edge_sigma,
            max_shift=max_shift,
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
        worker = create_worker(
            self.tomogram.set_radius,
            radius=radius,
            _progress={"total": 0, "desc": "Running"}
        )
        
        self._WorkerControl.info = "Measuring Radius"

        return worker
    
    @Splines.wraps
    @set_options(
        max_interval={"label": "Maximum interval (nm)"},
        corr_allowed={"label": "Correlation allowed", "max": 1.0, "step": 0.1},
        bin_size={"choices": _get_available_binsize},
    )
    @dispatch_worker
    def Refine_splines(self, max_interval: nm = 30, corr_allowed: float = 0.9, bin_size: int = 1):
        """
        Refine splines using the global MT structural parameters.
        
        Parameters
        ----------
        max_interval : nm, default is 30
            Maximum interval between anchors.
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        """
        tomo = self.tomogram
        
        worker = create_worker(
            tomo.refine,
            max_interval=max_interval,
            corr_allowed=corr_allowed,
            binsize = bin_size,
            _progress={"total": 0, "desc": "Running"}
        )
        
        worker.finished.connect(self._update_splines_in_images)

        self._WorkerControl.info = "Refining splines ..."
        
        self._init_widget_state()
        return worker
    
    @Splines.wraps
    def Molecules_to_spline(
        self, 
        layers: List[MonomerLayer],
        update_splines: bool = False,
    ):
        splines: List[MtSpline] = []
        for layer in layers:
            mole: Molecules = layer.metadata[MOLECULES]
            spl = MtSpline(degree=GVar.splOrder)
            npf = roundint(np.max(layer.features[Mole.pf]) + 1)
            all_coords = mole.pos.reshape(-1, npf, 3)
            mean_coords = np.mean(all_coords, axis=1)
            spl.fit(mean_coords, variance=GVar.splError**2)
            splines.append(spl)
        
        return None
        
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
        worker = create_worker(
            tomo.local_ft_params,
            ft_size=ft_size,
            _progress={"total": 0, "desc": "Running"}
        )
        @worker.returned.connect
        def _on_return(df):
            with self.macro.blocked():
                self.Sample_subtomograms()
                self._update_local_properties_in_widget()
        self._current_ft_size = ft_size
        self._WorkerControl.info = "Local Fourier transform ..."
        return worker
        
    @Analysis.wraps
    @dispatch_worker
    def Global_FT_analysis(self):
        """Determine MT global structural parameters by Fourier transformation."""        
        tomo = self.tomogram
        worker = create_worker(
            tomo.global_ft_params,
            _progress={"total": 0, "desc": "Running"}
        )
        worker.returned.connect(lambda e: self._update_global_properties_in_widget())
        
        self._WorkerControl.info = f"Global Fourier transform ..."
        
        return worker
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Molecules_.Mapping.wraps
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
        worker = create_worker(
            tomo.map_monomers,
            i=splines,
            length=length,
            _progress={"total": 0, "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_return(out: List[Molecules]):
            self.Panels.log.print_html("<code>Map_monomers</code>")
            for i, mol in enumerate(out):
                _name = f"Mono-{i}"
                spl = tomo.splines[splines[i]]
                layer = add_molecules(self.parent_viewer, mol, _name)
                npf = roundint(spl.globalprops[H.nPF])
                if len(mol) % npf != 0:
                    # just in case.
                    raise RuntimeError(
                        "Unexpected mismatch between number of molecules and protofilaments! "
                        "These molecules may not work in some analysis."
                    )
                update_features(layer, Mole.pf, np.arange(len(mol), dtype=np.uint8) % npf)
                self.Panels.log.print(f"{_name!r}: n = {len(mol)}")
                
        self._WorkerControl.info = "Monomer mapping ..."
        return worker

    @Molecules_.Mapping.wraps
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
        labels = np.arange(len(mol), dtype=np.uint8) % npf
        if layer_name not in viewer.layers:
            
            points_layer = self.parent_viewer.add_points(
                ndim=3, size=3, face_color="lime", edge_color="lime",
                out_of_slice_display=True, name=layer_name, 
                metadata={MOLECULES: mol, Mole.pf: labels}
                )
            
            points_layer.shading = "spherical"
        
        else:
            points_layer: Points = viewer.layers[layer_name]
            points_layer.data = mol.pos
            points_layer.selected_data = set()
            points_layer.metadata[MOLECULES] = mol
            update_features(points_layer, Mole.pf, labels)
        
    @Molecules_.Mapping.wraps
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
        self.Panels.log.print_html("<code>Map_centers</code>")
        for i, mol in enumerate(mols):
            _name = f"Center-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.Panels.log.print(f"{_name!r}: n = {len(mol)}")

    @Molecules_.Mapping.wraps
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
        self.Panels.log.print_html("<code>Map_along_PF</code>")
        for i, mol in enumerate(mols):
            _name = f"PF line-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.Panels.log.print(f"{_name!r}: n = {len(mol)}")

    @Molecules_.wraps
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
        
    @Molecules_.wraps
    @set_options(
        filter_length={"min": 1, "max": 49, "step": 2},
        filter_width={"min": 1, "max": 15, "step": 2},
        spline_precision={"max": 2.0, "step": 0.01, "label": "spline precision (nm)"}
    )
    def Calculate_intervals(
        self,
        layer: MonomerLayer,
        filter_length: int = 1,
        filter_width: int = 1,
        spline_precision: nm = 0.2,
    ):
        """
        Calculate intervals between adjucent molecules.
        
        If filter is applied, connections and boundary padding mode are safely defined using 
        global properties. For instance, with 13_3 microtubule, the 13th monomer in the first 
        round is connected to the 1st monomer in the 4th round.

        Parameters
        ----------
        layer : MonomerLayer
            Select which layer will be calculated.
        filter_length : int, default is 1
            Length of uniform filter kernel. Must be an odd number. If 1, no filter will be 
            applied.
        filter_width : int, default is 1
            Width (lateral length) of uniform filter kernel. Must be an odd number. If 1, no
            filter will be applied.
        spline_precision : nm, optional
            Precision in nm that is used to define the direction of molecules for calculating
            projective interval.
        """
        ndim = 3
        if filter_length % 2 == 0 or filter_width % 2 == 0:
            raise ValueError("'filter_length' and 'filter_width' must be odd numbers.")
        mole: Molecules = layer.metadata[MOLECULES]
        spl = molecules_to_spline(layer)
        npf = roundint(np.max(layer.features[Mole.pf]) + 1)
        try:
            pf_label = layer.features[Mole.pf]
            pos_list: List[np.ndarray] = []  # each shape: (y, ndim)
            for pf in range(pf_label.max() + 1):
                pos_list.append(mole.pos[pf_label == pf])
            pos = np.stack(pos_list, axis=1)  # shape: (y, pf, ndim)
            
        except Exception as e:
            raise TypeError(
                f"Reshaping failed. Molecules represented by layer {layer.name} must be "
                f"correctly labeled at {Mole.pf!r} feature. Original error is\n"
                f"{type(e).__name__}: {e}"
            ) from e
        
        pitch_vec = np.diff(pos, axis=0, append=(2*pos[-1] - pos[-2])[np.newaxis])
        u = spl.world_to_y(mole.pos, precision=spline_precision)
        spl_vec = spl(u, der=1)
        spl_vec_norm: np.ndarray = spl_vec / np.sqrt(np.sum(spl_vec**2, axis=1))[:, np.newaxis]
        spl_vec_norm = spl_vec_norm.reshape(-1, npf, ndim)
        y_dist: np.ndarray = np.sum(pitch_vec * spl_vec_norm, axis=2)  # inner product

        # apply filter
        if filter_length > 1 or filter_width > 1:
            l_ypad = filter_length // 2
            l_apad = filter_width // 2
            start = y_coords_to_start_number(u, npf)
            self.Panels.log.print(f"geometry: {npf}_{start}")
            input = pad_mt_edges(y_dist[:, ::-1], (l_ypad, l_apad), start=start)
            out = ndi.uniform_filter(input, (filter_length, filter_width), mode="constant")
            ly, lx = out.shape
            y_dist = out[l_ypad:ly-l_ypad, l_apad:lx-l_apad][:, ::-1]

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
    
    @Molecules_.wraps
    @set_options(method={"choices": ["residue", "each", "divide"]})
    def Split(
        self,
        layer: MonomerLayer,
        method: str = "residue",
        n_group: int = 2,
    ):
        mole: Molecules = layer.metadata[MOLECULES]
        nmole = len(mole)
        if method == "residue":
            slices = [slice(i, None, n_group) for i in range(n_group)]
        elif method == "each":
            _id = np.arange(nmole, dtype=np.uint16)
            slices = [_id % n_group == i for i in range(n_group)]
        elif method == "divide":
            borders = np.linspace(0, nmole, n_group + 1).astype(np.uint16)
            slices = [slice(borders[i], borders[i+1]) for i in range(n_group)]
        else:
            raise ValueError(f"{method} is not supported.")
        
        for i, sl in enumerate(slices):
            mol = mole.subset(sl)
            add_molecules(self.parent_viewer, mol, name=f"{layer.name}-G{i:0>2}")
        layer.visible = False
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
            self.mask = MASK_CHOICES[0]
            
        template_path = vfield(Path, options={"label": "Template", "filter": "*.mrc;*.tif"})
        mask = vfield(RadioButtons, options={"label": "Mask", "choices": MASK_CHOICES}, record=False)
        
        @magicclass(layout="horizontal", widget_type="groupbox", name="Parameters")
        class params(MagicTemplate):
            dilate_radius = vfield(1.0, options={"tooltip": "Radius of dilation applied to binarized template (unit: nm).", "step": 0.5, "max": 20}, record=False)
            sigma = vfield(1.0, options={"tooltip": "Standard deviation of Gaussian blur applied to the edge of binary image (unit: nm).", "step": 0.5, "max": 20}, record=False)
            
        @magicclass(layout="horizontal", widget_type="frame")
        class mask_path(MagicTemplate):
            mask_path = vfield(Path, options={"filter": "*.mrc;*.tif"}, record=False)
        
        chunk_size = vfield(200, options={"min": 1, "max": 600, "step": 10, "tooltip": "How many subtomograms will be loaded at the same time."})
        
        @mask.connect
        def _on_switch(self):
            v = self.mask
            self.params.visible = (v == MASK_CHOICES[1])
            self.mask_path.visible = (v == MASK_CHOICES[2])
        
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
            if v == MASK_CHOICES[0]:
                params = None
            elif v == MASK_CHOICES[1]:
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
                    self.mask = MASK_CHOICES[0]
                elif isinstance(params, tuple):
                    self.mask = MASK_CHOICES[1]
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
        
        def _set_mask_params(self, params):
            if params is None:
                self.mask = MASK_CHOICES[0]
            elif isinstance(params, (tuple, list, np.ndarray)):
                self.mask = MASK_CHOICES[1]
                self.params.dilate_radius, self.params.sigma = params
            else:
                self.mask = MASK_CHOICES[2]
                self.mask_path.mask_path = params
        
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
                self._viewer.add_image(
                    image.rescale_intensity(dtype=np.float32), scale=image.scale, name=name,
                    rendering="iso",
                )
            
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
            def Calculate_correlation(self): ...
            def Calculate_FSC(self): ...
            def Seam_search(self): ...
        
        @magicmenu
        class Refinement(MagicTemplate):
            def Align_averaged(self): ...
            def Align_all(self): ...
            def Multi_template_alignment(self): ...
        
        @magicmenu
        class Template(MagicTemplate):
            def Reshape_template(self): ...
        
        @do_not_record
        @set_options(
            new_shape={"options": {"min": 2, "max": 100}},
            save_as={"mode": "w", "filter": "*.mrc;*.tif"}
        )
        @Template.wraps
        def Reshape_template(
            self, 
            new_shape: _Tuple[nm, nm, nm] = (20.0, 20.0, 20.0),
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
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "size (nm)"},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        bin_size={"choices": _get_available_binsize},
    )
    @dispatch_worker
    def Average_all(
        self,
        layer: MonomerLayer,
        size: Optional[nm] = None,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
        interpolation: int = 1,
        bin_size: int = 1,
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
        bin_size : int, default is 1
            Set to >1 if you want to use binned image to boost image analysis.
        """
        molecules: Molecules = layer.metadata[MOLECULES]
        tomo = self.tomogram
        nmole = len(molecules)
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        loader = tomo.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation, chunksize=chunk_size
        )
        nbatch = 24
        worker = create_worker(
            loader.iter_average,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_returned(img: ip.ImgArray):
            self._subtomogram_averaging._show_reconstruction(img, f"[AVG]{layer.name}")
        
        self._WorkerControl.info = f"Subtomogram averaging of {layer.name} (n = {nmole})"
        return worker
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "Subtomogram size (nm)"},
        method={"choices": ["steps", "first", "last", "random"]},
        bin_size={"choices": _get_available_binsize},
    )
    @dispatch_worker
    def Average_subset(
        self,
        layer: MonomerLayer,
        size: Optional[nm] = None,
        method="steps", 
        number: int = 64,
        bin_size: int = 1,
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
        bin_size : int, default is 1
            Set to >1 if you want to use binned image to boost image analysis.
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
        loader = self.tomogram.get_subtomogram_loader(
            mole, shape, binsize=bin_size, order=1, chunksize=1
        )
        
        worker = create_worker(loader.iter_average,
                               _progress={"total": number, "desc": "Running"}
                               )
        
        @worker.returned.connect
        def _on_returned(img: ip.ImgArray):
            self._subtomogram_averaging._show_reconstruction(img, f"Subtomogram average (n={number})")
        
        self._WorkerControl.info = f"Subtomogram Averaging (subset)"

        return worker
    
    def _check_binning_for_alignment(
        self,
        template: Union[ip.ImgArray, List[ip.ImgArray]],
        mask: Union[ip.ImgArray, None],
        binsize: int,
        molecules: Molecules,
        order: int,
        chunk_size: int,
    ) -> Tuple[SubtomogramLoader, ip.ImgArray, Union[ip.ImgArray, None]]:
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=binsize, order=order, chunksize=chunk_size
        )
        if binsize > 1:
            binsize = roundint(self.layer_image.scale[0]/self.tomogram.scale)
            with ip.silent():
                if isinstance(template, list):
                    template = [tmp.binning(binsize, check_edges=False) for tmp in template]
                else:
                    template = template.binning(binsize, check_edges=False)
                if mask is not None:
                    mask = mask.binning(binsize, check_edges=False)
        return loader, template, mask
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        bin_size={"choices": _get_available_binsize}
    )
    @dispatch_worker
    def Align_averaged(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        cutoff: float = 0.5,
        bin_size: int = 1,
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
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied to averaged image.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        bin_size : int, default is 1
            Set to >1 if you want to use binned image to boost image analysis. Be careful! 
            This may cause unexpected fitting result.
        """
        molecules: Molecules = layer.metadata[MOLECULES]
        template: ip.ImgArray = self._subtomogram_averaging._get_template(path=template_path)
        mask: ip.ImgArray = self._subtomogram_averaging._get_mask(params=mask_params)
        if mask is not None and template.shape != mask.shape:
            raise ValueError("Shape mismatch between template and mask.")
        nmole = len(molecules)
        
        loader, template, mask = self._check_binning_for_alignment(
            template, mask, bin_size, molecules, order=1, chunk_size=chunk_size
        )
        _scale = self.tomogram.scale * bin_size
        max_shifts = tuple()
        npf = np.max(layer.features[Mole.pf]) + 1
        dy = np.sqrt(np.sum((molecules.pos[0] - molecules.pos[1])**2))  # longitudinal shift
        dx = np.sqrt(np.sum((molecules.pos[0] - molecules.pos[npf])**2))  # lateral shift
        
        max_shifts = tuple(np.array([dy*0.6, dy*0.6, dx*0.6])/_scale)
        nbatch = 24
        worker = create_worker(
            loader.iter_average,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
        
        self.Panels.log.print_html(f"<code>Align_averaged</code>")
        
        @worker.returned.connect
        def _on_return(image_avg: ip.ImgArray):
            from ..components._align_utils import align_image_to_template, transform_molecules
            with ip.silent():
                img = image_avg.lowpass_filter(cutoff=cutoff)
                
                # if multiscaled image is used, there could be shape mismatch
                if bin_size > 1 and img.shape != template.shape:
                    sl = tuple(slice(0, s) for s in template.shape)
                    img = img[sl]
                    
                rot, shift = align_image_to_template(img, template, mask, max_shifts=max_shifts)
                deg = np.rad2deg(rot)
                cval = np.percentile(image_avg, 1)
                shifted_image = image_avg.affine(
                    translation=shift, cval=cval
                ).rotate(deg, dims="zx", cval=cval)

            shift_nm = shift * img.scale
            vec_str = ", ".join(f"{x}<sub>shift</sub>" for x in "XYZ")
            shift_nm_str = ", ".join(f"{s:.2f} nm" for s in shift_nm[::-1])
            self.Panels.log.print_html(f"rotation = {deg:.2f}&deg;, {vec_str} = {shift_nm_str}")
            points = add_molecules(
                self.parent_viewer, 
                transform_molecules(molecules, shift_nm, [0, -rot, 0]),
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
            )
            points.features = layer.features
            self._subtomogram_averaging._show_reconstruction(shifted_image, "Aligned")
            layer.visible = False
            self.Panels.log.print(f"{layer.name!r} --> {points.name!r}")
                
        self._WorkerControl.info = f"Aligning averaged image (n={nmole}) to template"
        return worker
    
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"options": {"max": 90.0, "step": 0.1}},
        y_rotation={"options": {"max": 180.0, "step": 0.1}},
        x_rotation={"options": {"max": 180.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        bin_size={"choices": _get_available_binsize},
    )
    @dispatch_worker
    def Align_all(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        max_shifts: _Tuple[nm, nm, nm] = (1., 1., 1.),
        z_rotation: _Tuple[float, float] = (0., 0.),
        y_rotation: _Tuple[float, float] = (0., 0.),
        x_rotation: _Tuple[float, float] = (0., 0.),
        cutoff: float = 0.5,
        interpolation: int = 1,
        bin_size: int = 1,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
        template_path : ip.ImgArray, optional
            Template image.
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
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
        bin_size : int, default is 1
            Set to >1 if you want to use binned image to boost image analysis.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        """
        
        molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        nmole = len(molecules)
        
        loader, template, mask = self._check_binning_for_alignment(
            template, 
            mask, 
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
            chunk_size=chunk_size
        )
        nbatch = 24
        worker = create_worker(
            loader.iter_align,
            template=template, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
        
        self.Panels.log.print_html(f"<code>Align_all</code>")
                    
        @worker.returned.connect
        def _on_return(aligned_loader: SubtomogramLoader):
            points = add_molecules(
                self.parent_viewer, 
                aligned_loader.molecules,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
            )
            points.features = layer.features
            layer.visible = False
            self.Panels.log.print(f"{layer.name!r} --> {points.name!r}")
                
        self._WorkerControl.info = f"Aligning subtomograms (n = {nmole})"
        return worker

    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        other_templates={"filter": "*.mrc;*.tif"},
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"options": {"max": 8.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"options": {"max": 5.0, "step": 0.1}},
        y_rotation={"options": {"max": 5.0, "step": 0.1}},
        x_rotation={"options": {"max": 5.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        bin_size={"choices": _get_available_binsize},
    )
    @dispatch_worker
    def Multi_template_alignment(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        other_templates: List[Path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        max_shifts: _Tuple[nm, nm, nm] = (1., 1., 1.),
        z_rotation: _Tuple[float, float] = (0., 0.),
        y_rotation: _Tuple[float, float] = (0., 0.),
        x_rotation: _Tuple[float, float] = (0., 0.),
        cutoff: float = 0.5,
        interpolation: int = 1,
        bin_size: int = 1,
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
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
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
        bin_size : int, default is 1
            Set to >1 if you want to use binned image to boost image analysis.
        chunk_size : int, default is 200
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
        nmole = len(molecules)
        loader, templates, mask = self._check_binning_for_alignment(
            templates,
            mask,
            binsize=bin_size,
            molecules=molecules, 
            order=interpolation,
            chunk_size=chunk_size,
        )
        nbatch = 24
        worker = create_worker(
            loader.iter_align_multi_templates,
            templates=templates, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
                    
        @worker.returned.connect
        def _on_return(out: Tuple[np.ndarray, SubtomogramLoader]):
            labels, aligned_loader = out
            points = add_molecules(
                self.parent_viewer, 
                aligned_loader.molecules,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
            )
            points.features = layer.features
            update_features(points, "opt-template", labels)
            layer.visible = False
                
        self._WorkerControl.info = f"Aligning subtomograms (n={nmole})"
        return worker

    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        bin_size={"choices": _get_available_binsize},
    )
    @dispatch_worker
    def Calculate_correlation(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        interpolation: int = 1,
        bin_size: int = 1,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        nmole = len(molecules)
        # BUG: error when binsize != 1
        loader, template, mask = self._check_binning_for_alignment(
            template,
            mask,
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
            chunk_size=chunk_size,
        )
        nbatch = 24
        worker = create_worker(
            loader.iter_zncc,
            template=template, 
            mask=mask,
            nbatch=nbatch,
            _progress={"total": ceilint(nmole/nbatch), "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_return(corr):
            with self.Panels.log.set_plt(rc_context={"font.size": 15}):
                plt.hist(corr, bins=50)
                plt.title("Zero Normalized Cross Correlation")
                plt.xlabel("Correlation")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.show()
            update_features(layer, Mole.zncc, corr)
        
        self._WorkerControl.info = "Calculating Correlation"
        return worker
        
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        shape={"text": "Use template shape"},
        dfreq={"label": "Frequency precision", "text": "Choose proper value", "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02}},
    )
    @dispatch_worker
    def Calculate_FSC(
        self,
        layer: MonomerLayer,
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        shape: Optional[_Tuple[nm, nm, nm]] = None,
        seed: Optional[int] = 0,
        interpolation: int = 1,
        dfreq: Optional[float] = None,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        """
        Calculate Fourier Shell Correlation using the selected monomer layer.

        Parameters
        ----------
        layer : MonomerLayer
            Select which monomer layer to be used for subtomogram sampling.
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
        shape : (nm, nm, nm), optional
            Shape of subtomograms. Use mask shape by default.
        seed : int, optional
            Random seed used for subtomogram sampling.
        interpolation : int, default is 1
            Interpolation order.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        chunk_size : int, default is 200
            How many subtomograms will be loaded at the same time.
        """
        mole: Molecules = layer.metadata[MOLECULES]
        nmole = len(mole)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        if shape is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            mole,
            shape,
            order=interpolation,
            chunksize=chunk_size
        )
        if mask is None:
            mask = 1
        else:
            loader._check_shape(mask, "mask")
        if dfreq is None:
            dfreq = 1.5 / min(shape) * loader.scale
        worker = create_worker(
            loader.iter_average_split,
            seed=seed,
            _progress={"total": ceilint(nmole/loader.chunksize), "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_returned(out: tuple[ip.ImgArray, ip.ImgArray]):
            img0, img1 = out
            
            with ip.silent():
                freq, fsc = ip.fsc(img0*mask, img1*mask, dfreq=dfreq)
            
            ind = (freq <= 0.7)
            crit = 0.143
            with self.Panels.log.set_plt(rc_context={"font.size": 15}):
                plt.axhline(0.0, color="gray", alpha=0.5, ls="--")
                plt.axhline(1.0, color="gray", alpha=0.5, ls="--")
                plt.axhline(crit, color="violet")
                plt.plot(freq[ind], fsc[ind], color="gold")
                plt.xlabel("Spatial frequence (1/nm)")
                plt.ylabel("FSC")
                plt.ylim(-0.1, 1.1)
                plt.title(f"FSC of {layer.name}")
                xticks = np.linspace(0, 0.7, 8)
                per_nm = [r"$\infty$"] + [f"{x:.2f}" for x in self.tomogram.scale / xticks[1:]]
                plt.xticks(xticks, per_nm)
                plt.tight_layout()
                plt.show()
            
            freq0 = None
            for i, fsc1 in enumerate(fsc):
                if fsc1 < crit:
                    if i == 0:
                        resolution = "N.A."
                        break
                    f0 = freq[i-1]
                    f1 = freq[i]
                    fsc0 = fsc[i-1]
                    freq0 = (crit - fsc1)/(fsc0 - fsc1) * (f0 - f1) + f1
                    resolution = f"{self.tomogram.scale / freq0:.3f}"
                    break
            else:
                resolution = "N.A."
            
            self.Panels.log.print_html(f"resolution = <b>{resolution} nm</b>")
        
        self._WorkerControl.info = "Calculating FSC ..."
        return worker
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        npf={"text": "Use global properties"},
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
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
        interpolation : int, default is 1
            Interpolation order.
        npf : int, optional
            Number of protofilaments. By default the global properties stored in the corresponding spline
            will be used.
        """
        molecules: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape, order=interpolation, chunksize=chunk_size
        )
        if npf is None:
            npf = np.max(layer.features[Mole.pf]) + 1
        
        total = ceilint(len(molecules) / chunk_size)
            
        worker = create_worker(
            loader.iter_each_seam,
            npf=npf,
            template=template,
            mask=mask,
            _progress={"total": total, "desc": "Running"}
        )
        
        @worker.returned.connect
        def _on_returned(result: Tuple[np.ndarray, ip.ImgArray]):
            corrs, img_ave, all_labels = result
            self._subtomogram_averaging._show_reconstruction(img_ave, layer.name)
            
            # calculate score and the best PF position
            corr1, corr2 = corrs[:npf], corrs[npf:]
            score = np.empty_like(corrs)
            score[:npf] = corr1 - corr2
            score[npf:] = corr2 - corr1
            imax = np.argmax(score)
                
            # plot all the correlation
            self.Panels.log.print_html("<code>Seam_search</code>")
            with self.Panels.log.set_plt(rc_context={"font.size": 15}):
                plt.axvline(imax, color="gray", alpha=0.6)
                plt.axhline(corrs[imax], color="gray", alpha=0.6)
                plt.plot(corrs)
                plt.xlabel("Seam position")
                plt.ylabel("Correlation")
                plt.xticks(np.arange(0, 2*npf+1, 4))
                plt.title("Seam search result")
                plt.tight_layout()
                plt.show()
                
                # plot the score
                plt.plot(score)
                plt.xlabel("PF position")
                plt.ylabel("ΔCorr")
                plt.xticks(np.arange(0, 2*npf+1, 4))
                plt.title("Score")
                plt.tight_layout()
                plt.show()
            
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
        
        length_px = tomo.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)
        
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
        
        length_px = tomo.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)
        
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
        if self._current_ft_size is None:
            raise ValueError("Local structural parameters have not been determined yet.")
        lbl = np.zeros(self.layer_image.data.shape, dtype=np.uint8)
        color: dict[int, List[float]] = {0: [0, 0, 0, 0]}
        tomo = self.tomogram
        bin_scale = self.layer_image.scale[0] # scale of binned reference image
        binsize = roundint(bin_scale/tomo.scale)
        ft_size = self._current_ft_size
        
        lz, ly, lx = [roundint(r/bin_scale*1.4)*2 + 1 for r in [15, ft_size/2, 15]]
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
            center = tomo.nm2pixel(crd, binsize=binsize)
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
        limit={"options": {"min": -20, "max": 20, "step": 0.01}, "label": "limit (nm)"},
        color_by={"choices": [H.yPitch, H.skewAngle, H.nPF, H.riseAngle]},
        auto_call=True
    )
    def Set_colormap(
        self,
        start: Color = (0, 0, 1, 1),
        end: Color = (1, 0, 0, 1),
        limit: _Tuple[float, float] = (4.00, 4.24), 
        color_by: str = H.yPitch,
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
        color_by : str, default is "yPitch"
            Select what property will be colored.
        """        
        self.label_colormap = Colormap([start, end], name="LocalProperties")
        self.label_colorlimit = limit
        self._update_colormap(prop=color_by)
        return None

    @Molecules_.wraps
    @do_not_record
    def Open_feature_control(self):
        self._FeatureControl.show()
        return None
    
    @Image.wraps
    def Show_colorbar(self):
        """Create a colorbar from the current colormap."""
        arr = self.label_colormap.colorbar[:5]  # shape == (5, 28, 4)
        xmin, xmax = self.label_colorlimit
        with self.Panels.log.set_plt(rc_context={"font.size": 15}):
            plt.imshow(arr)
            plt.xticks([0, 27], [f"{xmin:.2f}", f"{xmax:.2f}"])
            plt.yticks([], [])
            plt.show()
        return None
    
    @nogui
    @do_not_record
    def get_molecules(self, name: str) -> Molecules:
        """Retrieve Molecules object from layer list."""
        return self.parent_viewer.layers[name].metadata[MOLECULES]

    @nogui
    @do_not_record
    def get_loader(self, name: str, order: int = 1, chunksize: int = 64) -> SubtomogramLoader:
        mole = self.get_molecules(name)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            mole, shape, order=order, chunksize=chunksize
        )
        return loader
    
    @nogui
    @do_not_record
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

    def _imread(self, path: str, scale: nm, binsize: int):
        tomo = MtTomogram.imread(path, scale=scale)
        tomo.add_multiscale(binsize)
        if self._current_ft_size is not None:
            tomo.metadata["ft_size"] = self._current_ft_size
        return tomo
        
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
            box_size = (GVar.fitWidth,) + ((GVar.fitWidth+GVar.fitLength)/1.41,)*2
            
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
        
        length_px = tomo.nm2pixel(GVar.fitLength)
        width_px = tomo.nm2pixel(GVar.fitWidth)
        
        coords = spl.local_cartesian(
            shape=(width_px, width_px), 
            n_pixels=length_px,
            u=spl.anchors[j],
            scale=tomo.scale
        )
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
        if self._current_ft_size is None:
            raise ValueError("Local structural parameters have not been determined yet.")
        
        ylen = tomo.nm2pixel(self._current_ft_size)
        spl = tomo._splines[i]
        
        rmin = tomo.nm2pixel(spl.radius*GVar.inner)
        rmax = tomo.nm2pixel(spl.radius*GVar.outer)
        
        coords = spl.local_cylindrical(
            r_range=(rmin, rmax), 
            n_pixels=ylen, 
            u=spl.anchors[j],
            scale=tomo.scale
        )
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
        else:
            self.GlobalProperties._init_text()
    
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
        else:
            self.LocalProperties._init_plot()
            self.LocalProperties._init_text()
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
    bin_size: int,
    interval: nm,
    ft_size,
    n_refine: int,
    max_shift,
    edge_sigma,
    local_props,
    global_props
) -> Iterator[str]:
    n_spl = len(splines)
    for i_spl in splines:
        if i_spl > 0:
            yield f"[{i_spl + 1}/{n_spl}] Spline fitting"
        tomo.fit(i=i_spl, edge_sigma=edge_sigma, max_shift=max_shift, binsize=bin_size)
        
        for i in range(n_refine):
            yield f"[{i_spl + 1}/{n_spl}] Spline refinement (iteration {i + 1}/{n_refine})"
            tomo.refine(i=i_spl, max_interval=max(interval, 30), binsize=bin_size)
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


def json_encoder(obj):    
    """Enable Enum and pandas encoding."""
    if isinstance(obj, Ori):
        return obj.name
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        raise TypeError(f"{obj!r} is not JSON serializable")

