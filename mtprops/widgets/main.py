import os
import re
from typing import Iterable, Union, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import napari
from napari.utils import Colormap
from napari.layers import Points, Image, Labels, Layer

import impy as ip
import macrokit as mk
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
    nogui,
    mark_preview,
    )
from magicclass.types import Color, Bound, Optional
from magicclass.widgets import (
    Logger,
    Separator,
    RadioButtons,
    ConsoleTextEdit,
    Select,
    FloatRangeSlider,
)
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.utils import thread_worker

from ..components import (
    SubtomogramLoader,
    Molecules,
    MtSpline,
    MtTomogram,
    AlignmentModel,
)
from ..components.microtubule import angle_corr
from ..utils import (
    crop_tomogram,
    interval_filter,
    make_slice_and_pad,
    map_coordinates,
    mirror_zncc,
    pad_template, 
    roundint,
    ceilint,
    set_gpu
)

from .global_variables import GlobalVariables
from .properties import GlobalPropertiesWidget, LocalPropertiesWidget
from .spline_control import SplineControl
from .spline_fitter import SplineFitter
from .feature_control import FeatureControl
from .image_processor import ImageProcessor
from .project import MTPropsProject
from .project_editor import SubtomogramAveragingProjectEditor
from ._previews import view_tables, view_text, view_image, view_surface
from .widget_utils import (
    FileFilter,
    add_molecules,
    change_viewer_focus,
    update_features,
    molecules_to_spline,
    y_coords_to_start_number,
    get_versions,
    resolve_path,
)

from ..const import nm, H, Ori, GVar, Mole
from ..const import WORKING_LAYER_NAME, SELECTION_LAYER_NAME, ALN_SUFFIX, MOLECULES
from ..types import MonomerLayer, get_monomer_layers
from ..ext.etomo import PEET

ICON_DIR = Path(__file__).parent / "icons"
SPLINE_ID = "spline-id"
MASK_CHOICES = ("No mask", "Use blurred template as a mask", "Supply a image")

def _fmt_layer_name(fmt: str):
    """Define a formatter for progressbar description."""
    def _formatter(**kwargs):
        layer: Layer = kwargs["layer"]
        return fmt.format(layer.name)
    return _formatter


@magicclass(widget_type="scrollable", name="MTProps")
class MTPropsWidget(MagicTemplate):
    # Main GUI class.
    
    _SplineFitter = field(SplineFitter, name="Spline fitter")
    _ImageProcessor = field(ImageProcessor, name="Image Processor")
    _FeatureControl = field(FeatureControl, name="Feature Control")
    _STAProjectEditor = field(SubtomogramAveragingProjectEditor, name="Subtomogram Averaging project editor")
    
    @magicclass(labels=False, name="Logger")
    @set_design(min_height=200)
    class _LoggerWindow(MagicTemplate):
        log = field(Logger, name="Log")
    
    @property
    def log(self):
        return self._LoggerWindow.log
    
    @magicmenu
    class File(MagicTemplate):
        """File I/O."""  
        def open_image(self): ...
        def load_project(self): ...
        def load_splines(self): ...
        def load_molecules(self): ...
        sep0 = field(Separator)
        def save_project(self): ...
        def save_spline(self): ...
        def save_molecules(self): ...
        sep1 = field(Separator)
        def process_images(self): ...
        PEET = PEET

    @magicmenu
    class Image(MagicTemplate):
        """Image processing and visualization"""
        def show_image_info(self): ...
        def filter_reference_image(self): ...
        def add_multiscale(self): ...
        def set_multiscale(self): ...
        @magicmenu
        class Cylindric(MagicTemplate):
            def show_current_ft(self): ...
            def show_global_ft(self): ...
            def show_r_proj(self): ...
            def show_global_r_proj(self): ...
        sep0 = field(Separator)
        def sample_subtomograms(self): ...
        def paint_mt(self): ...
        def set_colormap(self): ...
        def show_colorbar(self): ...
    
    @magicmenu
    class Splines(MagicTemplate):
        """Operations on splines"""
        def show_splines(self): ...
        def add_anchors(self): ...
        sep0 = field(Separator)
        def invert_spline(self): ...
        def align_to_polarity(self): ...
        def clip_spline(self): ...
        sep1 = field(Separator)
        def fit_splines(self): ...
        def fit_splines_manually(self): ...
        def refine_splines(self): ...
        def molecules_to_spline(self): ...

    @magicmenu
    class Molecules_(MagicTemplate):
        """Operations on molecules"""
        @magicmenu
        class Mapping(MagicTemplate):
            def map_monomers(self): ...
            def map_monomers_manually(self): ...
            def map_centers(self): ...
            def map_along_pf(self): ...
        def show_orientation(self): ...
        def extend_molecules(self): ...
        def calculate_intervals(self): ...
        sep0 = field(Separator)
        def open_feature_control(self): ...
        
    @magicmenu
    class Analysis(MagicTemplate):
        """Analysis of tomograms."""
        def set_radius(self): ...
        def local_ft_analysis(self): ...
        def global_ft_analysis(self): ...
        sep0 = field(Separator)
        def open_subtomogram_analyzer(self): ...
        def open_project_editor(self): ...
    
    @magicmenu
    class Others(MagicTemplate):
        """Other menus."""
        @magicmenu
        class Macro:
            def show_macro(self): ...
            def show_full_macro(self): ...
            def show_native_macro(self): ...
            sep0 = field(Separator)
            def run_file(self): ...
        Global_variables = GlobalVariables
        def open_logger(self): ...
        def clear_cache(self): ...
        def restore_layers(self): ...
        @magicmenu
        class Help(MagicTemplate):
            def open_help(self): ...
            def MTProps_info(self): ...
            def report_issues(self): ...
        
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
            max_shifts = vfield(5.0, options={"min": 1., "max": 20., "step": 0.5, "tooltip": "Maximum shift (nm) in auto centering"}, record=False)
        sep1 = field(Separator)
        def clear_current(self): ...
        def clear_all(self): ...
    
    SplineControl = SplineControl
    LocalProperties = field(LocalPropertiesWidget, name="Local Properties")
    GlobalProperties = field(GlobalPropertiesWidget, name="Global Properties")
    overview = field(QtImageCanvas, name="Overview", options={"tooltip": "Overview of splines"})
    
    ### methods ###
    
    def __init__(self):
        self.tomogram: MtTomogram = None
        self._current_ft_size: nm = None
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
        
        # automatically set scale and binsize
        mgui = get_function_gui(self, "open_image")
        @mgui.path.changed.connect
        def _read_scale():
            path = mgui.path.value
            if not os.path.exists(path):
                return
            img = ip.lazy_imread(path, chunks=GVar.daskChunk)
            scale = img.scale.x
            mgui.scale.value = f"{scale:.4f}"
            mgui.bin_size.value = ceilint(0.96 / scale)
        
        return None

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
        def _get_splines(self, _=None) -> List[Tuple[str, int]]:
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
                n_choices = len(self["splines"].choices)
                return list(range(n_choices))
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
    @bind_key("F2")
    @do_not_record
    def open_runner(self):
        """Run MTProps with various settings."""
        self._runner.show(run=False)
        return None
    
    @_runner.wraps
    @set_design(text="Run")
    @thread_worker(progress={"desc": "Running MTProps", 
                             "total": "(1+n_refine+int(local_props)+int(global_props))*len(splines)"})
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
        if self.layer_work.data.size > 0:
            raise ValueError("The last spline is not registered yet.")
        if self.tomogram.n_splines == 0:
            raise ValueError("No spline found.")
        elif len(splines) == 0:
            splines = range(self.tomogram.n_splines)
        
        tomo = self.tomogram
        for i_spl in splines:
            tomo.fit(i=i_spl, edge_sigma=edge_sigma, max_shift=max_shift, binsize=bin_size)
            
            for _ in range(n_refine):
                yield
                tomo.refine(i=i_spl, max_interval=max(interval, 30), binsize=bin_size)
            tomo.set_radius(i=i_spl)
                
            tomo.make_anchors(i=i_spl, interval=interval)
            if local_props:
                yield
                tomo.local_ft_params(i=i_spl, ft_size=ft_size)
            if global_props:
                yield
                tomo.global_ft_params(i=i_spl)
            yield
                    
        tomo.metadata["ft_size"] = self._current_ft_size
        self._current_ft_size = ft_size
        self._need_save = True
        
        return local_props, global_props, splines, paint
    
    @run_mtprops.started.connect
    def _run_mtprops_on_start(self):
        return self._runner.close()
    
    @run_mtprops.returned.connect
    def _run_mtprops_on_return(self, out):
        local_props, global_props, splines, paint = out
        if local_props or global_props:
            self.sample_subtomograms()
            if global_props:
                df = self.tomogram.collect_globalprops(i=splines).transpose()
                df.columns = [f"Spline-{i}" for i in splines]
                self.log.print_table(df, precision=3)
        if local_props and paint:
            self.paint_mt()
        if global_props:
            self._update_global_properties_in_widget()
        self._update_splines_in_images()
        return None
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"clear_last.png")
    @do_not_record
    def clear_current(self):
        """Clear current selection."""        
        self.layer_work.data = []
        return None
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"clear_all.png")
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
    @set_options(path={"filter": "Python (*.py);;All files (*)"})
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
        new = self.macro.widget.new()
        new.value = str(self._format_macro()[self._macro_offset:])
        new.show()
        return None
    
    @Others.Macro.wraps
    @set_design(text="Show full macro")
    @do_not_record
    def show_full_macro(self):
        """Create Python executable script since the startup this time."""
        new = self.macro.widget.new()
        new.value = str(self._format_macro())
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
        self.macro.widget.textedit.read_only = True
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
    @set_design(text="Restore layers")
    @do_not_record
    def restore_layers(self):
        """Restore mistakenly deleted layers."""
        for layer in (self.layer_image, self.layer_work, self.layer_prof, self.layer_paint):
            if layer not in self.parent_viewer.layers:
                self.parent_viewer.add_layer(layer)
        return None
    
    @Others.Help.wraps
    @do_not_record
    def MTProps_info(self):
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
        open_url("https://github.com/hanjinliu/MTProps/issues/new")
        return None
        
    @File.wraps
    @set_options(
        path={"filter": FileFilter.IMAGE},
        scale={"min": 0.001, "step": 0.0001, "max": 10.0, "label": "scale (nm)"},
        bin_size={"min": 1, "max": 8}
    )
    @set_design(text="Open image")
    @thread_worker(progress={"desc": "Reading image"})
    @confirm(text="You may have unsaved data. Open a new tomogram?", condition="self._need_save")
    def open_image(
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
        
        tomo = self._imread(
            path=path,
            scale=scale,
            binsize=bin_size,
        )

        if self._current_ft_size is not None:
            tomo.metadata["ft_size"] = self._current_ft_size
        self._macro_offset = len(self.macro)
        self.tomogram = tomo
        return tomo
    
    @File.wraps
    @set_options(path={"filter": FileFilter.JSON})
    @set_design(text="Load project")
    @thread_worker(progress={"desc": "Reading project"})
    @confirm(text="You may have unsaved data. Open a new project?", condition="self._need_save")
    def load_project(self, path: Path):
        """Load a project json file."""
        project = MTPropsProject.from_json(path)
        file_dir = Path(path).parent
        
        # load image and multiscales
        multiscales = project.multiscales
        
        self.tomogram = self._imread(
            path=resolve_path(project.image, file_dir), 
            scale=project.scale, 
            binsize=multiscales.pop(-1), 
        )
        
        # resolve paths
        project.localprops = resolve_path(project.localprops, file_dir)
        project.globalprops = resolve_path(project.globalprops, file_dir)
        project.template_image = resolve_path(project.template_image, file_dir)
        project.global_variables = resolve_path(project.global_variables, file_dir)
        project.splines = [resolve_path(p, file_dir) for p in project.splines]
        project.molecules = [resolve_path(p, file_dir) for p in project.molecules]
        
        self._current_ft_size = project.current_ft_size
        self._macro_offset = len(self.macro)
        
        for size in multiscales:
            self.tomogram.add_multiscale(size)
            
        return project

    @load_project.returned.connect
    def _load_project_on_return(self, project: MTPropsProject):
        self._send_tomogram_to_viewer()
        
        # load splines
        splines = [MtSpline.from_json(path) for path in project.splines]
        localprops_path = project.localprops
        if localprops_path is not None:
            all_localprops = dict(iter(pd.read_csv(localprops_path).groupby("SplineID")))
        else:
            all_localprops = {}
        globalprops_path = project.globalprops
        if globalprops_path is not None:
            all_globalprops = dict(pd.read_csv(globalprops_path).iterrows())
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
        
        # load subtomogram analyzer state
        self._subtomogram_averaging.template_path = project.template_image or ""
        self._subtomogram_averaging._set_mask_params(project.mask_parameters)
        self._subtomogram_averaging.chunk_size = project.chunksize or 200
        self.reset_choices()
        self._need_save = False
        return None
    
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
        features will be exported as csv files.

        Parameters
        ----------
        json_path : Path
            Path of json file.
        results_dir : Path, optional
            Optionally you can specify the directory to save csv files.
        """
        _versions = get_versions()
        tomo = self.tomogram
        localprops = tomo.collect_localprops()    
        globalprops = tomo.collect_globalprops()
        
        _json_path = Path(json_path)
        if results_dir is None:
            results_dir = _json_path.parent / (_json_path.stem + "_results")
        else:
            results_dir = Path(results_dir)
        localprops_path = None if localprops is None else results_dir / "localprops.csv"
        globalprops_path = None if globalprops is None else results_dir / "globalprops.csv"
        
        # Save path of splines
        spline_paths: List[Path] = []
        for i, spl in enumerate(self.tomogram.splines):
            spline_paths.append(results_dir/f"spline-{i}.json")
            
        # Save path of molecules
        molecule_dataframes: List[pd.DataFrame] = []
        molecules_paths: List[Path] = []
        for layer in filter(
            lambda x: isinstance(x, Points) and MOLECULES in x.metadata,
            self.parent_viewer.layers
        ):
            layer: Points
            mole: Molecules = layer.metadata[MOLECULES]
            molecule_dataframes.append(mole.to_dataframe())
            molecules_paths.append((results_dir/layer.name).with_suffix(".csv"))
        
        # Save path of  global variables
        gvar_path = results_dir / "global_variables.json"
        
        # Save path of macro
        macro_path = results_dir / "script.py"
        macro_str = str(self._format_macro(self.macro[self._macro_offset:]))
        
        from datetime import datetime
        
        file_dir = json_path.parent
        def as_relative(p):
            try:
                out = p.relative_to(file_dir)
            except Exception:
                out = p
            return out
        
        project = MTPropsProject(
            datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            version = _versions.pop("MTProps"),
            dependency_versions = _versions,
            image = as_relative(tomo.source),
            scale = tomo.scale,
            multiscales = [x[0] for x in tomo.multiscaled],
            current_ft_size = self._current_ft_size,
            splines = [as_relative(p) for p in spline_paths],
            localprops = as_relative(localprops_path),
            globalprops = as_relative(globalprops_path),
            molecules = [as_relative(p) for p in molecules_paths],
            global_variables = as_relative(gvar_path),
            template_image = as_relative(self._subtomogram_averaging.template_path),
            mask_parameters = self._subtomogram_averaging._get_mask_params(),
            chunksize = self._subtomogram_averaging.chunk_size,
            macro = as_relative(macro_path),
        )
        
        # save objects
        project.to_json(_json_path)
        
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)  # create a directory if not exists.
        if localprops_path:
            localprops.to_csv(localprops_path)
        if globalprops_path:
            globalprops.to_csv(globalprops_path)
        if spline_paths:
            for spl, path in zip(self.tomogram.splines, spline_paths):
                spl.to_json(path)
        if molecules_paths:
            for df, fp in zip(molecule_dataframes, molecules_paths):
                df.to_csv(fp, index=False)
        
        self.Others.Global_variables.save_variables(gvar_path)
        
        if macro_str:
            with open(macro_path, mode="w") as f:
                f.write(macro_str)
        
        self._need_save = False
        return
    
    @File.wraps
    @set_options(paths={"filter": FileFilter.JSON})
    @set_design(text="Load splines")
    def load_splines(self, paths: List[Path]):
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        splines = [MtSpline.from_json(path) for path in paths]
        self.tomogram.splines.extend(splines)
        self.reset_choices()
        return None
        
    @File.wraps
    @set_options(paths={"filter": FileFilter.CSV})
    @set_design(text="Load molecules")
    def load_molecules(self, paths: List[Path]):
        """Load molecules from a csv file."""
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        for path in paths:
            mole = Molecules.from_csv(path)
            name = Path(path).stem
            add_molecules(self.parent_viewer, mole, name)
        return None
    
    @File.wraps
    @set_options(
        spline={"choices": _get_splines},
        save_path={"mode": "w", "filter": FileFilter.JSON}
    )
    @set_design(text="Save spline")
    def save_spline(self, spline: int, save_path: Path):
        spl = self.tomogram.splines[spline]
        spl.to_json(save_path)
        return None
        
    @File.wraps
    @set_design(text="Save molecules")
    @set_options(save_path={"mode": "w", "filter": FileFilter.CSV})
    def save_molecules(
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
        self._ImageProcessor.show()
        return None
    
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
    @thread_worker(progress={"desc": "Low-pass filtering"})
    def filter_reference_image(self):
        """Apply low-pass filter to enhance contrast of the reference image."""
        cutoff = 0.2
        with ip.silent(), set_gpu():
            img: ip.ImgArray = self.layer_image.data
            overlap = [min(s, 32) for s in img.shape]
            self.layer_image.data = img.tiled_lowpass_filter(
                cutoff, chunks=(96, 96, 96), overlap=overlap,
                )
        return np.percentile(self.layer_image.data, [1, 97])

    @filter_reference_image.returned.connect
    def _filter_reference_image_on_return(self, contrast_limits):
        self.layer_image.contrast_limits = contrast_limits
        with ip.silent():
            proj = self.layer_image.data.proj("z")
        self.overview.image = proj
        self.overview.contrast_limits = contrast_limits
        return None

    @Image.wraps
    @set_options(bin_size={"min": 2, "max": 64})
    @set_design(text="Add multi-scale")
    @thread_worker(progress={"desc": "Adding multiscale (bin = {bin_size})".format})
    def add_multiscale(self, bin_size: int = 2):
        """
        Add a new multi-scale image of current tomogram.

        Parameters
        ----------
        bin_size : int, default is 2
            Bin size of the new image
        """
        tomo = self.tomogram        
        tomo.get_multiscale(binsize=bin_size, add=True)
        self._need_save = True
        return bin_size
    
    @Image.wraps
    @add_multiscale.returned.connect
    @set_options(bin_size={"choices": _get_available_binsize})
    @set_design(text="Set multi-scale")
    def set_multiscale(self, bin_size: int):
        tomo = self.tomogram
        imgb = tomo.get_multiscale(bin_size)
        self.layer_image.data = imgb
        self.layer_image.scale = imgb.scale
        self.layer_image.name = f"{imgb.name} (bin {bin_size})"
        self.layer_image.translate = [tomo.multiscale_translation(bin_size)] * 3
        self.layer_image.contrast_limits = [np.min(imgb), np.max(imgb)]
        with ip.silent():
            self.overview.image = imgb.proj("z")
        self.layer_image.metadata["current_binsize"] = bin_size
        return None
        
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
    
    @Image.Cylindric.wraps
    @set_design(text="R-projection")
    def show_r_proj(self, i: Bound[SplineControl.num], j: Bound[SplineControl.pos]):
        """Show radial projection of cylindrical image around the current MT fragment."""
        with ip.silent():
            polar = self._current_cylindrical_img().proj("r")
        
        canvas = QtImageCanvas()
        canvas.image = polar.value
        canvas.text_overlay.update(visible=True, text=f"{i}-{j}", color="lime")
        canvas.show()
        return None
    
    @Image.Cylindric.wraps
    @set_design(text="R-projection (Global)")
    def show_global_r_proj(self):
        """Show radial projection of cylindrical image along current MT."""        
        i = self.SplineControl.num
        with ip.silent():
            polar = self.tomogram.straighten_cylindric(i).proj("r")
        canvas = QtImageCanvas()
        canvas.image = polar.value
        canvas.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        canvas.show()
        return None
    
    @Image.Cylindric.wraps
    @set_design(text="2D-FT")
    def show_current_ft(self, i: Bound[SplineControl.num], j: Bound[SplineControl.pos]):
        """View Fourier space of local cylindrical coordinate system at current position."""        
        with ip.silent():
            polar = self._current_cylindrical_img(i, j)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw /= pw.max()
        
        canvas = QtImageCanvas()
        canvas.image = pw.value
        canvas.text_overlay.update(visible=True, text=f"{i}-{j}", color="lime")
        canvas.show()
        return None
    
    @Image.Cylindric.wraps
    @set_design(text="2D-FT (Global)")
    def show_global_ft(self, i: Bound[SplineControl.num]):
        """View Fourier space along current MT."""  
        with ip.silent():
            polar: ip.ImgArray = self.tomogram.straighten_cylindric(i)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw = pw.max()
        
        canvas = QtImageCanvas()
        canvas.image = pw.value
        canvas.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        canvas.show()
        return None
    
    @Splines.wraps
    @set_design(text="Show splines")
    def show_splines(self):
        """Show 3D spline paths of microtubule center axes as a layer."""        
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
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_options(orientation={"choices": ["MinusToPlus", "PlusToMinus"]})
    @set_design(text="Align to polarity")
    def align_to_polarity(self, orientation: Ori = "MinusToPlus"):
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
            self.sample_subtomograms()
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_options(
        auto_call=True,
        spline={"choices": _get_splines},
        limits = {"min": 0.0, "max": 1.0, "widget_type": FloatRangeSlider},
    )
    @set_design(text="Clip splines")
    def clip_spline(self, spline: int, limits: Tuple[float, float] = (0., 1.)):
        # BUG: properties may be inherited in a wrong way
        if spline is None:
            return
        start, stop = limits
        spl = self.tomogram.splines[spline]
        self.tomogram.splines[spline] = spl.restore().clip(start, stop)
        self._update_splines_in_images()
        self._need_save = True
        return None
        
    @Splines.wraps
    @set_options(
        max_interval={"label": "Max interval (nm)"},
        bin_size={"choices": _get_available_binsize},
        edge_sigma={"text": "Do not mask image"},
    )
    @set_design(text="Fit splines")
    @thread_worker(progress={"desc": "Spline Fitting"})
    def fit_splines(
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
        self.tomogram.fit(
            max_interval=max_interval,
            binsize=bin_size,
            degree_precision=degree_precision,
            edge_sigma=edge_sigma,
            max_shift=max_shift,
        )
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_options(max_interval={"label": "Max interval (nm)"})
    @set_design(text="Fit splines manually")
    @do_not_record
    def fit_splines_manually(self, max_interval: nm = 50.0):
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
    @set_design(text="Add anchors")
    @set_options(interval={"label": "Interval between anchors (nm)", "min": 1.0})
    def add_anchors(self, interval: nm = 25.0):
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
        self._need_save = True
        return None
    
    @Analysis.wraps
    @set_options(radius={"text": "Measure radii by radial profile."})
    @set_design(text="Set radius")
    @thread_worker(progress={"desc": "Measuring Radius"})
    def set_radius(self, radius: Optional[nm] = None):
        """Measure MT radius for each spline path."""        
        self.tomogram.set_radius(radius=radius)
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_options(
        max_interval={"label": "Maximum interval (nm)"},
        corr_allowed={"label": "Correlation allowed", "max": 1.0, "step": 0.1},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Refine splines")
    @thread_worker(progress={"desc": "Refining splines"})
    def refine_splines(
        self,
        max_interval: nm = 30,
        corr_allowed: float = 0.9,
        bin_size: int = 1
    ):
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
        
        tomo.refine(
            max_interval=max_interval,
            corr_allowed=corr_allowed,
            binsize = bin_size,
        )
        
        self._need_save = True
        return None
    
    @Splines.wraps
    @set_options(
        layers={"widget_type": "Select", "choices": get_monomer_layers},
        interval={"label": "Interval (nm)", "min": 1.0},
    )
    @set_design(text="Molecules to spline")
    @confirm(
        text="The existing splines will be removed.\n Do you want to run?",
        condition="len(self.SplineControl._get_splines()) > 0",
    )
    def molecules_to_spline(
        self, 
        layers: List[MonomerLayer],
        interval: nm = 24.5,
    ):
        """
        Create splines from molecules.
        
        This function is useful to refine splines using results of subtomogram 
        alignment. Note that this function only works with molecules that is
        correctly assembled by such as :func:`map_monomers`.

        Parameters
        ----------
        layers : list of MonomerLayer
            Select which monomer layers will be used for spline creation.
        interval : nm, default is 24.5
            Interval of spline anchors.
        """        
        splines: List[MtSpline] = []
        for layer in layers:
            mole: Molecules = layer.metadata[MOLECULES]
            spl = MtSpline(degree=GVar.splOrder)
            npf = roundint(np.max(mole.features[Mole.pf]) + 1)
            all_coords = mole.pos.reshape(-1, npf, 3)
            mean_coords = np.mean(all_coords, axis=1)
            spl.fit(mean_coords, variance=GVar.splError**2)
            splines.append(spl)
        
        self.tomogram.splines.clear()
        self.tomogram.splines.extend(splines)
        self.tomogram.make_anchors(interval=interval)
        self.sample_subtomograms()
        self._update_splines_in_images()
        return None
        
    @Analysis.wraps
    @set_design(text="Local FT analysis")
    @thread_worker(progress={"desc": "Local Fourier transform"})
    def local_ft_analysis(self, interval: nm = 24.5, ft_size: nm = 32.0):
        """
        Determine MT structural parameters by local Fourier transformation.

        Parameters
        ----------
        interval : nm, default is 24.5
            Interval of subtomogram analysis.
        ft_size : nm, default is 32.0
            Longitudinal length of local discrete Fourier transformation used for 
            structural analysis.
        """
        tomo = self.tomogram
        if tomo.splines[0].radius is None:
            self.tomogram.set_radius()
        tomo.make_anchors(interval=interval)
        tomo.local_ft_params(ft_size=ft_size)
        self._current_ft_size = ft_size
        self._need_save = True
        return None
    
    @local_ft_analysis.returned.connect
    def _local_ft_analysis_on_return(self, _=None):
        self.sample_subtomograms()
        self._update_splines_in_images()
        self._update_local_properties_in_widget()
        return None
        
    @Analysis.wraps
    @set_design(text="Global FT analysis")
    @thread_worker(progress={"desc": "Global Fourier transform"})
    def global_ft_analysis(self):
        """Determine MT global structural parameters by Fourier transformation."""        
        self.tomogram.global_ft_params()
        self._need_save = True
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    @Molecules_.Mapping.wraps
    @set_options(
        splines={"widget_type": "Select", "choices": _get_splines},
        length={"text": "Use full length"}
    )
    @set_design(text="Map monomers")
    @bind_key("M")
    def map_monomers(
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
        splines={"widget_type": "Select", "choices": _get_splines},
        interval={"text": "Set to dimer length"},
        length={"text": "Use full length"}
    )
    @set_design(text="Map centers")
    def map_centers(
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
        self.log.print_html("<code>Map_centers</code>")
        for i, mol in enumerate(mols):
            _name = f"Center-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.log.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return None
    
    @Molecules_.Mapping.wraps
    @set_options(
        splines={"widget_type": "Select", "choices": _get_splines},
        interval={"text": "Set to dimer length"},
        angle_offset={"max": 360}
    )
    @set_design(text="Map alogn PF")
    def map_along_pf(
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
        self.log.print_html("<code>Map_along_PF</code>")
        for i, mol in enumerate(mols):
            _name = f"PF line-{i}"
            add_molecules(self.parent_viewer, mol, _name)
            self.log.print(f"{_name!r}: n = {len(mol)}")
        self._need_save = True
        return None

    @Molecules_.wraps
    @set_options(orientation={"choices": ["x", "y", "z"]})
    @set_design(text="Show orientation")
    def show_orientation(
        self,
        layer: MonomerLayer,
        orientation: str = "z",
        color: Color = "crimson",
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
        color : Color, default is "crimson
            Vector color shown in viewer.
        """
        mol: Molecules = layer.metadata[MOLECULES]
        name = f"{layer.name} {orientation.upper()}-axis"
        
        vector_data = np.stack([mol.pos, getattr(mol, orientation)], axis=1)
        
        self.parent_viewer.add_vectors(
            vector_data, edge_width=0.3, edge_color=color, length=2.4,
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
        layer : MonomerLayer
            Points layer that contain the molecules.
        prepend : int, default is 0
            Number of molecules to be prepended for each protofilament.
        append : int, default is 0
            Number of molecules to be appended for each protofilament.
        """        
        ndim = 3
        mole: Molecules = layer.metadata[MOLECULES]
        npf = roundint(np.max(mole.features[Mole.pf]) + 1)
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
        features = {Mole.pf: np.arange(len(mole_new), dtype=np.uint32) % npf}
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
    @set_options(
        filter_length={"min": 3, "max": 49, "step": 2},
        filter_width={"min": 1, "max": 15, "step": 2},
        spline_precision={"min": 0.05, "max": 5.0, "step": 0.05, "label": "spline precision (nm)"}
    )
    @set_design(text="Calculate intervals")
    def calculate_intervals(
        self,
        layer: MonomerLayer,
        filter_length: int = 3,
        filter_width: int = 3,
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
        if filter_length % 2 == 0 or filter_width % 2 == 0:
            raise ValueError("'filter_length' and 'filter_width' must be odd numbers.")
        mole: Molecules = layer.metadata[MOLECULES]
        spl = molecules_to_spline(layer)
        npf = roundint(np.max(mole.features[Mole.pf]) + 1)
        try:
            pf_label = mole.features[Mole.pf]
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
        
        u = spl.world_to_y(mole.pos, precision=spline_precision)
        spl_vec = spl(u, der=1)
        start = y_coords_to_start_number(u, npf)
        self.log.print(f"Predicted geometry of {layer.name}: {npf}_{start}")
        y_interval = interval_filter(
            pos,
            spl_vec,
            filter_length=filter_length, 
            filter_width=filter_width,
            start=-start
        )
        
        properties = y_interval.ravel()
        _clim = [GVar.yPitchMin, GVar.yPitchMax]
        
        update_features(layer, {Mole.interval: np.abs(properties)})
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
        self._subtomogram_averaging.show()
    
    @Analysis.wraps
    @set_design(text="Open project editor")
    @do_not_record
    def open_project_editor(self):
        """Open the subtomogram averaging project editor."""
        self._STAProjectEditor.show()
        
    @magicclass(name="Subtomogram averaging")
    class _subtomogram_averaging(MagicTemplate):
        # Widget for subtomogram averaging
        
        def __post_init__(self):
            self._template = None
            self._viewer: Union[napari.Viewer, None] = None
            self._next_layer_name = None
            self.mask = MASK_CHOICES[0]
            
        template_path = vfield(Path, options={"label": "Template", "filter": FileFilter.IMAGE}, record=False)
        mask = vfield(RadioButtons, options={"label": "Mask", "choices": MASK_CHOICES}, record=False)
        
        @magicclass(layout="horizontal", widget_type="groupbox", name="Parameters")
        class params(MagicTemplate):
            dilate_radius = vfield(1.0, options={"tooltip": "Radius of dilation applied to binarized template (unit: nm).", "step": 0.5, "max": 20}, record=False)
            sigma = vfield(1.0, options={"tooltip": "Standard deviation of Gaussian blur applied to the edge of binary image (unit: nm).", "step": 0.5, "max": 20}, record=False)
            
        @magicclass(layout="horizontal", widget_type="frame")
        class mask_path(MagicTemplate):
            mask_path = vfield(Path, options={"filter": FileFilter.IMAGE}, record=False)
        
        chunk_size = vfield(200, options={"min": 1, "max": 600, "step": 10, "tooltip": "How many subtomograms will be loaded at the same time."}, record=False)
        
        @mask.connect
        def _on_switch(self):
            v = self.mask
            self.params.visible = (v == MASK_CHOICES[1])
            self.mask_path.visible = (v == MASK_CHOICES[2])
        
        def _get_template(self, path: Union[str, None] = None, rescale: bool = True) -> ip.ImgArray:
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
            if parent.tomogram is not None and rescale:
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
        
        def _get_mask(
            self,
            params: Union[str, Tuple[int, float], None] = _sentinel
        ) -> Union[ip.ImgArray, None]:
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
                self._viewer = napari.Viewer(title=name, axis_labels=("z", "y", "x"), ndisplay=3)
                volume_menu = Volume()
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
            def average_all(self): ...
            def average_subset(self): ...
            def split_and_average(self): ...
            # def calculate_properties(self): ...
            def calculate_fsc(self): ...
            def seam_search(self): ...
        
        @magicmenu
        class Refinement(MagicTemplate):
            def align_averaged(self): ...
            def align_all(self): ...
            def align_all_template_free(self): ...
            def align_all_multi_template(self): ...
        
        @magicmenu
        class Tools(MagicTemplate):
            def reshape_template(self): ...
            def render_molecules(self): ...
        
        @Tools.wraps
        @do_not_record
        @set_options(
            new_shape={"options": {"min": 2, "max": 100}},
            save_as={"mode": "w", "filter": FileFilter.IMAGE}
        )
        @set_design(text="Reshape template")
        def reshape_template(
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
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "size (nm)"},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Average all")
    @thread_worker(progress={"desc": _fmt_layer_name("Subtomogram averaging of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])"})
    def average_all(
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
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        loader = tomo.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation, chunksize=chunk_size
        )
        img = yield from loader.iter_average()
        return img, f"[AVG]{layer.name}"
        
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "Subtomogram size (nm)"},
        method={"choices": ["steps", "first", "last", "random"]},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Average subset")
    @thread_worker(progress={"desc": _fmt_layer_name("Subtomogram averaging (subset) of {!r}"),
                             "total": "number"})
    def average_subset(
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
        
        img = yield from loader.iter_average()
        
        return img, f"[AVG(n={number})]{layer.name}"
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        size={"text": "Use template shape", "options": {"max": 100.}, "label": "size (nm)"},
        n_set={"min": 1, "label": "number of image pairs"},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Split-and-average")
    @thread_worker(progress={"desc": _fmt_layer_name("Split-and-averaging of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])//chunk_size"})
    def split_and_average(
        self,
        layer: MonomerLayer,
        n_set: int = 1,
        size: Optional[nm] = None,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
        interpolation: int = 1,
        bin_size: int = 1,
    ):
        molecules: Molecules = layer.metadata[MOLECULES]
        tomo = self.tomogram
        if size is None:
            shape = self._subtomogram_averaging._get_shape_in_nm()
        else:
            shape = (size,) * 3
        loader = tomo.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation, chunksize=chunk_size
        )
        img = yield from loader.iter_average_split(n_set=n_set)
        
        return img, f"[Split]{layer.name}"
    
    def _check_binning_for_alignment(
        self,
        template: Union[ip.ImgArray, List[ip.ImgArray]],
        mask: Union[ip.ImgArray, None],
        binsize: int,
        molecules: Molecules,
        order: int,
        chunk_size: int,
    ) -> Tuple[SubtomogramLoader, ip.ImgArray, Union[ip.ImgArray, None]]:
        """
        Returns proper subtomogram loader, template image and mask image that matche the 
        bin size.
        """
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=binsize, order=order, chunksize=chunk_size
        )
        if binsize > 1:
            binsize = roundint(self.layer_image.scale[0]/self.tomogram.scale)
            with ip.silent():
                if template is None:
                    pass
                elif isinstance(template, list):
                    template = [tmp.binning(binsize, check_edges=False) for tmp in template]
                else:
                    template = template.binning(binsize, check_edges=False)
                if mask is not None:
                    mask = mask.binning(binsize, check_edges=False)
        return loader, template, mask
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        z_rotation={"options": {"max": 180.0, "step": 1.0}},
        y_rotation={"options": {"max": 180.0, "step": 1.0}},
        x_rotation={"options": {"max": 90.0, "step": 1.0}},
        bin_size={"choices": _get_available_binsize},
        method={"choices": [("Phase Cross Correlation", "pcc"), ("Zero-mean Normalized Cross Correlation", "ZNCC")]},
    )
    @set_design(text="Align averaged")
    @thread_worker(progress={"desc": _fmt_layer_name("Aligning averaged image of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])+1"})
    def align_averaged(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        z_rotation: Tuple[float, float] = (3., 3.),
        y_rotation: Tuple[float, float] = (15., 3.),
        x_rotation: Tuple[float, float] = (3., 3.),
        bin_size: int = 1,
        method: str = "pcc",
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
        z_rotation : tuple of float, optional
            Rotation in external degree around z-axis.
        y_rotation : tuple of float, optional
            Rotation in external degree around y-axis.
        x_rotation : tuple of float, optional
            Rotation in external degree around x-axis.
        layer : MonomerLayer
            Layer of subtomogram positions and angles.
        chunk_size : int, default is 64
            How many subtomograms will be loaded at the same time.
        bin_size : int, default is 1
            Set to >1 if you want to use binned image to boost image analysis. Be careful! 
            This may cause unexpected fitting result.
        """
        mole: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        if mask is not None and template.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch between tempalte image ({tuple(template.shape)}) "
                f"and mask image ({tuple(mask.shape)})."
            )
        
        loader, template, mask = self._check_binning_for_alignment(
            template, mask, bin_size, mole, order=1, chunk_size=chunk_size
        )
        _scale = self.tomogram.scale * bin_size
        max_shifts = tuple()
        npf = np.max(mole.features[Mole.pf]) + 1
        dy = np.sqrt(np.sum((mole.pos[0] - mole.pos[1])**2))  # longitudinal shift
        dx = np.sqrt(np.sum((mole.pos[0] - mole.pos[npf])**2))  # lateral shift
        
        max_shifts = tuple(np.array([dy*0.6, dy*0.6, dx*0.6])/_scale)
        img = yield from loader.iter_average()
            
        self.log.print_html(f"<code>Align_averaged</code>")
        
        if bin_size > 1 and img.shape != template.shape:
            sl = tuple(slice(0, s) for s in template.shape)
            img = img[sl]
            
        from ..components.align import transform_molecules
        from scipy.spatial.transform import Rotation
        with ip.silent():
            # if multiscaled image is used, there could be shape mismatch
            model = AlignmentModel(
                template,
                mask,
                cutoff=1.0,
                rotations=(z_rotation, y_rotation, x_rotation),
                method=method,
            )
            img_trans, result = model.fit(img, max_shifts=max_shifts)
            rotvec = Rotation.from_quat(result.quat).as_rotvec()
            mole_trans = transform_molecules(
                mole, 
                result.shift * img.scale, 
                rotvec,
            )
        yield
        # logging
        shift_nm = result.shift * img.scale
        vec_str = ", ".join(f"{x}<sub>shift</sub>" for x in "XYZ")
        rotvec_str = ", ".join(f"{x}<sub>rot</sub>" for x in "XYZ")
        shift_nm_str = ", ".join(f"{s:.2f} nm" for s in shift_nm[::-1])
        rot_str = ", ".join(f"{s:.2f}" for s in rotvec[::-1])
        self.log.print_html(f"{rotvec_str} = {rot_str}, {vec_str} = {shift_nm_str}")

        self._need_save = True
        return img_trans, template, mole_trans, layer

    @align_averaged.returned.connect
    def _align_averaged_on_return(self, out: Tuple[ip.ImgArray, ip.ImgArray, Molecules, MonomerLayer]):
        img, template, mole, layer = out
        points = add_molecules(
            self.parent_viewer, 
            mole,
            name=_coerce_aligned_name(layer.name, self.parent_viewer),
        )
        with ip.silent():
            img_norm = img.rescale_intensity(dtype=np.uint8).value
            temp_norm = template.rescale_intensity(dtype=np.uint8).value
        merge: np.ndarray = np.stack([img_norm, temp_norm, img_norm], axis=-1)
        layer.visible = False
        self.log.print(f"{layer.name!r} --> {points.name!r}")
        with self.log.set_plt():
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(np.max(merge, axis=0))
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[1].imshow(np.max(merge, axis=1))
            axes[1].set_xlabel("X")
            axes[0].set_ylabel("Z")
            plt.tight_layout()
            plt.show()
            
        return None
    
    @average_all.returned.connect
    @average_subset.returned.connect
    @split_and_average.returned.connect
    def _average_all_on_return(self, out: Tuple[ip.ImgArray, str]):
        img, layer_name = out
        self._subtomogram_averaging._show_reconstruction(img, layer_name)
        return None
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"options": {"max": 180.0, "step": 0.1}},
        y_rotation={"options": {"max": 180.0, "step": 0.1}},
        x_rotation={"options": {"max": 90.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        method={"choices": [("Phase Cross Correlation", "pcc"), ("Zero-mean Normalized Cross Correlation", "ZNCC")]},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Align all")
    @thread_worker(progress={"desc": _fmt_layer_name("Alignment of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])"})
    def align_all(
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
        method: str = "pcc",
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
        
        loader, template, mask = self._check_binning_for_alignment(
            template, 
            mask, 
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
            chunk_size=chunk_size
        )
        aligned_loader = yield from loader.iter_align(
            template=template, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            method=method,
        )
        
        self.log.print_html(f"<code>Align_all</code>")
        self._need_save = True
        return aligned_loader, layer

    @align_all.returned.connect
    def _align_all_on_return(self, out: Tuple[SubtomogramLoader, MonomerLayer]):
        aligned_loader, layer = out
        points = add_molecules(
            self.parent_viewer, 
            aligned_loader.molecules,
            name=_coerce_aligned_name(layer.name, self.parent_viewer),
        )
        layer.visible = False
        self.log.print(f"{layer.name!r} --> {points.name!r}")
        return None
    
    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"options": {"max": 90.0, "step": 0.1}},
        y_rotation={"options": {"max": 180.0, "step": 0.1}},
        x_rotation={"options": {"max": 180.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        method={"choices": [("Phase Cross Correlation", "pcc"), ("Zero-mean Normalized Cross Correlation", "ZNCC")]},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Align all (template-free)")
    @thread_worker(progress={"desc": _fmt_layer_name("Template-free alignment of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])*2"})
    def align_all_template_free(
        self,
        layer: MonomerLayer,
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        max_shifts: Tuple[nm, nm, nm] = (1., 1., 1.),
        z_rotation: Tuple[float, float] = (0., 0.),
        y_rotation: Tuple[float, float] = (0., 0.),
        x_rotation: Tuple[float, float] = (0., 0.),
        cutoff: float = 0.5,
        interpolation: int = 1,
        method: str = "pcc",
        bin_size: int = 1,
        chunk_size: Bound[_subtomogram_averaging.chunk_size] = 200,
    ):
        """
        Align all the molecules for subtomogram averaging.
        
        Parameters
        ----------
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

        loader, _, _ = self._check_binning_for_alignment(
            None, 
            None, 
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
            chunk_size=chunk_size
        )
        aligned_loader = yield from loader.iter_align_no_template(
            mask_params=mask_params,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            method=method,
        )
        
        self.log.print_html(f"<code>Align_all (template-free)</code>")
                
        self._need_save = True
        return aligned_loader, layer
    
    @align_all_template_free.returned.connect
    def _align_all_template_free_on_return(
        self,
        out: Tuple[SubtomogramLoader, MonomerLayer]
    ):
        aligned_loader, layer = out
        points = add_molecules(
            self.parent_viewer, 
            aligned_loader.molecules,
            name=_coerce_aligned_name(layer.name, self.parent_viewer),
        )
        layer.visible = False
        self.log.print(f"{layer.name!r} --> {points.name!r}")
        return None

    @_subtomogram_averaging.Refinement.wraps
    @set_options(
        other_templates={"filter": FileFilter.IMAGE},
        cutoff={"max": 1.0, "step": 0.05},
        max_shifts={"options": {"max": 8.0, "step": 0.1}, "label": "Max shifts (nm)"},
        z_rotation={"options": {"max": 5.0, "step": 0.1}},
        y_rotation={"options": {"max": 5.0, "step": 0.1}},
        x_rotation={"options": {"max": 5.0, "step": 0.1}},
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        method={"choices": [("Phase Cross Correlation", "pcc"), ("Zero-mean Normalized Cross Correlation", "ZNCC")]},
        bin_size={"choices": _get_available_binsize},
    )
    @set_design(text="Align all (multi-template)")
    @thread_worker(progress={"desc": _fmt_layer_name("Multi-template alignment of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])"})
    def align_all_multi_template(
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
        method: str = "pcc",
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
        loader, templates, mask = self._check_binning_for_alignment(
            templates,
            mask,
            binsize=bin_size,
            molecules=molecules, 
            order=interpolation,
            chunk_size=chunk_size,
        )
        aligned_loader = yield from loader.iter_align_multi_templates(
            templates=templates, 
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            method=method,
        )
        
        self._need_save = True
        return aligned_loader, layer
    
    @align_all_multi_template.returned.connect
    def _align_all_multi_template_on_return(self, out: Tuple[SubtomogramLoader, MonomerLayer]):
        aligned_loader, layer = out
        add_molecules(
            self.parent_viewer, 
            aligned_loader.molecules,
            name=_coerce_aligned_name(layer.name, self.parent_viewer),
        )
        layer.visible = False
        return None
        
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        shape={"text": "Use template shape"},
        n_set={"min": 1, "label": "number of image pairs"},
        dfreq={"label": "Frequency precision", "text": "Choose proper value", "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02}},
    )
    @set_design(text="Calculate FSC")
    @thread_worker(progress={"desc": _fmt_layer_name("Calculating FSC of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])//chunk_size+1"})
    def calculate_fsc(
        self,
        layer: MonomerLayer,
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        shape: Optional[Tuple[nm, nm, nm]] = None,
        seed: Optional[int] = 0,
        interpolation: int = 1,
        n_set: int = 1,
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
        n_set : int, default is 1
            How many sets of image pairs will be generated to average FSC.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        chunk_size : int, default is 200
            How many subtomograms will be loaded at the same time.
        """
        mole: Molecules = layer.metadata[MOLECULES]
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
        img = yield from loader.iter_average_split(n_set=n_set, seed=seed)
        
        fsc_all: List[np.ndarray] = []
        with ip.silent():
            for i in range(n_set):
                img0, img1 = img[i]
                freq, fsc = ip.fsc(img0*mask, img1*mask, dfreq=dfreq)
                fsc_all.append(fsc)
        fsc_all = np.stack(fsc_all, axis=1)
        return freq, fsc_all, layer
    
    @calculate_fsc.returned.connect
    def _calculate_fsc_on_return(self, out: Tuple[np.ndarray, np.ndarray, MonomerLayer]):
        freq, fsc_all, layer = out
        fsc_mean = np.mean(fsc_all, axis=1)
        fsc_std = np.std(fsc_all, axis=1)
        crit_0143 = 0.143
        crit_0500 = 0.500
        
        self.log.print_html(f"<b>Fourier Shell Correlation of {layer.name!r}</b>")
        with self.log.set_plt(rc_context={"font.size": 15}):
            _plot_fsc(freq, fsc_mean, fsc_std, [crit_0143, crit_0500], self.tomogram.scale)
        
        resolution_0143 = _calc_resolution(freq, fsc_mean, crit_0143, self.tomogram.scale)
        resolution_0500 = _calc_resolution(freq, fsc_mean, crit_0500, self.tomogram.scale)
        str_0143 = "N.A." if resolution_0143 == 0 else f"{resolution_0143:.3f} nm"
        str_0500 = "N.A." if resolution_0500 == 0 else f"{resolution_0500:.3f} nm"
        
        self.log.print_html(f"Resolution at FSC=0.5 ... <b>{str_0500}</b>")
        self.log.print_html(f"Resolution at FSC=0.143 ... <b>{str_0143}</b>")
        self._LoggerWindow.show()
        return None    
    
    @_subtomogram_averaging.Subtomogram_analysis.wraps
    @set_options(
        interpolation={"choices": [("linear", 1), ("cubic", 3)]},
        npf={"text": "Use global properties"},
    )
    @set_design(text="Seam search")
    @thread_worker(progress={"desc": _fmt_layer_name("Seam search of {!r}"),
                             "total": f"len(layer.metadata[{MOLECULES!r}])//chunk_size+1"})
    def seam_search(
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
            Select which monomer layer to be used for subtomogram sampling.
        template_path : ip.ImgArray, optional
            Template image.
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
        interpolation : int, default is 1
            Interpolation order.
        npf : int, optional
            Number of protofilaments. By default the global properties stored in the 
            corresponding spline will be used.
        """
        mole: Molecules = layer.metadata[MOLECULES]
        template = self._subtomogram_averaging._get_template(path=template_path)
        mask = self._subtomogram_averaging._get_mask(params=mask_params)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            mole, shape, order=interpolation, chunksize=chunk_size
        )
        if npf is None:
            npf = np.max(mole.features[Mole.pf]) + 1

        result = yield from loader.iter_each_seam(
            npf=npf,
            template=template,
            mask=mask,
        )
        
        self._need_save = True
        return result + (layer, npf)

    @seam_search.returned.connect
    def _seam_search_on_return(
        self, 
        out: Tuple[np.ndarray, ip.ImgArray, np.ndarray, MonomerLayer, int]
    ):
        corrs, img_ave, all_labels, layer, npf = out
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
            _plot_seam_search_result(corrs, score, npf)
            
        self.sub_viewer.layers[-1].metadata["Correlation"] = corrs
        self.sub_viewer.layers[-1].metadata["Score"] = score
        
        update_features(layer, {Mole.isotype: all_labels[imax].astype(np.uint8)})
        return None

    @_subtomogram_averaging.Tools.wraps
    @set_options(
        feature_name={"text": "Do not color molecules."},
        cutoff={"options": {"min": 0.05, "max": 0.8, "step": 0.05}},
    )
    @set_design(text="Render molecules")
    def render_molecules(
        self,
        layer: MonomerLayer,
        template_path: Bound[_subtomogram_averaging.template_path],
        mask_params: Bound[_subtomogram_averaging._get_mask_params],
        feature_name: Optional[str] = None,
        cutoff: Optional[float] = None,
    ):
        """
        Render molecules using the template image.
        
        This method is only for visualization purpose. Iso-surface will be calculated
        using the input template image and mapped to every molecule position. The input
        template image does not have to be the image used for subtomogram alignment.

        Parameters
        ----------
        layer : MonomerLayer
            Select which monomer layer to be used for subtomogram sampling.
        template_path : ip.ImgArray, optional
            Template image.
        mask_params : str or (float, float), optional
            Mask image path or dilation/Gaussian blur parameters. If a path is given,
            image must in the same shape as the template.
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
            with ip.silent(), set_gpu():
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
    
    @mark_preview(render_molecules)
    def _preview_rendering(self, template_path: str, mask_params, cutoff: float):
        from skimage.measure import marching_cubes
        # prepare template and mask
        template = self._subtomogram_averaging._get_template(template_path).copy()
        if cutoff is not None:
            with ip.silent(), set_gpu():
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
        view_surface([verts, faces], parent=self)
        return None
    
    @toolbar.wraps
    @set_design(icon_path=ICON_DIR/"pick_next.png")
    @bind_key("F3")
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
            drot = 5.0
            max_shifts = (stride_nm/tomo.scale) * np.tan(np.deg2rad(drot))
            centering(img_next, point2, angle_deg, drot=drot, max_shifts=max_shifts)
            
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
    @bind_key("F4")
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
                angle_deg = angle_corr(img_input, ang_center=0, drot=89.5, nrots=31)
                centering(img_input, point, angle_deg, drot=3, nrots=7)
                last_i = i
        
        self.layer_work.data = points * imgb.scale.x
        if len(selected) == 1:
            change_viewer_focus(self.parent_viewer, points[last_i], imgb.scale.x)
        return None
    
    @Image.wraps
    @set_design(text="Paint MT")
    def paint_mt(self):
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
    @set_design(text="Set colormap")
    def set_colormap(
        self,
        start: Color = (0, 0, 1, 1),
        end: Color = (1, 0, 0, 1),
        limit: Tuple[float, float] = (4.00, 4.24), 
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
    
    @Image.wraps
    @set_design(text="Show color-bar")
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
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Non-GUI methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
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
    def get_loader(
        self,
        name: str = None,
        order: int = 1,
        chunksize: int = 64
    ) -> SubtomogramLoader:
        """
        Create a subtomogram loader using current tomogram and a molecules layer.

        Parameters
        ----------
        name : str, optional
            Name of the molecules layer.
        order : int, default is 1
            Interpolation order of the subtomogram loader.
        chunksize : int, default is 64
            Chunk size of the subtomogram loader.
        """        
        mole = self.get_molecules(name)
        shape = self._subtomogram_averaging._get_shape_in_nm()
        loader = self.tomogram.get_subtomogram_loader(
            mole, shape, order=order, chunksize=chunksize
        )
        return loader
    
    @nogui
    @do_not_record
    def get_spline(self, i: int = None) -> MtSpline:
        """Get a spline object"""
        tomo = self.tomogram
        if i is None:
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
    
    @refine_splines.returned.connect
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
    
    @open_image.returned.connect
    def _send_tomogram_to_viewer(self, _=None):
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
        with ip.silent():
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
        """Return local Cartesian image at the current position."""
        i = i or self.SplineControl.num
        j = j or self.SplineControl.pos
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
        """Return cylindric-transformed image at the current position"""
        i = i or self.SplineControl.num
        j = j or self.SplineControl.pos
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
    
    def _on_layer_removing(self, event):
        # NOTE: To make recorded macro completely reproducible, removing molecules 
        # from the viewer layer list must be always monitored.
        layer: Layer = self.parent_viewer.layers[event.index]
        if MOLECULES in layer.metadata.keys():
            expr = mk.Mock(mk.symbol(self)).parent_viewer.layers[layer.name].expr
            self.macro.append(mk.Expr("del", [expr]))
        return
    
    def _init_layers(self):
        viewer: napari.Viewer = self.parent_viewer
        viewer.layers.events.removing.disconnect(self._on_layer_removing)
        
        # remove all the molecules layers
        _layers_to_remove: List[str] = []
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
        
        viewer.layers.events.removing.connect(self._on_layer_removing)
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
        self.layer_prof.face_color = "black"
        self.layer_prof.face_color[spec] = [0.8, 0.0, 0.5, 1]
        self.layer_prof.refresh()
        return None
    
    @global_ft_analysis.returned.connect
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
    
    def _add_spline_to_images(self, spl: MtSpline, i: int):
        interval = 15
        length = spl.length()
        scale = self.layer_image.scale[0]
        
        n = int(length/interval) + 1
        fit = spl(np.linspace(0, 1, n))
        self.layer_prof.feature_defaults[SPLINE_ID] = i
        self.layer_prof.add(fit)
        self.overview.add_curve(
            fit[:, 2]/scale, fit[:, 1]/scale, color="lime", lw=2, name=f"spline-{i}",)
        return None
    
    @fit_splines.returned.connect
    @refine_splines.returned.connect
    @run_mtprops.yielded.connect
    @run_mtprops.returned.connect
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
    
    @mark_preview(load_project)
    def _preview_text(self, path: str):
        view_text(path, parent=self)
    
    @mark_preview(load_molecules)
    def _preview_table(self, paths: List[str]):
        view_tables(paths, parent=self)
    
    @mark_preview(open_image)
    def _preview_image(self, path: str):
        view_image(path, parent=self)
        

def centering(
    imgb: ip.ImgArray,
    point: np.ndarray,
    angle: float,
    drot: float = 5, 
    nrots: int = 7,
    max_shifts: int = None,
):
    angle_deg2 = angle_corr(imgb, ang_center=angle, drot=drot, nrots=nrots)
    
    img_next_rot = imgb.rotate(-angle_deg2, cval=np.mean(imgb))
    proj = img_next_rot.proj("y")
    shift = mirror_zncc(proj, max_shifts=max_shifts)
    
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

def _plot_seam_search_result(corrs: np.ndarray, score: np.ndarray, npf: int):
    imax = np.argmax(score)
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

def _plot_fsc(
    freq: np.ndarray,
    fsc_mean: np.ndarray,
    fsc_std: np.ndarray,
    crit: List[float],
    scale: nm,
):
    ind = (freq <= 0.7)
    plt.axhline(0.0, color="gray", alpha=0.5, ls="--")
    plt.axhline(1.0, color="gray", alpha=0.5, ls="--")
    for cr in crit:
        plt.axhline(cr, color="violet", alpha=0.5, ls="--")
    plt.plot(freq[ind], fsc_mean[ind], color="gold")
    plt.fill_between(
        freq[ind],
        y1=fsc_mean[ind] - fsc_std[ind],
        y2=fsc_mean[ind] + fsc_std[ind],
        color="gold",
        alpha=0.3
    )
    plt.xlabel("Spatial frequence (1/nm)")
    plt.ylabel("FSC")
    plt.ylim(-0.1, 1.1)
    xticks = np.linspace(0, 0.7, 8)
    per_nm = [r"$\infty$"] + [f"{x:.2f}" for x in scale / xticks[1:]]
    plt.xticks(xticks, per_nm)
    plt.tight_layout()
    plt.show()

def _calc_resolution(
    freq: np.ndarray,
    fsc: np.ndarray,
    crit: float = 0.143,
    scale: nm = 1.0
) -> nm:
    """
    Calculate resolution using arrays of frequency and FSC.
    This function uses linear interpolation to find the solution.
    If the inputs are not accepted, 0 will be returned.
    """
    freq0 = None
    for i, fsc1 in enumerate(fsc):
        if fsc1 < crit:
            if i == 0:
                resolution = 0
                break
            f0 = freq[i-1]
            f1 = freq[i]
            fsc0 = fsc[i-1]
            freq0 = (crit - fsc1)/(fsc0 - fsc1) * (f0 - f1) + f1
            resolution = scale / freq0
            break
    else:
        resolution = 0
    return resolution
