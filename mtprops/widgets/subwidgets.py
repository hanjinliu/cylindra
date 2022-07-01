import os
from typing import Union, List, Tuple, TYPE_CHECKING
from magicclass import (
    magicclass, magicmenu, magictoolbar, do_not_record, field, vfield, MagicTemplate, 
    set_options, set_design
)
from magicclass.widgets import Separator
from magicclass.types import OneOf, SomeOf, Optional
from pathlib import Path
import numpy as np
import impy as ip
import napari

from .widget_utils import FileFilter
from .global_variables import GlobalVariables

from ..const import nm
from ..utils import pad_template, roundint, normalize_image
from ..ext.etomo import PEET

if TYPE_CHECKING:
    from napari.layers import Image

ICON_DIR = Path(__file__).parent / "icons"

# Menus

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
    def delete_spline(self): ...
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
    def concatenate_molecules(self): ...
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
    def send_ui_to_console(self): ...
    @magicmenu
    class Help(MagicTemplate):
        def open_help(self): ...
        def MTProps_info(self): ...
        def report_issues(self): ...

# Toolbar

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
        """
        Adjust auto picker parameters.
        
        Attributes
        ----------
        stride : nm
            Stride length (nm) of auto picker.
        angle_deviation : float
            Angle deviation (degree) of auto picker.
        angle_precision : float
            Angle precision (degree) of auto picker.
        max_shifts : nm
            Maximum shift (nm) in auto centering.
        """
        stride = vfield(50.0, widget_type="FloatSlider", options={"min": 10, "max": 100}, record=False)
        angle_deviation = vfield(12.0, widget_type="FloatSlider", options={"min": 1.0, "max": 40.0, "step": 0.5}, record=False)
        angle_precision = vfield(1.0, widget_type="FloatSlider", options={"min": 0.5, "max": 5.0, "step": 0.1}, record=False)
        max_shifts = vfield(20.0, options={"min": 1., "max": 50., "step": 0.5}, record=False)
    sep1 = field(Separator)
    def clear_current(self): ...
    def clear_all(self): ...

# STA widget

MASK_CHOICES = ("No mask", "Use blurred template as a mask", "Supply a image")

@magicclass(layout="horizontal", widget_type="groupbox", name="Parameters", visible=False)
class params(MagicTemplate):
    """
    Parameters for soft mask creation.
    
    Soft mask creation has three steps. 
    (1) Create binary mask by applying thresholding to the template image.
    (2) Morphological dilation of the binary mask.
    (3) Gaussian filtering the mask.

    Attributes
    ----------
    dilate_radius : nm
        Radius of dilation (nm) applied to binarized template.
    sigma : nm
        Standard deviation (nm) of Gaussian blur applied to the edge of binary image.
    """
    dilate_radius = vfield(1.0, options={"step": 0.5, "max": 20}, record=False)
    sigma = vfield(1.0, options={"step": 0.5, "max": 20}, record=False)
    
@magicclass(layout="horizontal", widget_type="frame", visible=False)
class mask_path(MagicTemplate):
    """Path to the mask image."""
    mask_path = vfield(Path, options={"filter": FileFilter.IMAGE}, record=False)

@magicclass(name="Subtomogram averaging")
class SubtomogramAveraging(MagicTemplate):
    """
    Widget for subtomogram averaging.
    
    Attributes
    ----------
    template_path : Path
        Path to the template (reference) image file, or layer name of reconstruction.
    mask : str
        Select how to create a mask.
    tilt_range : tuple of float, options
        Tilt range (degree) of the tomogram.
    """
    def __post_init__(self):
        self._template = None
        self._viewer: Union[napari.Viewer, None] = None
        self._next_layer_name = None
        self.mask = MASK_CHOICES[0]

    template_path = vfield(Path, label="Template", options={"filter": FileFilter.IMAGE}, record=False)
    mask = vfield(OneOf[MASK_CHOICES], label="Mask", record=False)
    params = field(params)
    mask_path = field(mask_path)
    tilt_range = vfield(Optional[Tuple[nm, nm]], label="Tilt range (deg)", options={"value": (-60., 60.), "text": "No missing-wedge", "options": {"options": {"min": -90.0, "max": 90.0, "step": 1.0}}}, record=False)
    
    @mask.connect
    def _on_switch(self):
        v = self.mask
        self.params.visible = (v == MASK_CHOICES[1])
        self.mask_path.visible = (v == MASK_CHOICES[2])
    
    def _get_template(self, path: Union[Path, None] = None, rescale: bool = True) -> ip.ImgArray:
        if path is None:
            path = self.template_path
        else:
            self.template_path = path
        
        # check path
        if not os.path.exists(path) or not os.path.isfile(path):
            # BUG: using other viewer from other thread may be forbidden.
            # img = None
            # s = str(path)
            # if self._viewer is not None and s in self._viewer.layers:
            #     data = self._viewer.layers[s].data
            #     if isinstance(data, ip.ImgArray) and data.ndim == 3:
            #         img: ip.ImgArray = data
            
            # if img is None:
            raise FileNotFoundError(f"Path '{path}' is not a valid file.")
        
        else:
            img = ip.imread(path)
            
        if img.ndim != 3:
            raise TypeError(f"Template image must be 3-D, got {img.ndim}-D.")
        
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        if parent.tomogram is not None and rescale:
            scale_ratio = img.scale.x / parent.tomogram.scale
            if scale_ratio < 0.99 or 1.01 < scale_ratio:
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
        from .main import MTPropsWidget
        
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
        
    def _show_reconstruction(self, image: ip.ImgArray, name: str) -> "Image":
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
        input_image = normalize_image(image)
        from skimage.filters.thresholding import threshold_yen
        thr = threshold_yen(input_image.value)
        layer = self._viewer.add_image(
            input_image, scale=image.scale, name=name,
            rendering="iso", iso_threshold=thr,
        )
        
        return layer
    
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
        """Analysis of subtomograms."""
        def average_all(self): ...
        def average_subset(self): ...
        def split_and_average(self): ...
        def calculate_correlation(self): ...
        def calculate_fsc(self): ...
        def seam_search(self): ...
    
    @magicmenu
    class Refinement(MagicTemplate):
        """Refinement and alignment of subtomograms."""
        def align_averaged(self): ...
        def align_all(self): ...
        def align_all_template_free(self): ...
        def align_all_multi_template(self): ...
        def align_all_viterbi(self): ...
        def polarity_check(self): ...
        def polarity_check_fast(self): ...
    
    @magicmenu
    class Tools(MagicTemplate):
        """Other tools."""
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
        reshaped = pad_template(template, shape)
        reshaped.imsave(save_as)
        if update_template_path:
            self.template_path = save_as
        return None


# Runner

@magicclass(widget_type="groupbox", name="Parameters")
class runner_params1:
    """
    Parameters used in spline fitting.
    
    Attributes
    ----------
    edge_sigma : nm
        Sharpness of dense-mode mask at the edges.
    max_shift : nm
        Maximum shift in nm of manually selected spline to the true center.
    """
    edge_sigma = vfield(2.0, options={"label": "edge sigma"}, record=False)
    max_shift = vfield(5.0, options={"label": "Maximum shift (nm)", "max": 50.0, "step": 0.5}, record=False)


@magicclass(widget_type="groupbox", name="Parameters")
class runner_params2:
    """
    Parameters used in calculation of local properties.
    
    Attributes
    ----------
    interval : nm
        Interval of sampling points of microtubule fragments.
    ft_size: nm
        Longitudinal length of local discrete Fourier transformation used 
        for structural analysis.
    paint : bool
        Check if paint the tomogram with the local properties.
    """
    interval = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Interval (nm)"}, record=False)
    ft_size = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Local DFT window size (nm)"}, record=False)
    paint = vfield(True, record=False)


@magicclass(name="Run MTProps")
class Runner(MagicTemplate):
    """
    Attributes
    ----------
    all_splines : bool
        Uncheck to select along which spline algorithms will be executed.
    splines : list of int
        Splines that will be analyzed
    bin_size : int
        Set to >1 to use binned image for fitting.
    dense_mode : bool
        Check if microtubules are densely packed. Initial spline position
        must be 'almost' fitted in dense mode.
    n_refine : int
        Iteration number of spline refinement.
    local_props : bool
        Check if calculate local properties.
    global_props : bool
        Check if calculate global properties.
    """
    def _get_splines(self, _=None) -> List[Tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        from .main import MTPropsWidget
        try:
            tomo = self.find_ancestor(MTPropsWidget).tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]
    
    def _get_available_binsize(self, _=None) -> List[int]:
        from .main import MTPropsWidget
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
    
    all_splines = vfield(True, options={"text": "Run for all the splines."}, record=False)
    splines = vfield(SomeOf[_get_splines], options={"visible": False}, record=False)
    bin_size = vfield(OneOf[_get_available_binsize], record=False)
    dense_mode = vfield(True, options={"label": "Use dense-mode"}, record=False)
    params1 = runner_params1
    n_refine = vfield(1, options={"label": "Refinement iteration", "max": 4}, record=False)
    local_props = vfield(True, options={"label": "Calculate local properties"}, record=False)
    params2 = runner_params2
    global_props = vfield(True, label="Calculate global properties", record=False)

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
