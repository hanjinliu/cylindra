import os
from typing import Union, List, Tuple, TYPE_CHECKING
from magicclass import (
    magicclass, magicmenu, magictoolbar, do_not_record, field, vfield, MagicTemplate, 
    set_options, set_design, abstractapi
)
from magicclass.widgets import Separator, HistoryFileEdit
from magicclass.types import OneOf, SomeOf, Optional
from pathlib import Path
from superqt import ensure_main_thread
import numpy as np
import impy as ip
import napari

from .widget_utils import FileFilter
from .global_variables import GlobalVariables, GVar
from ._previews import view_image

from cylindra.const import nm
from cylindra.utils import pad_template, roundint, normalize_image, ceilint
from cylindra.ext.etomo import PEET

if TYPE_CHECKING:
    from napari.layers import Image

ICON_DIR = Path(__file__).parent / "icons"

# Menus

@magicmenu
class File(MagicTemplate):
    """File I/O."""  
    open_image_loader = abstractapi()
    load_project = abstractapi()
    load_splines = abstractapi()
    load_molecules = abstractapi()
    sep0 = field(Separator)
    save_project = abstractapi()
    save_spline = abstractapi()
    save_molecules = abstractapi()
    sep1 = field(Separator)
    process_images = abstractapi()
    view_project = abstractapi()
    PEET = PEET

@magicmenu
class Image(MagicTemplate):
    """Image processing and visualization"""
    show_image_info = abstractapi()
    filter_reference_image = abstractapi()
    add_multiscale = abstractapi()
    set_multiscale = abstractapi()
    sep0 = field(Separator)
    open_sweeper = abstractapi()
    open_simulator = abstractapi()
    sep1 = field(Separator)
    sample_subtomograms = abstractapi()
    paint_cylinders = abstractapi()
    set_colormap = abstractapi()
    show_colorbar = abstractapi()

@magicmenu
class Splines(MagicTemplate):
    """Operations on splines"""
    show_splines = abstractapi()
    add_anchors = abstractapi()
    sep0 = field(Separator)
    invert_spline = abstractapi()
    align_to_polarity = abstractapi()
    clip_spline = abstractapi()
    delete_spline = abstractapi()
    sep1 = field(Separator)
    fit_splines = abstractapi()
    fit_splines_manually = abstractapi()
    refine_splines = abstractapi()
    molecules_to_spline = abstractapi()

@magicmenu
class Molecules_(MagicTemplate):
    """Operations on molecules"""
    @magicmenu
    class Mapping(MagicTemplate):
        map_monomers = abstractapi()
        map_centers = abstractapi()
        map_along_pf = abstractapi()
    show_orientation = abstractapi()
    extend_molecules = abstractapi()
    concatenate_molecules = abstractapi()
    calculate_intervals = abstractapi()
    sep0 = field(Separator)
    open_feature_control = abstractapi()
    
@magicmenu
class Analysis(MagicTemplate):
    """Analysis of tomograms."""
    set_radius = abstractapi()
    local_ft_analysis = abstractapi()
    global_ft_analysis = abstractapi()
    sep0 = field(Separator)
    open_subtomogram_analyzer = abstractapi()

@magicmenu
class Others(MagicTemplate):
    """Other menus."""
    @magicmenu
    class Macro:
        show_macro = abstractapi()
        show_full_macro = abstractapi()
        show_native_macro = abstractapi()
        sep0 = field(Separator)
        run_file = abstractapi()
    Global_variables = GlobalVariables
    open_logger = abstractapi()
    clear_cache = abstractapi()
    send_ui_to_console = abstractapi()
    @magicmenu
    class Help(MagicTemplate):
        open_help = abstractapi()
        cylindra_info = abstractapi()
        report_issues = abstractapi()

# Toolbar

@magictoolbar(labels=False)
class toolbar(MagicTemplate):
    """Frequently used operations."""        
    register_path = abstractapi()
    open_runner = abstractapi()
    sep0 = field(Separator)
    pick_next = abstractapi()
    auto_center = abstractapi()

    @magicmenu(icon=ICON_DIR/"adjust_intervals.png")
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
    clear_current = abstractapi()
    clear_all = abstractapi()

# STA widget

# TEMPLATE_CHOICES = ("From file", "From layer")
MASK_CHOICES = ("No mask", "Blur template", "From file")

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
    dilate_radius = vfield(0.3, options={"step": 0.1, "max": 20}, record=False)
    sigma = vfield(0.3, options={"step": 0.1, "max": 20}, record=False)
    
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
    
    template_path = vfield(HistoryFileEdit, label="Template", options={"filter": FileFilter.IMAGE}, record=False)
    mask = vfield(OneOf[MASK_CHOICES], label="Mask", record=False)
    params = field(params)
    mask_path = field(mask_path)
    tilt_range = vfield(Optional[Tuple[nm, nm]], label="Tilt range (deg)", options={"value": (-60., 60.), "text": "No missing-wedge", "options": {"options": {"min": -90.0, "max": 90.0, "step": 1.0}}}, record=False)
        
    @mask.connect
    def _on_mask_switch(self):
        v = self.mask
        self.params.visible = (v == MASK_CHOICES[1])
        self.mask_path.visible = (v == MASK_CHOICES[2])
    
    @ensure_main_thread(await_return=True)
    def _get_template(self, path: Union[Path, None] = None, rescale: bool = True) -> ip.ImgArray:
        if path is None:
            path = self.template_path
        else:
            self.template_path = path
        
        # check path
        if not os.path.exists(path) or not os.path.isfile(path):
            img = None
            s = str(path)
            if self._viewer is not None and s in self._viewer.layers:
                data = self._viewer.layers[s].data
                if isinstance(data, ip.ImgArray) and data.ndim == 3:
                    img: ip.ImgArray = data
            
            if img is None:
                raise FileNotFoundError(f"Path '{path}' is not a valid file.")
        
        else:
            img = ip.imread(path)
            
        if img.ndim != 3:
            raise TypeError(f"Template image must be 3-D, got {img.ndim}-D.")
        
        from .main import CylindraMainWidget
        parent = self.find_ancestor(CylindraMainWidget)
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
        from .main import CylindraMainWidget
        
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
        scale_ratio = mask_image.scale.x/self.find_ancestor(CylindraMainWidget).tomogram.scale
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
            self._viewer.window.add_dock_widget(volume_menu)
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
    @set_design(text="Show template")
    def show_template(self):
        """Load and show template image."""
        self._show_reconstruction(self._get_template(), name="Template image")
    
    @do_not_record
    @set_design(text="Show mask")
    def show_mask(self):
        """Load and show mask image."""
        self._show_reconstruction(self._get_mask(), name="Mask image")
    
    @magicmenu
    class Subtomogram_analysis(MagicTemplate):
        """Analysis of subtomograms."""
        average_all = abstractapi()
        average_subset = abstractapi()
        split_and_average = abstractapi()
        calculate_correlation = abstractapi()
        calculate_fsc = abstractapi()
        seam_search = abstractapi()
    
    @magicmenu
    class Refinement(MagicTemplate):
        """Refinement and alignment of subtomograms."""
        align_averaged = abstractapi()
        align_all = abstractapi()
        align_all_template_free = abstractapi()
        align_all_multi_template = abstractapi()
        align_all_viterbi = abstractapi()
        polarity_check = abstractapi()
    
    @magicmenu
    class Tools(MagicTemplate):
        """Other tools."""
        reshape_template = abstractapi()
        render_molecules = abstractapi()
    
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
        Interval of sampling points of cylinder fragments.
    ft_size: nm
        Longitudinal length of local discrete Fourier transformation used 
        for structural analysis.
    paint : bool
        Check if paint the tomogram with the local properties.
    """
    interval = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Interval (nm)"}, record=False)
    ft_size = vfield(32.0, options={"min": 1.0, "max": 200.0, "label": "Local DFT window size (nm)"}, record=False)
    paint = vfield(True, record=False)


@magicclass(name="Run cylindrical fitting")
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
        Check if cylindric structures are densely packed. Initial spline position
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
        from .main import CylindraMainWidget
        try:
            tomo = self.find_ancestor(CylindraMainWidget).tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]
    
    def _get_available_binsize(self, _=None) -> List[int]:
        from .main import CylindraMainWidget
        try:
            parent = self.find_ancestor(CylindraMainWidget)
        except Exception:
            return [1]
        if parent.tomogram is None:
            return [1]
        out = [x[0] for x in parent.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return sorted(out)
    
    all_splines = vfield(True, options={"text": "Run for all the splines."}, record=False)
    splines = vfield(SomeOf[_get_splines], options={"visible": False}, record=False)
    bin_size = vfield(OneOf[_get_available_binsize], record=False)
    dense_mode = vfield(True, options={"label": "Use dense-mode"}, record=False)
    params1 = runner_params1
    n_refine = vfield(2, options={"label": "Refinement iteration", "max": 4}, record=False)
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
    
    cylindrical_fit = abstractapi()

@magicclass(name="Open image")
class ImageLoader(MagicTemplate):
    """
    Load an image file and process it before sending it to the viewer.

    Attributes
    ----------
    path : Path
        Path to the tomogram. Must be 3-D image.
    bin_size : int or list of int, default is [1]
        Initial bin size of image. Binned image will be used for visualization in the viewer.
        You can use both binned and non-binned image for analysis.
    filter_reference_image : bool, default is True
        Apply low-pass filter on the reference image (does not affect image data itself).
    """
    path = vfield(Path, options={"filter": FileFilter.IMAGE}, record=False)
    
    @magicclass(layout="horizontal", labels=False)
    class scale(MagicTemplate):
        """
        Scale of the image.

        Attributes
        ----------
        scale_value : float
            Scale of the image in nm/pixel.
        """
        scale_label = vfield("scale (nm)", widget_type="Label")
        scale_value = vfield(1.0, options={"min": 0.001, "step": 0.0001, "max": 10.0, "label": "scale (nm)"}, record=False)
        read_header = abstractapi()
            
    bin_size = vfield([1], options={"options": {"min": 1, "max": 8}}, record=False)
    filter_reference_image = vfield(True, record=False)
    
    @scale.wraps
    @do_not_record
    def read_header(self):
        """Read scale from image header."""
        path = self.path
        if not os.path.exists(path) or not os.path.isfile(path):
            return
        img = ip.lazy_imread(path, chunks=GVar.daskChunk)
        scale = img.scale.x
        self.scale.scale_value = f"{scale:.4f}"
        if len(self.bin_size) < 2:
            self.bin_size = [ceilint(0.96 / scale)]
    
    open_image = abstractapi()
    
    @do_not_record
    @set_design(text="Preview")
    def preview_image(self):
        """Preview image at the path."""
        return view_image(self.path, parent=self)
