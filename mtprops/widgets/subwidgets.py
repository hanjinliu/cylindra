from magicclass import magicclass, vfield, MagicTemplate
from pathlib import Path
from .widget_utils import FileFilter

# STA widget

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

# runner

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
