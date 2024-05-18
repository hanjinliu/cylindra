from __future__ import annotations

from typing import Callable, NamedTuple, TypeVar

# This file is for documentation and tooltip purposes only.

_T = TypeVar("_T", bound=type)


class Parameter(NamedTuple):
    """
    Structure for a parameter description. A Parameter tuple is equivalent to the following.

    _name_ : _type_
        _desc_
    """

    name: str
    type: str
    desc: str

    def to_string(self, indent: int = 3) -> str:
        _indent1 = "    " * indent
        desc = self.desc.replace("\n", f"\n{_indent1}")
        return f"{self.name} : {self.type}\n{_indent1}{desc}"


# fmt: off
_PARAMETERS = [
    Parameter(name="layer", type="MoleculesLayer", desc="Points layer of molecules to be used."),
    Parameter(name="layers", type="list of MoleculesLayer", desc="All the points layers of molecules to be used."),
    Parameter(name="loader_name", type="str", desc="Name of the batch subtomogram loader to be used."),
    Parameter(name="landscape_layer", type="LandscapeSurface", desc="Landscape layer to be used in this algorithm."),
    Parameter(name="template_path", type="Path or str", desc="Path to template image."),
    Parameter(name="mask_params", type="str or (float, float), optional", desc="Mask image path or dilation/Gaussian blur parameters.\nIf a path is given, image must in the same shape as the template."),
    Parameter(name="tilt_range", type="(float, float), optional", desc="Tilt range of tomogram tilt series in degree."),
    Parameter(name="max_shifts", type="int or tuple of int", desc="Maximum shift between subtomograms and template in nm. ZYX order."),
    Parameter(name="rotations", type="((float, float), (float, float), (float, float))", desc="Rotation in external degree around each axis."),
    Parameter(name="cutoff", type="float", desc="Cutoff frequency of low-pass filter applied in each subtomogram."),
    Parameter(name="interpolation", type="int", desc="Interpolation order."),
    Parameter(name="size", type="nm", desc="Size of the template in nm. Use the size of template image by default."),
    Parameter(name="method", type="str", desc="Correlation metrics for alignment."),
    Parameter(name="spline", type="int", desc="Index of splines to be used."),
    Parameter(name="splines", type="list of int", desc="Indices of splines to be used."),
    Parameter(name="bin_size", type="int", desc="Bin size of multiscale image to be used. Set to >1 to boost performance."),
    Parameter(name="interval", type="nm", desc="Interval (nm) between spline anchors. Please note that resetting interval will discard\nall the existing local properties."),
    Parameter(name="depth", type="nm", desc="Depth (length parallel to the spline tangent) of the subtomograms to be sampled."),
    Parameter(name="max_interval", type="nm", desc="Maximum interval (nm) between spline anchors."),
    Parameter(name="molecule_interval", type="nm", desc="Interval (nm) between molecules.\n`col` is available in this namespace to refer to the spline global properties.\nFor example, `col('spacing') * 2` means twice the spacing of the spline."),
    Parameter(name="orientation", type="None, 'PlusToMinus', 'MinusToPlus'", desc="Orientation of molecules' y-axis. If none, use the\ncurrent spline orientation as is."),
    Parameter(name="offsets", type="(float, float), optional", desc="Offset values that will be used to define molecule positions."),
    Parameter(name="filter", type="ImageFilter", desc="Filter to be applied to the reference image. This does not affect the image data itself.\n\n- Lowpass: butterworth low-pass filter.\n- Gaussian: Gaussian blur.\n- DoG: difference of Gaussian.\n- LoG: Laplacian of Gaussian."),
    Parameter(name="inherit_source", type="bool", desc="If True and the input molecules layer has its spline source, the new layer will inherit it."),
    Parameter(name="color_by", type="str", desc="Name of the feature to paint by."),
    Parameter(name="cmap", type="colormap", desc="Colormap to be used for painting."),
    Parameter(name="limits", type="(float, float)", desc="Lower and upper limits of the colormap."),
    Parameter(name="upsample_factor", type="int", desc="Upsampling factor of ZNCC landscape. Be careful not to set this parameter too large. \nCalculation will take much longer for larger ``upsample_factor``."),
    Parameter(name="range_long", type="(float, float)", desc="Minimum and maximum allowed distances between longitudinally consecutive monomers"),
    Parameter(name="range_lat", type="(float, float)", desc="Minimum and maximum allowed distances between laterally consecutive monomers"),
    Parameter(name="angle_max", type="float", desc="Maximum allowed angle between longitudinally consecutive monomers and the Y axis."),
    Parameter(name="temperature_time_const", type="float", desc="Time constant of the temperature decay during annealing. Larger value results in slower annealing. 1.0 is a moderate value."),
    Parameter(name="random_seeds", type="iterable of int", desc="Random seed integers. Number of integers will be the number of trials."),
    Parameter(name="target", type="str", desc="Target column name on which calculation will run."),
    Parameter(name="footprint", type="array-like", desc="2D binary array that define the convolution kernel structure."),
    Parameter(name="projective", type="bool", desc="If true, only the vector components parallel to the cylinder surface will be considered."),
    Parameter(name="err_max", type="float", desc="S.D. allowed for spline fitting. Larger value will result in smoother spline, i.e. fewer spline knots."),
    Parameter(name="update_glob", type="bool", desc="If true, also update the global property to the mean of local properties."),
    Parameter(name="min_radius", type="nm", desc="Minimum possible radius in nm."),
    Parameter(name="max_radius", type="nm", desc="Maximum possible radius in nm."),
    Parameter(name="prefix", type="str", desc="Prefix of the new molecules layer(s)."),
]
# fmt: on

_TRANSLATION_MAP = {param.name: param.to_string() for param in _PARAMETERS}
assert len(_TRANSLATION_MAP) == len(_PARAMETERS)  # check duplication


def update_doc(doc: str, indent: int = 2) -> str:
    """Update docstring"""
    ind = "    " * indent
    doc = doc.replace("}{", "}\n" + ind + "{")
    out = doc.format(**_TRANSLATION_MAP)
    if indent < 2:  # only used for mkdocs
        out = out.replace("\n" + (2 - indent) * "    ", "\n")
    return out


def update_func(f: Callable):
    """Update the __doc__ of a function."""
    if doc := f.__doc__:
        f.__doc__ = update_doc(doc)
    return f


def update_cls(cls: _T) -> _T:
    """Update the __doc__ of all methods in a class."""
    for attr in cls.__dict__.values():
        if callable(attr) and hasattr(attr, "__doc__"):
            update_func(attr)
    return cls
