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
        return f"{self.name} : {self.type}\n{_indent1}{self.desc}"


# fmt: off
_PARAMETERS = [
    Parameter(name="layer", type="MoleculesLayer", desc="Points layer of molecules to be used."),
    Parameter(name="layers", type="list of MoleculesLayer", desc="All the points layers of molecules to be used."),
    Parameter(name="template_path", type="Path or str", desc="Path to template image."),
    Parameter(name="mask_params", type="str or (float, float), optional", desc="Mask image path or dilation/Gaussian blur parameters. If a path is given, image must in the same shape as the template."),
    Parameter(name="tilt_range", type="(float, float), optional", desc="Tilt range of tomogram tilt series in degree."),
    Parameter(name="max_shifts", type="int or tuple of int", desc="Maximum shift between subtomograms and template in nm. ZYX order."),
    Parameter(name="z_rotation", type="(float, float), optional", desc="Rotation in external degree around z-axis."),
    Parameter(name="y_rotation", type="(float, float), optional", desc="Rotation in external degree around y-axis."),
    Parameter(name="x_rotation", type="(float, float), optional", desc="Rotation in external degree around x-axis."),
    Parameter(name="cutoff", type="float", desc="Cutoff frequency of low-pass filter applied in each subtomogram."),
    Parameter(name="interpolation", type="int", desc="Interpolation order."),
    Parameter(name="size", type="nm", desc="Size of the template in nm. Use the size of template image by default."),
    Parameter(name="method", type="str", desc="Correlation metrics for alignment."),
    Parameter(name="spline", type="int", desc="Index of splines to be used."),
    Parameter(name="splines", type="list of int", desc="Indices of splines to be used."),
    Parameter(name="bin_size", type="int", desc="Bin size of multiscale image to be used. Set to >1 to boost performance."),
    Parameter(name="interval", type="nm", desc="Interval (nm) between spline anchors. Please note that resetting interval will discard all the existing local properties."),
    Parameter(name="depth", type="nm", desc="Depth (length parallel to the spline tangent) of the subtomograms to be sampled."),
    Parameter(name="max_interval", type="nm", desc="Maximum interval (nm) between spline anchors."),
    Parameter(name="molecule_interval", type="nm", desc="Interval (nm) between molecules."),
    Parameter(name="orientation", type="None, 'PlusToMinus', 'MinusToPlus'", desc="Orientation of molecules' y-axis. If none, use the current spline orientation as is."),
    Parameter(name="offsets", type="(float, float), optional", desc="Offset values that will be used to define molecule positions."),
    Parameter(name="filter", type="ImageFilter", desc="Filter to be applied to the reference image. This does not affect the image data itself. Lowpass: butterworth low-pass filter. Gaussian: Gaussian blur. DoG: difference of Gaussian. LoG: Laplacian of Gaussian."),
]
# fmt: on

_TRANSLATION_MAP = {param.name: param.to_string() for param in _PARAMETERS}


def update_doc(f: Callable):
    """Update the __doc__ of a function."""
    doc = f.__doc__
    if doc:
        doc = doc.replace("}{", "}\n        {")
        f.__doc__ = doc.format(**_TRANSLATION_MAP)
    return f


def update_cls(cls: _T) -> _T:
    """Update the __doc__ of all methods in a class."""
    for attr in cls.__dict__.values():
        if callable(attr) and hasattr(attr, "__doc__"):
            update_doc(attr)
    return cls
