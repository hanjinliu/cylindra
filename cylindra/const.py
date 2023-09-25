from __future__ import annotations
from enum import Enum
from types import SimpleNamespace
import polars as pl

nm = float  # type alias for nanometer
pf = float  # type alias for protofilament numbering

# Constant strings for layer names and metadata keys
WORKING_LAYER_NAME = "Working Layer"
SELECTION_LAYER_NAME = "Splines"
ALN_SUFFIX = "ALN"
PREVIEW_LAYER_NAME = "<Preview>"
SPLINE_ID = "spline-id"


class strEnum(Enum):
    """Enum with string values"""

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


class Ori(strEnum):
    """Orientations"""

    none = "none"
    PlusToMinus = "PlusToMinus"
    MinusToPlus = "MinusToPlus"

    @classmethod
    def invert(cls, ori: Ori, allow_none: bool = True) -> Ori:
        """
        Invert orientation.

        Parameters
        ----------
        ori : Ori
            Ori object to be inverted.
        allow_none : bool, default is True
            If true, convert ``Ori.none`` into ``Ori.none``. Raise an error
            otherwise.

        Returns
        -------
        Ori
            Inverted ``Ori`` object.
        """
        if ori == Ori.PlusToMinus:
            out = Ori.MinusToPlus
        elif ori == Ori.MinusToPlus:
            out = Ori.PlusToMinus
        else:
            if allow_none:
                out = Ori.none
            else:
                raise ValueError(f"{ori} cannot be inverted.")
        return out


class ExtrapolationMode(strEnum):
    """Extrapolation mode for splines."""

    default = "default"
    linear = "linear"


class PropertyNames(SimpleNamespace):
    """Header names for result table of local properties."""

    spline_id = "spline-id"
    pos_id = "pos-id"
    spl_dist = "spl_dist"
    spl_pos = "spl_pos"
    rise = "rise_angle"
    rise_length = "rise_length"
    pitch = "pitch"
    spacing = "spacing"
    dimer_twist = "dimer_twist"
    skew = "skew_angle"
    npf = "npf"
    radius = "radius"
    orientation = "orientation"  # global only
    offset_radial = "offset_radial"  # global only
    offset_axial = "offset_axial"  # global only
    offset_angular = "offset_angular"  # global only
    start = "start"


class FileFilter(SimpleNamespace):
    """File dialog filter strings"""

    IMAGE = "Tomograms (*.mrc;*.rec;*.tif;*.tiff;*.map);;All files (*)"
    PNG = "PNG (*.png;*.jpg);;All files (*)"
    JSON = "JSON(*.json;*.txt);;All files (*)"
    PROJECT = "Project file (project.json;*.tar;*.zip);;All files (*)"
    CSV = "CSV (*.csv;*.txt;*.dat);;All files (*)"
    PY = "Python (*.py);;All files (*)"
    MOD = "Model files (*.mod);;All files (*.txt;*.csv)"


class Mode(SimpleNamespace):
    """Padding mode used in scipy.ndimage."""

    grid_wrap = "grid-wrap"
    wrap = "wrap"
    reflect = "reflect"
    mirror = "mirror"
    constant = "constant"
    nearest = "nearest"


class MoleculesHeader(SimpleNamespace):
    """Feature header names for Molecules."""

    nth = "nth"  # n-th molecule of a protofilament
    pf = "pf-id"  # protofilament ID
    isotype = "isotype-id"
    score = "score"  # alignment score
    interval = "interval-nm"  # interval between two molecules
    lateral_interval = "lateral-interval-nm"  # lateral interval between two molecules
    dimer_twist = "dimer-twist-deg"  # dimer twist between two dimers
    skew = "skew-deg"  # skew tilt angle between two molecules
    rise = "rise-angle-deg"  # rise angle between two molecules
    radius = "radius-nm"  # distance between the molecule and the spline
    lateral_angle = "lateral-angle-deg"  # lateral angle between molecules
    elev_angle = "elevation-angle-deg"  # elevation angle between molecules
    position = "position-nm"  # position of the molecule along the spline
    id = "molecules-id"
    image = "image-id"
    z = "z"
    y = "y"
    x = "x"
    zvec = "zvec"
    yvec = "yvec"
    xvec = "xvec"


class EulerAxes(strEnum):
    """Sequence of Euler angles."""

    xyz = "xyz"
    yzx = "yzx"
    zxy = "zxy"
    xzy = "xzy"
    yxz = "yxz"
    zyx = "zyx"
    xyx = "xyx"
    xzx = "xzx"
    yxy = "yxy"
    yzy = "yzy"
    zxz = "zxz"
    zyz = "zyz"
    XYZ = "XYZ"
    YZX = "YZX"
    ZXY = "ZXY"
    XZY = "XZY"
    YXZ = "YXZ"
    ZYX = "ZYX"
    XYX = "XYX"
    XZX = "XZX"
    YXY = "YXY"
    YZY = "YZY"
    ZXZ = "ZXZ"
    ZYZ = "ZYZ"


def get_versions() -> dict[str, str]:
    """Return version info of relevant libraries."""
    import napari
    import numpy as np
    import impy as ip
    import magicgui
    from cylindra import __version__
    import magicclass as mcls
    import dask

    return {
        "cylindra": __version__,
        "numpy": np.__version__,
        "impy": ip.__version__,
        "magicgui": magicgui.__version__,
        "magicclass": mcls.__version__,
        "napari": napari.__version__,
        "dask": dask.__version__,
    }


class SplineColor(SimpleNamespace):
    """Color of spline used in the 2D/3D canvas."""

    DEFAULT = [0.0, 0.0, 1.0, 1.0]
    SELECTED = [0.8, 0.0, 0.5, 1.0]


class ImageFilter(strEnum):
    """Available filters for the reference image."""

    Lowpass = "Lowpass"
    Gaussian = "Gaussian"
    DoG = "DoG"
    LoG = "LoG"


_POLARS_DTYPES = {
    PropertyNames.npf: pl.UInt8,
    PropertyNames.orientation: pl.Utf8,
    PropertyNames.spline_id: pl.UInt16,
    MoleculesHeader.image: pl.UInt16,
}


def cast_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Cast the dataframe to the appropriate dtype based on the columns."""
    out = list[pl.Expr]()
    for cname in df.columns:
        if dtype := _POLARS_DTYPES.get(cname, None):
            out.append(pl.col(cname).cast(dtype))
        elif df[cname].dtype in pl.FLOAT_DTYPES:
            out.append(pl.col(cname).cast(pl.Float32))
    return df.with_columns(out)
