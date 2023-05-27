from __future__ import annotations
from enum import Enum
from types import SimpleNamespace
from pathlib import Path

from psygnal import EventedModel
from psygnal._evented_model import EventedModel

from appdirs import user_config_dir

nm = float  # type alias for nanometer
pf = float  # type alias for protofilament numbering

# Constant strings for layer names and metadata keys
WORKING_LAYER_NAME = "Working Layer"
SELECTION_LAYER_NAME = "Splines"
ALN_SUFFIX = "ALN"


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

    splDist = "splDistance"
    splPos = "splPosition"
    rise = "riseAngle"
    spacing = "spacing"
    skew = "skewAngle"
    nPF = "nPF"
    radius = "radius"
    orientation = "orientation"
    start = "start"


class IDName(SimpleNamespace):
    """ID names used in local properties."""

    spline = "SplineID"
    pos = "PosID"


class Mode(SimpleNamespace):
    """Padding mode used in scipy.ndimage."""

    grid_wrap = "grid-wrap"
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
    skew = "skew-deg"  # skew angle between two molecules
    radius = "radius-nm"  # distance between the molecule and the spline
    lateral_angle = (
        "lateral-angle-deg"  # lateral angle between the molecule and the spline
    )
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


class GlobalVariableModel(EventedModel):
    """Global variables used in this module."""

    npf_min: int = 11
    npf_max: int = 17
    spline_degree: int = 3
    spacing_min: nm = 3.9
    spacing_max: nm = 4.3
    skew_min: float = -1.0
    skew_max: float = 1.0
    min_curvature_radius: nm = 400.0
    deconv_range: int = 2
    clockwise: str = "MinusToPlus"
    thickness_inner: float = 2.0
    thickness_outer: float = 3.0
    fit_depth: nm = 48.0
    fit_width: nm = 44.0
    dask_chunk: tuple[int, int, int] = (256, 256, 256)
    point_size: float = 4.2
    use_gpu: bool = True

    def update(self, values: EventedModel | dict, recurse: bool = True) -> None:
        # validate values
        Inf = float("inf")
        if isinstance(values, dict):
            if values.get("npf_min", -Inf) >= values.get("npf_max", Inf):
                raise ValueError(f"npf_min > npf_max must be satisfied.")
            if values.get("spacing_min", -Inf) >= values.get("spacing_max", Inf):
                raise ValueError(f"spacing_min > spacing_max must be satisfied.")
            if values.get("skew_min", -Inf) >= values.get("skew_max", Inf):
                raise ValueError(f"skew_min > skew_max must be satisfied.")
            if values.get("deconv_range", Inf) < 0:
                raise ValueError(f"deconv_range must be >= 0.")
        # In psygnal==0.9.0, events are paused (i.e., each signal will be emitted one
        # by one). This is not desirable because min/max values should be updated at
        # the same time. Therefore, we block the events and emit the signal manually.
        with self.events.blocked():
            super().update(values, recurse)
        self.events.emit(self.dict())
        return None


GlobalVariables = GlobalVariableModel()


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


class ConfigConst(SimpleNamespace):
    VAR_PATH = Path(user_config_dir("variables", "cylindra"))
    SETTINGS_PATH = Path(user_config_dir("settings", "cylindra"))
    USER_SETTINGS_NAME = "user-settings.json"
    DEFAULT_VARIABLES = "default_variables"


class SplineColor(SimpleNamespace):
    """Color of spline used in the 2D/3D canvas."""

    DEFAULT = [0.0, 0.0, 1.0, 1.0]
    SELECTED = [0.8, 0.0, 0.5, 1.0]
