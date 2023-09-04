from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import impy as ip
from magicclass import magicclass, field
from magicclass.box import resizable
from magicclass.ext.vispy import Vispy3DCanvas, VispyImageCanvas
from cylindra.const import PropertyNames as H, nm, Mode
from cylindra.utils import map_coordinates
from cylindra.components import CylSpline
from cylindra.cyltransform import CylindricTransformer
from ._child_widget import ChildWidget

if TYPE_CHECKING:
    from cylindra.components import CylTomogram


@magicclass(record=False)
class FilamentFilter(ChildWidget):
    canvas = resizable(field(Vispy3DCanvas))

    def __init__(self) -> None:
        self._spl = None
        self._img_straight = None
        self._image_layer = None

    def _set_components(self, tomo: CylTomogram, idx: int, bin_size: int = 1):
        self._spl = tomo.splines[idx]
        self._img_straight = tomo.straighten(idx, binsize=bin_size)
        self._image_layer = self.canvas.add_image(self._img_straight.value)

    def _filter_cylindric(self):
        cp = self._spl.cylinder_params()
        shape = self._img_straight.shape
        size_y = shape.y * self._img_straight.scale.y
        size_a = shape.a * self._img_straight.scale.a
        vy = cp.pitch / size_y
        vx = cp.pitch / cp.tan_rise / size_y
        hx = cp.lat_spacing_proj / size_a
        hy = cp.lat_spacing_proj * cp.tan_skew_tilt / size_a
        v = np.array([vy, vx])
        h = np.array([hy, hx])

    def _straight_spline(self) -> CylSpline:
        shape = self._img_straight.shape
        center = (np.array(shape) - 1) / 2
        start = np.array([center[0], 0, center[2]])
        end = np.array([center[0], shape[1] - 1, center[2]])
        scale = self._img_straight.scale.x
        start_nm = start * scale
        end_nm = end * scale
        return CylSpline.line(start_nm, end_nm)

    def _transform_cylindric(self):
        rmin_nm, rmax_nm = self._spl.radius_range()
        scale = self._img_straight.scale.x
        rmin, rmax = rmin_nm / scale, rmax_nm / scale
        coords = self._straight_spline().cylindrical((rmin, rmax), scale=scale)
        rc = (rmin_nm + rmax_nm) / 2
