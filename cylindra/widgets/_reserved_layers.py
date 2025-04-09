from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakSet

import impy as ip
import numpy as np
from napari.layers import Image, Layer, Points, Shapes

from cylindra.const import (
    SELECTION_LAYER_NAME,
    SPLINE_ID,
    WORKING_LAYER_NAME,
    Ori,
    SplineColor,
)
from cylindra.widgets._main_utils import fast_percentile

if TYPE_CHECKING:
    from cylindra.components import CylSpline


class ReservedLayers:
    """A class that stores layers reserved for a cylindra session."""

    def __init__(self):
        self.image = Image(np.zeros((1, 1, 1)))
        self.prof = _prof_layer()
        self.work = _work_layer()
        self.highlight = Points(
            ndim=3,
            name="Highlight",
            face_color="transparent",
            border_color="crimson",
            border_width=0.16,
            border_width_is_relative=True,
            out_of_slice_display=True,
            blending="translucent_no_depth",
        )
        self.plane = Shapes(
            ndim=3,
            name="Picker Plane",
            face_color=[1.0, 0.2, 0.2, 0.44],
            edge_color=[1.0, 0.2, 0.2, 0.84],
        )
        self.highlight.editable = False
        self.plane.editable = False
        self.to_be_removed = WeakSet[Layer]()
        self.is_lazy = False

    def update_image(self, img: ip.ImgArray, tr: float):
        """Update the reserved image layer"""
        self.image.data = img
        self.image.scale = list(img.scale.values())
        self.image.name = img.name
        self.image.translate = [tr] * 3
        self.image.contrast_limits = _calc_contrast_limits(img)

    def reset_image(self, img: ip.ImgArray, tr: float):
        """Reset the reserved image layer"""
        self.image = Image(
            img,
            scale=list(img.scale.values()),
            name=img.name,
            translate=[tr, tr, tr],
            contrast_limits=_calc_contrast_limits(img),
            blending="translucent_no_depth",
        )
        self.image.bounding_box.points = False
        self.image.bounding_box.line_color = "#a0a0a0"

    def highlight_spline(self, i: int):
        """Highlight the current spline."""
        spec = self.prof.features[SPLINE_ID] == i
        self.prof.face_color = SplineColor.DEFAULT
        self.prof.face_color[spec] = SplineColor.SELECTED
        self.prof.refresh()

    def add_spline(self, i: int, spl: CylSpline):
        """Add spline sample data to the layer."""
        interval = 8.0
        length = spl.length()
        n = max(int(length / interval) + 1, 2)
        fit = spl.map(np.linspace(0, 1, n))
        self.prof.feature_defaults[SPLINE_ID] = i
        self.prof.add(fit)
        return fit

    def rescale_layers(self, factor: float):
        """Update the scale of the reserved layers."""
        self.image.scale = [s * factor for s in self.image.scale]
        self.image.translate = [t * factor for t in self.image.translate]
        self.prof.data = [d * factor for d in self.prof.data]
        self.work.data = self.work.data * factor

    @property
    def image_data(self) -> ip.ImgArray:
        return self.image.data

    @property
    def scale(self) -> float:
        """Scale of the reserved image layer."""
        return self.image.scale[-1]

    def select_spline(self, i: int, default: int):
        features = self.prof.features
        spline_id = features[SPLINE_ID]
        spec = spline_id != i
        old_data = self.prof.data
        self.prof.data = old_data[spec]
        new_features = features[spec].copy()
        spline_id = np.asarray(new_features[SPLINE_ID])
        spline_id[spline_id >= i] -= 1
        new_features[SPLINE_ID] = spline_id
        self.prof.features = new_features
        self.prof.feature_defaults[SPLINE_ID] = default

    def init_layers(self):
        self.prof = _prof_layer()
        self.work = _work_layer()

    def set_orientation(self, idx: int, orientation: Ori):
        """Set the orientation marker."""
        layer = self.prof
        spline_id = layer.features[SPLINE_ID]
        spec = spline_id == idx
        symbol_arr = layer.symbol.copy()

        symbol_of_interest = symbol_arr[spec]

        match orientation:
            case Ori.none:
                symbol_of_interest[:] = "o"
            case Ori.MinusToPlus:
                symbol_of_interest[0], symbol_of_interest[-1] = "-", "+"
                if len(symbol_of_interest) > 2:
                    symbol_of_interest[1:-1] = "o"
            case Ori.PlusToMinus:
                symbol_of_interest[0], symbol_of_interest[-1] = "+", "-"
                if len(symbol_of_interest) > 2:
                    symbol_of_interest[1:-1] = "o"
            case ori:  # pragma: no cover
                raise RuntimeError(ori)

        # update
        symbol_arr[spec] = symbol_of_interest
        layer.symbol = list(symbol_arr)
        layer.refresh()
        return None


def _prof_layer() -> Points:
    prof = Points(
        ndim=3,
        out_of_slice_display=True,
        size=8,
        name=SELECTION_LAYER_NAME,
        features={SPLINE_ID: []},
        opacity=0.4,
        border_color="black",
        face_color=SplineColor.DEFAULT,
        text={"color": "yellow"},
    )
    prof.feature_defaults[SPLINE_ID] = 0
    prof.editable = False
    return prof


def _work_layer() -> Points:
    work = Points(
        ndim=3,
        out_of_slice_display=True,
        size=8,
        name=WORKING_LAYER_NAME,
        face_color="yellow",
        blending="translucent",
    )
    work.mode = "add"
    return work


def _calc_contrast_limits(arr: np.ndarray) -> tuple[float, float]:
    """Calculate contrast limits for an array."""
    cmin, cmax = fast_percentile(arr, [0.1, 99.9])
    if cmin >= cmax:
        cmax = cmin + 1
    return float(cmin), float(cmax)
