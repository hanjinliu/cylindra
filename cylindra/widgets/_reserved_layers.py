from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakSet

import impy as ip
import numpy as np
from napari.layers import Image, Layer, Points, Shapes

from cylindra.const import (
    IS_SEGMENT,
    SELECTION_LAYER_NAME,
    SPLINE_ID,
    WORKING_LAYER_NAME,
    Ori,
    SplineColor,
)
from cylindra.widgets._main_utils import fast_percentile

if TYPE_CHECKING:
    from cylindra._napari._layers import SplineLayer
    from cylindra.components import CylSpline, SplineSegment


class ReservedLayers:
    """A class that stores layers reserved for a cylindra session."""

    def __init__(self):
        self.image = Image(ip.zeros((1, 1, 1), axes="zyx"))
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
        self.ref_inverted = False

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
        layer = self.prof
        is_this_spl = layer.features[SPLINE_ID] == i
        is_segment = layer.features[IS_SEGMENT]
        # update face color
        layer.face_color = SplineColor.DEFAULT
        layer.face_color[is_this_spl & (~is_segment)] = SplineColor.SELECTED
        layer.face_color[is_segment] = SplineColor.SEGMENT
        # update size
        size = layer.size.astype(np.float32)
        size[is_segment] = self.prof._size_spline * 1.8
        layer.size = size
        layer.refresh()

    def add_spline(self, i: int, spl: CylSpline):
        """Add spline sample data to the layer."""
        interval = self.prof._size_spline
        length = spl.length()
        num = max(int(length / interval) + 1, 2)
        offset = interval * 1.8 / length
        spl_pos = np.linspace(0, 1, num)
        pos = np.concatenate([[-offset], spl_pos, [1 + offset]])
        fit = spl.map(pos)
        self.prof.feature_defaults[SPLINE_ID] = i
        self.prof.feature_defaults[IS_SEGMENT] = False
        self.prof.current_size = self.prof._size_spline
        self.prof.add(fit)
        sizes = self.prof.size.astype(np.float32)
        sizes[-num - 2] = sizes[-1] = 0.01
        self.prof.size = sizes
        for seg in spl.segments:
            self.add_spline_segment(i, spl, seg)
        return spl.map(spl_pos)

    def add_spline_segment(self, i: int, spl: CylSpline, segment: SplineSegment):
        interval = self.prof._size_spline / 2
        length = segment.length(spl)
        num = int(length / interval)
        fit = segment.sample(spl, num)
        self.prof.feature_defaults[SPLINE_ID] = i
        self.prof.feature_defaults[IS_SEGMENT] = True
        self.prof.current_symbol = "ring"
        self.prof.current_face_color = SplineColor.SEGMENT
        self.prof.current_size = self.prof._size_spline
        self.prof.add(fit)

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
        self.prof.feature_defaults[IS_SEGMENT] = False

    def init_layers(self):
        self.prof = _prof_layer()
        self.work = _work_layer()

    def set_orientation(self, idx: int, orientation: Ori):
        """Set the orientation marker."""
        layer = self.prof
        spec = (layer.features[SPLINE_ID] == idx) & (~layer.features[IS_SEGMENT])
        symbol_arr = layer.symbol.copy()
        size_arr = layer.size.astype(np.float32)

        symbol_of_interest = symbol_arr[spec]
        size_of_interest = size_arr[spec]

        _size_pol = layer._size_polarity_marker
        match orientation:
            case Ori.none:
                symbol_a, symbol_b = "x", "x"
                size_edge = 0.01
            case Ori.MinusToPlus:
                symbol_a, symbol_b = "-", "+"
                size_edge = _size_pol if layer.show_polarity else 0.01
            case Ori.PlusToMinus:
                symbol_a, symbol_b = "+", "-"
                size_edge = _size_pol if layer.show_polarity else 0.01
            case ori:  # pragma: no cover
                raise RuntimeError(ori)

        symbol_of_interest[0], symbol_of_interest[-1] = symbol_a, symbol_b
        size_of_interest[0] = size_of_interest[-1] = size_edge
        if len(symbol_of_interest) > 2:
            symbol_of_interest[1:-1] = "o"

        # update
        symbol_arr[spec] = symbol_of_interest
        size_arr[spec] = size_of_interest
        layer.symbol = list(symbol_arr)
        layer.size = size_arr
        layer.selected_data = []
        layer.refresh()


def _prof_layer() -> SplineLayer:
    from cylindra._napari._layers import SplineLayer

    prof = SplineLayer(
        ndim=3,
        out_of_slice_display=True,
        size=8.0,
        name=SELECTION_LAYER_NAME,
        features={
            SPLINE_ID: np.zeros(0, dtype=np.uint32),
            IS_SEGMENT: np.zeros(0, dtype=bool),
        },
        opacity=0.4,
        border_color="black",
        face_color=SplineColor.DEFAULT,
        text={"color": "yellow"},
    )
    prof.feature_defaults[SPLINE_ID] = 0
    prof.feature_defaults[IS_SEGMENT] = False
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
    work.bind_key("Ctrl-C")(_work_layer_copy)
    work.bind_key("Ctrl-X")(_work_layer_cut)
    work.bind_key("Ctrl-V")(_work_layer_paste)
    return work


_CLIPBOARD_KEY = "cylindra.clipboard"


def _work_layer_copy(layer: Points):
    """Copy the points to the internal clipboard."""
    if sel := sorted(layer.selected_data):
        layer.metadata[_CLIPBOARD_KEY] = layer.data[sel].copy()


def _work_layer_cut(layer: Points):
    """Cut the points to the internal clipboard."""
    if sel := sorted(layer.selected_data):
        layer.metadata[_CLIPBOARD_KEY] = layer.data[sel].copy()
        layer.data = np.delete(layer.data, sel, axis=0)


def _work_layer_paste(layer: Points):
    """Paste the points from the internal clipboard."""
    if _CLIPBOARD_KEY in layer.metadata:
        clip_data = layer.metadata[_CLIPBOARD_KEY]
        if not isinstance(clip_data, np.ndarray):
            return
        data_old = layer.data
        data_new = layer.data = np.vstack([data_old, clip_data])
        layer.selected_data = list(range(data_old.shape[0], data_new.shape[0]))


def _calc_contrast_limits(arr: np.ndarray) -> tuple[float, float]:
    """Calculate contrast limits for an array."""
    cmin, cmax = fast_percentile(arr, [0.1, 99.9])
    if cmin >= cmax:
        cmax = cmin + 1
    return float(cmin), float(cmax)
