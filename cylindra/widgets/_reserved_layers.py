from __future__ import annotations

from typing import TYPE_CHECKING

import impy as ip
import numpy as np
import polars as pl
from napari.layers import Image, Points

from cylindra._napari import CylinderLabels
from cylindra.const import (
    SELECTION_LAYER_NAME,
    SPLINE_ID,
    WORKING_LAYER_NAME,
    Ori,
    SplineColor,
)

if TYPE_CHECKING:
    from cylindra.components import CylSpline


class ReservedLayers:
    def __init__(self):
        self.image: Image | None = None
        self.prof: Points | None = None
        self.work: Points | None = None
        self.paint: CylinderLabels | None = None
        self.highlight = Points(
            ndim=3,
            name="Highlight",
            face_color="transparent",
            edge_color="crimson",
            edge_width=0.16,
            edge_width_is_relative=True,
            out_of_slice_display=True,
            blending="translucent_no_depth",
        )
        self.highlight.editable = False

    def update_image(self, img: ip.ImgArray, bin_size: int, tr: float):
        """Update the reserved image layer"""
        self.image.data = img
        self.image.scale = img.scale
        self.image.name = f"{img.name} (bin {bin_size})"
        self.image.translate = [tr] * 3
        self.image.contrast_limits = [np.min(img), np.max(img)]

        if self.paint is not None:
            self.paint.scale = self.image.scale
            self.paint.translate = self.image.translate

    def reset_image(self, img: ip.ImgArray, bin_size: int, tr: float):
        """Reset the reserved image layer"""
        self.image = Image(
            img,
            scale=img.scale,
            name=f"{img.name} (bin {bin_size})",
            translate=[tr, tr, tr],
            contrast_limits=[np.min(img), np.max(img)],
            blending="translucent_no_depth",
        )

    def init_paint(self):
        """Initialize the cylinder paint layer."""
        if self.paint is not None:
            self.paint.data = np.zeros_like(self.paint.data)
            if self.image is not None:
                self.paint.scale = self.image.scale
                self.paint.translate = self.image.translate

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

    @property
    def image_data(self) -> ip.ImgArray:
        return self.image.data

    @property
    def scale(self) -> float:
        return self.image.scale[-1]

    def contains(self, layer):
        return layer in (self.image, self.prof, self.work, self.paint)

    def add_paint(self, lbl, props: pl.DataFrame):
        """Add cylinder paint layer"""
        if self.paint is None:
            self.paint = CylinderLabels(
                lbl,
                scale=self.image.scale,
                translate=self.image.translate,
                opacity=0.33,
                name="Cylinder properties",
                features=props.to_pandas(),
            )
        else:
            self.paint.data = lbl
            self.paint.features = props.to_pandas()

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

    def init_prof_and_work(self):
        common_properties = {"ndim": 3, "out_of_slice_display": True, "size": 8}
        self.prof = Points(
            **common_properties,
            name=SELECTION_LAYER_NAME,
            features={SPLINE_ID: []},
            opacity=0.4,
            edge_color="black",
            face_color=SplineColor.DEFAULT,
            text={"color": "yellow"},
        )
        self.prof.feature_defaults[SPLINE_ID] = 0
        self.prof.editable = False

        self.work = Points(
            **common_properties,
            name=WORKING_LAYER_NAME,
            face_color="yellow",
            blending="translucent_no_depth",
        )

        self.work.mode = "add"

    def set_orientation(self, idx: int, orientation: Ori):
        """Set the orientation marker text."""
        layer = self.prof
        spline_id = layer.features[SPLINE_ID]
        spec = spline_id == idx
        if layer.text.string.encoding_type == "ConstantStringEncoding":
            # if text uses constant string encoding, update it to ManualStringEncoding
            string_arr = np.zeros(len(layer.data), dtype="<U1")
        else:
            string_arr = np.asarray(layer.text.string.array, dtype="<U1")

        str_of_interest = string_arr[spec]

        if orientation is Ori.none:
            str_of_interest[:] = ""
        elif orientation is Ori.MinusToPlus:
            str_of_interest[0], str_of_interest[-1] = "-", "+"
        elif orientation is Ori.PlusToMinus:
            str_of_interest[0], str_of_interest[-1] = "+", "-"
        else:  # pragma: no cover
            raise RuntimeError(orientation)

        # update
        string_arr[spec] = str_of_interest
        layer.text.string = list(string_arr)

        layer.text.string = list(string_arr)
        layer.refresh()
        return None
