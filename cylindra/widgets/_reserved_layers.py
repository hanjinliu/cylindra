from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import impy as ip
import polars as pl
from napari.layers import Image, Points
from cylindra._custom_layers import CylinderLabels

from cylindra.const import (
    SplineColor,
    SPLINE_ID,
    WORKING_LAYER_NAME,
    SELECTION_LAYER_NAME,
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
        self.highlight.interactive = False

    def update_image(self, imgb: ip.ImgArray, bin_size: int, tr: float):
        self.image.data = imgb
        self.image.scale = imgb.scale
        self.image.name = f"{imgb.name} (bin {bin_size})"
        self.image.translate = [tr] * 3
        self.image.contrast_limits = [np.min(imgb), np.max(imgb)]

        if self.paint is not None:
            self.paint.scale = self.image.scale
            self.paint.translate = self.image.translate

    def reset_image(self, imgb: ip.ImgArray, bin_size: int, tr: float):
        self.image = Image(
            imgb,
            scale=imgb.scale,
            name=f"{imgb.name} (bin {bin_size})",
            translate=[tr, tr, tr],
            contrast_limits=[np.min(imgb), np.max(imgb)],
        )

    def init_paint(self):
        if self.paint is not None:
            self.paint.data = np.zeros_like(self.paint.data)
            if self.image is not None:
                self.paint.scale = self.image.scale
                self.paint.translate = self.image.translate

    def highlight_spline(self, i: int):
        spec = self.prof.features[SPLINE_ID] == i
        self.prof.face_color = SplineColor.DEFAULT
        self.prof.face_color[spec] = SplineColor.SELECTED
        self.prof.refresh()

    def add_spline(self, i: int, spl: CylSpline):
        interval = 15
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
        common_properties = dict(ndim=3, out_of_slice_display=True, size=8)
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
