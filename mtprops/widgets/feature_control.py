from typing import TYPE_CHECKING, List
from magicgui.widgets import Table
from magicclass import (
    magicclass,
    MagicTemplate,
    field,
    vfield,
    do_not_record,
    set_design,
)
from magicclass.widgets import FloatRangeSlider, ColorEdit, ListEdit
from magicclass.types import Color, Bound
import numpy as np
from ..types import get_monomer_layers
from napari.utils.colormaps import Colormap, label_colormap
from napari.layers import Points
from ..const import MOLECULES

if TYPE_CHECKING:
    import pandas as pd
    
@magicclass
class FeatureViewer(MagicTemplate):
    layer = vfield(Points, options={"choices": get_monomer_layers}, record=False)
    table = field(Table, record=False)
    
    @property
    def data(self) -> "pd.DataFrame":
        return self.table.to_dataframe()
    
    def __post_init__(self):
        self.table.read_only = True
        self.table.min_height = 200
    
    @layer.connect
    def _update_table(self):
        if self.visible:
            self._update_table_force()
    
    def _update_table_force(self):
        self.table.value = self.layer.features
        self.Filter["feature_name"].reset_choices()
        self.Filter.expression =""
    
    @magicclass(widget_type="collapsible")
    class Filter(MagicTemplate):
        def _get_feature_names(self, _=None):
            try:
                layer: Points = self.find_ancestor(FeatureViewer).layer
            except RuntimeError:
                layer = None
            if layer is None:
                return []
            cols = layer.features.columns
            return cols
        feature_name = vfield(options={"choices": _get_feature_names}, record=False)
        expression = vfield(str, record=False)
        
        def filter_table(self): ...
        def reset_filter(self): ...
        def add_filtered_molecules(self): ...
    
    @Filter.wraps
    @set_design(text="Filter table")
    @do_not_record
    def filter_table(
        self,
        layer: Bound[layer],
        feature_name: Bound[Filter.feature_name], 
        expr: Bound[Filter.expression]
    ):
        X = "X"
        if X not in self.Filter.expression:
            raise ValueError("Expression does not contain variable 'X'.")
        df = layer.features
        arr = df[feature_name].values
        out = eval(expr, {}, {"X": arr, "np": np})
        dfout = df[out]
        self.table.value = dfout
        return None
    
    @Filter.wraps
    @set_design(text="Resset filter")
    @do_not_record
    def reset_filter(self, layer: Bound[layer]):
        self.table.value = layer.features
        return None
        
    @Filter.wraps
    @set_design(text="Add filtered molecules")
    def add_filtered_molecules(
        self,
        layer: Bound[layer],
        feature_name: Bound[Filter.feature_name], 
        expr: Bound[Filter.expression]
    ):
        X = "X"
        if X not in self.Filter.expression:
            raise ValueError("Expression does not contain variable 'X'.")
        df = layer.features
        arr = df[feature_name].values
        out = eval(expr, {}, {"X": arr, "np": np})
        mole = layer.metadata[MOLECULES]
        mole_filt = mole[out]
        from .main import add_molecules
        expr_str = expr.replace(X, f"'{feature_name}'")
        add_molecules(self.parent_viewer, mole_filt, name=f"{layer.name} ({expr_str})")
        return mole_filt
        
        
@magicclass
class FeatureControl(MagicTemplate):
    def _get_feature_names(self, _=None):
        if self.layer is None:
            return []
        self.layer: Points
        cols = self.layer.features.columns
        return cols
    
    layer = vfield(options={"choices": get_monomer_layers}, record=False)
    feature_name = vfield(options={"choices": _get_feature_names}, record=False)
    
    @magicclass(widget_type="stacked")
    class Colors(MagicTemplate):
        @magicclass
        class ContinuousColorMap(MagicTemplate):
            """Colormap editor for sequencial features."""
            
            @magicclass(layout="horizontal", labels=False)
            class LUT(MagicTemplate):
                start = vfield("blue", widget_type=ColorEdit, options={"tooltip": "Color that represents lower component of LUT."}, record=False)
                end = vfield("red", widget_type=ColorEdit, options={"tooltip": "Color that represents higher component of LUT."}, record=False)
            limits = field(FloatRangeSlider, options={"tooltip": "Contrast limits of features"}, record=False)
            
            def _set_params(self, min: float, max: float):
                self.limits.min = min
                self.limits.max = max
                self.limits.value = self.limits.range
            
            def _apply_colormap(self, layer: Points, feature_name: str):
                cmap = Colormap([self.LUT.start, self.LUT.end], "MonomerFeatures")
                rng = self.limits.value
                layer.face_color = feature_name
                layer.edge_color = feature_name
                layer.face_colormap = cmap
                layer.edge_colormap = cmap
                layer.face_contrast_limits = rng
                layer.edge_contrast_limits = rng
                layer.refresh()
        
        @magicclass(labels=False, widget_type="scrollable")
        class CategoricalColorMap(MagicTemplate):
            """Colormap editor for discrete features."""
            
            colors = vfield(ListEdit, options={"annotation": List[Color], "labels": True, "layout": "vertical"}, record=False)
            def __post_init__(self):
                self["colors"].buttons_visible = False

            def _set_params(self, num: int):
                cmap = np.atleast_2d(self.colors)
                ncolor_now = cmap.size // 4  # NOTE: should NOT be cmap.shape[0]
                if ncolor_now < num:
                    colors = random_color(num - ncolor_now)
                    if ncolor_now == 0:
                        cmap = colors
                    else:
                        cmap = np.concatenate([cmap, colors], axis=0)
                elif ncolor_now > num:
                    cmap = cmap[:num]
                self.colors = cmap

            def _apply_colormap(self, layer: Points, feature_name: str):
                cmap = np.array(self.colors)
                layer.face_color = feature_name
                layer.edge_color = feature_name
                layer.face_color_cycle = cmap
                layer.edge_color_cycle = cmap
                layer.refresh()
            
            def random_color(self):
                ncolors = len(self.colors)
                self.colors = random_color(ncolors)
        
        def _apply_colormap(self, layer: Points, feature_name: str):
            if self.current_index == 0:
                return self.ContinuousColorMap._apply_colormap(layer, feature_name)
            else:
                return self.CategoricalColorMap._apply_colormap(layer, feature_name)
    
    @layer.connect
    def _update_choices(self):
        self["feature_name"].reset_choices()
    
    @layer.connect
    @feature_name.connect
    def _update_limits(self):
        if self.layer is None:
            return []
        feature = self.layer.features[self.feature_name]
        if feature.dtype.kind == "f":
            self.Colors.current_index = 0    
            self.Colors.ContinuousColorMap._set_params(feature.min(), feature.max())
        elif feature.dtype.kind in "iu":
            self.Colors.current_index = 1
            self.Colors.CategoricalColorMap._set_params(len(set(feature)))
        else:
            return []
    
    @magicclass
    class Buttons(MagicTemplate):
        def apply(self): ...
        def reset(self): ...
        def reload(self): ...

    @Buttons.wraps
    @do_not_record
    def apply(self):
        self.Colors._apply_colormap(self.layer, self.feature_name)

    @Buttons.wraps
    @do_not_record
    def reset(self):
        self.layer.face_color = "lime"
        self.layer.edge_color = "lime"
    
    @Buttons.wraps
    @do_not_record
    def reload(self):
        self.reset_choices()


def random_color(ncolors: int, seed: float = None):
    if seed is None:
        seed = np.random.random()
    # NOTE: the first color is translucent
    cmap = label_colormap(num_colors=ncolors + 1, seed=seed)
    return cmap.colors[1:]