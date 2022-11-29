from typing import TYPE_CHECKING, List
from magicgui.widgets import Table, Label, ListEdit
from magicclass import (
    magicclass,
    MagicTemplate,
    field,
    vfield,
    do_not_record,
    set_design,
)
from magicclass.widgets import FloatRangeSlider, ColorEdit
from magicclass.types import Color, Bound, OneOf
import numpy as np
from napari.utils.colormaps import Colormap, label_colormap
from napari.layers import Points
from cylindra.types import get_monomer_layers
from cylindra.const import MOLECULES

if TYPE_CHECKING:
    import pandas as pd

@magicclass
class FeatureControl(MagicTemplate):
    """
    Feature control widget.

    Attributes
    ----------
    layer : Points
        Select a monomers layer.
    feature_name : str
        Select target feature of the layer.
    """
    def _get_feature_names(self, _=None):
        try:
            layer: Points = self.layer
        except RuntimeError:
            layer = None
        if layer is None:
            return []
        cols = layer.features.columns
        return cols
    
    layer = vfield(OneOf[get_monomer_layers], record=False)
    feature_name = field(OneOf[_get_feature_names], record=False)
    table = field(Table, record=False)
    
    def _get_feature_dataframe(self) -> "pd.DataFrame":
        df = self.layer.features
        return df
            
    def __post_init__(self):
        self.table.read_only = True
        self.table.min_height = 200
    
    @property
    def data(self) -> "pd.DataFrame":
        return self.table.to_dataframe()

    @magicclass(widget_type="tabbed")
    class Tabs(MagicTemplate):        
        @magicclass(name="Colormap Editor")
        class ColorEditor(MagicTemplate):
            @magicclass(widget_type="stacked")
            class Colors(MagicTemplate):
                @magicclass(labels=False, widget_type="scrollable")
                class ContinuousColorMap(MagicTemplate):
                    """
                    Colormap editor for sequencial features.
                    
                    Attributes
                    ----------
                    limits : tuple of float
                        Contrast limits of features
                    """
                    @magicclass(layout="horizontal", labels=False)
                    class LUT(MagicTemplate):
                        """
                        Look up table of a continuous color map.
                        
                        Attributes
                        ----------
                        start : Color
                            Color that represents lower component of LUT.
                        end : Color
                            Color that represents higher component of LUT.
                        """
                        start = vfield("blue", widget_type=ColorEdit, record=False)
                        end = vfield("red", widget_type=ColorEdit, record=False)
                        
                    limits = field(FloatRangeSlider, record=False)
                    
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
                    
                    colors = vfield(ListEdit, options={"annotation": List[Color], "layout": "vertical"}, record=False)
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
                        """Generate random colors."""
                        ncolors = len(self.colors)
                        self.colors = random_color(ncolors)
                
            def _apply_colormap(self, layer: Points, feature_name: str):
                if self.Colors.current_index == 0:
                    return self.Colors.ContinuousColorMap._apply_colormap(layer, feature_name)
                else:
                    return self.Colors.CategoricalColorMap._apply_colormap(layer, feature_name)
            
            def apply(self): ...
            def reset(self): ...
            def reload(self): ...

        
        @magicclass(labels=False)
        class Filter(MagicTemplate):
            _TEMPLATE = (
                "Use 'X' to substitute '{}' feature.\n"
                "Numpy is available in this scope as 'np'.\n"
                "e.g. 'X > 10', 'X > np.mean(X)'"
            )
            doc = field(Label, record=False)
            expression = vfield(str, record=False)
            
            def filter_table(self): ...
            def reset_filter(self): ...
            def create_molecules(self): ...
            
    @Tabs.Filter.wraps
    @set_design(text="Filter table")
    @do_not_record
    def filter_table(self, expr: Bound[Tabs.Filter.expression]):
        """Filter table using current expression. Molecules layer is not affected."""
        X = "X"
        if X not in self.Tabs.Filter.expression:
            raise ValueError("Expression does not contain variable 'X'.")
        df = self._get_feature_dataframe()
        arr = df.values
        out = eval(expr, {}, {"X": arr, "np": np})
        dfout = df[out]
        self.table.value = dfout
        return None
    
    @Tabs.Filter.wraps
    @set_design(text="Resset filter")
    @do_not_record
    def reset_filter(self):
        """Reset filter and restore original table."""
        self.table.value = self.layer.features
        return None
        
    @Tabs.Filter.wraps
    @set_design(text="Create molecules")
    def create_molecules(
        self,
        layer: Bound[layer],
        feature_name: Bound[feature_name], 
        expr: Bound[Tabs.Filter.expression],
    ):
        """Create molecules object with current table state."""
        X = "X"
        if X not in expr:
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

    @Tabs.ColorEditor.wraps
    @do_not_record
    def apply(self):
        """Apply current colormap to the selected points layer."""
        self.Tabs.ColorEditor._apply_colormap(self.layer, self.feature_name.value)
        return None

    @Tabs.ColorEditor.wraps
    @do_not_record
    def reset(self):
        """Reset layer colors."""
        self.layer.face_color = "lime"
        self.layer.edge_color = "lime"
        return None
    
    @Tabs.ColorEditor.wraps
    @do_not_record
    def reload(self):
        """Reload this widget."""
        self.reset_choices()
        return None
    
    @layer.connect
    def _on_layer_change(self):
        if self.visible:
            self._update_table_and_expr()
        self.feature_name.reset_choices()
    
    def _update_table_and_expr(self):
        if self.layer is None:
            return
        self.table.value = self.layer.features
        self.Tabs.Filter.expression =""
    
    @layer.connect
    @feature_name.connect
    def _on_selection_change(self):
        if self.layer is None:
            return []
        feature = self.layer.features[self.feature_name.value]
        self.Tabs.Filter.doc.value = self.Tabs.Filter._TEMPLATE.format(self.feature_name.value)
        _color_widget = self.Tabs.ColorEditor.Colors
        if feature.dtype.kind == "f":
            _color_widget.current_index = 0    
            _color_widget.ContinuousColorMap._set_params(feature.min(), feature.max())
        elif feature.dtype.kind in "iu":
            _color_widget.current_index = 1
            _color_widget.CategoricalColorMap._set_params(len(set(feature)))
        else:
            return []

def random_color(ncolors: int, seed: float = None):
    if seed is None:
        seed = np.random.random()
    # NOTE: the first color is translucent
    cmap = label_colormap(num_colors=ncolors + 1, seed=seed)
    return cmap.colors[1:]
