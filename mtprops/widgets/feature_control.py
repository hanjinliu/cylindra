from magicclass import magicclass, MagicTemplate, field, vfield, do_not_record
from magicclass.widgets import FloatRangeSlider, ColorEdit
from ..types import get_monomer_layers
from napari.layers import Points
from napari.utils import Colormap

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
    
    @magicclass(layout="horizontal", labels=False)
    class Colors(MagicTemplate):
        start = vfield("blue", widget_type=ColorEdit, options={"tooltip": "Color that represents lower component of LUT."})
        end = vfield("red", widget_type=ColorEdit, options={"tooltip": "Color that represents higher component of LUT."})
    
    limits = field(FloatRangeSlider, record=False, options={"tooltip": "Contrast limits of features"})
    
    @layer.connect
    def _update_choices(self):
        self["feature_name"].reset_choices()
    
    @feature_name.connect
    def _update_limits(self):
        if self.layer is None:
            return []
        feature = self.layer.features[self.feature_name]
        self.limits.min = feature.min()
        self.limits.max = feature.max()
        self.limits.value = self.limits.range
    
    @magicclass
    class Buttons(MagicTemplate):
        def apply(self): ...
        def reset(self): ...
        def reload(self): ...

    @Buttons.wraps
    @do_not_record
    def apply(self):
        rng = self.limits.range
        self.layer.face_color = self.feature_name
        self.layer.edge_color = self.feature_name
        cmap = Colormap([self.Colors.start, self.Colors.end], "MonomerFeatures")
        self.layer.face_colormap = cmap
        self.layer.edge_colormap = cmap
        self.layer.face_contrast_limits = rng
        self.layer.edge_contrast_limits = rng
        self.layer.refresh()
    
    @Buttons.wraps
    @do_not_record
    def reset(self):
        self.layer.face_color = "lime"
        self.layer.edge_color = "lime"
    
    @Buttons.wraps
    @do_not_record
    def reload(self):
        self.reset_choices()