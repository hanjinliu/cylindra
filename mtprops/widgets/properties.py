from typing import TYPE_CHECKING
import numpy as np
from magicclass import (
    magicclass,
    field,
    vfield,
    MagicTemplate,
    FieldGroup,
    )
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas
from ..const import H

if TYPE_CHECKING:
    import pandas as pd

class LabeledText(FieldGroup):
    lbl = field("pitch", widget_type="Label")
    txt = vfield("", options={"enabled": False})
    
    def __init__(self, label_text: str):
        super().__init__(labels=False)
        self.lbl.value = label_text
        self.margins = (0, 0, 0, 0)
    

@magicclass(widget_type="collapsible", name="Local Properties")
class LocalPropertiesWidget(MagicTemplate):
    """Local properties."""
    
    @magicclass(widget_type="groupbox", layout="horizontal", labels=False, name="lattice parameters")
    class params(MagicTemplate):
        """Structural parameters at the current position"""
        
        pitch = LabeledText("pitch")
        skew = LabeledText("skew angle")
        structure = LabeledText("structure")
        
    plot = field(QtMultiPlotCanvas,
                 name="Plot", 
                 options={"nrows": 2, 
                          "ncols": 1, 
                          "sharex": True, 
                          "tooltip": "Plot of local properties"}
                 )
    
    def __post_init__(self):
        # Initialize multi-plot canvas
        self.plot.min_height = 240
        self.plot[0].ylabel = "pitch (nm)"
        self.plot[0].legend.visible = False
        self.plot[0].border = [1, 1, 1, 0.2]
        self.plot[1].xlabel = "position (nm)"
        self.plot[1].ylabel = "skew (deg)"
        self.plot[1].legend.visible = False
        self.plot[1].border = [1, 1, 1, 0.2]
        
        self._init_text()
        
        self._y_pitch = None
        self._skew_angle = None
        
    def _init_text(self):
        self.params.pitch.txt = " -- nm"
        self.params.skew.txt = " -- °"
        self.params.structure.txt = " -- "
        return None
        
    def _set_text(self, pitch, skew, npf, start):
        self.params.pitch.txt = f" {pitch:.2f} nm"
        self.params.skew.txt = f" {skew:.2f}°"
        self.params.structure.txt = f" {int(npf)}_{start:.1f}"
        return None
    
    def _init_plot(self):
        self.plot[0].layers.clear()
        self.plot[1].layers.clear()
        self._y_pitch = None
        self._skew_angle = None
        return None
    
    def _plot_properties(self, props: "pd.DataFrame"):
        if props is None:
            return None
        x = np.asarray(props[H.splDistance])
        if x[0] > x[-1]:
            x = x[::-1]
        pitch_color = "lime"
        skew_color = "gold"
        
        self._init_plot()
        
        self._y_pitch = np.asarray(props[H.yPitch])
        self._skew_angle = np.asarray(props[H.skewAngle])
        
        self.plot[0].add_curve(x, self._y_pitch, color=pitch_color)
        self.plot[1].add_curve(x, self._skew_angle, color=skew_color)

        self.plot[0].xlim = (x[0] - 2, x[-1] + 2)
        self.plot[0].add_infline(pos=[x[0], 0], angle=90, color=[1., 0., 0., 0.3], lw=2)
        self.plot[1].add_infline(pos=[x[0], 0], angle=90, color=[1., 0., 0., 0.3], lw=2)
        self._plot_spline_position(x[0])
        return None
    
    def _plot_spline_position(self, x: float):
        self.plot[0].layers[-1].pos = [x, 0]
        self.plot[1].layers[-1].pos = [x, 0]
        xmin, xmax = self.plot[0].xlim
        if x < xmin or xmax < x:
            dx = xmax - xmin
            self.plot[0].xlim = (x - dx/2, x + dx/2)
        return None
            

@magicclass(widget_type="collapsible", name="Global Properties")
class GlobalPropertiesWidget(MagicTemplate):
    """Global properties."""
    
    def __post_init__(self):
        self._init_text()
        
    @magicclass(widget_type="groupbox", labels=False, name="lattice parameters")
    class params(MagicTemplate):
        
        @magicclass(layout="horizontal", labels=False)
        class params1(MagicTemplate):
            
            pitch = LabeledText("pitch")
            skew = LabeledText("skew angle")
            structure = LabeledText("structure")
            
        @magicclass(layout="horizontal", labels=False)
        class params2(MagicTemplate):
        
            radius = LabeledText("radius")
            polarity = LabeledText("polarity")
            
    def _init_text(self):
        self.params.params1.pitch.txt = " -- nm"
        self.params.params1.skew.txt = " -- °"
        self.params.params1.structure.txt = " -- "
        self.params.params2.radius.txt = " -- nm"
        self.params.params2.polarity.txt = " -- "
        return None
        
    def _set_text(self, pitch, skew, npf, start, radius, orientation):
        self.params.params1.pitch.txt = f" {pitch:.2f} nm"
        self.params.params1.skew.txt = f" {skew:.2f}°"
        self.params.params1.structure.txt = f" {int(npf)}_{start:.1f}"
        self.params.params2.radius.txt = f" {radius:.2f} nm" if radius is not None else " -- nm"
        self.params.params2.polarity.txt = orientation
        return None