from typing import TYPE_CHECKING
import numpy as np
from magicclass import (
    magicclass,
    field,
    vfield,
    MagicTemplate
    )
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas
from ..const import Ori, H

if TYPE_CHECKING:
    import pandas as pd

@magicclass(widget_type="collapsible")
class LocalPropertiesWidget(MagicTemplate):
    """Local properties."""
    
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
    
    @magicclass(widget_type="groupbox", layout="horizontal", labels=False, name="lattice parameters")
    class params(MagicTemplate):
        """Structural parameters at the current position"""
        
        @magicclass(labels=False)
        class pitch(MagicTemplate):
            """Longitudinal pitch length (interval between monomers)"""
            lbl = field("pitch", widget_type="Label")
            txt = vfield("", enabled=False)
            
        @magicclass(labels=False)
        class skew(MagicTemplate):
            """Skew angle """
            lbl = field("skew angle", widget_type="Label")
            txt = vfield("", enabled=False)
            
        @magicclass(labels=False)
        class structure(MagicTemplate):
            lbl = field("structure", widget_type="Label")
            txt = vfield("", enabled=False)

    plot = field(QtMultiPlotCanvas,
                 name="Plot", 
                 options={"nrows": 2, 
                          "ncols": 1, 
                          "sharex": True, 
                          "tooltip": "Plot of local properties"}
                 )
        
    def _init_text(self):
        self.params.pitch.txt = " -- nm"
        self.params.skew.txt = " -- °"
        self.params.structure.txt = " -- "
        
    def _set_text(self, pitch, skew, npf, start):
        self.params.pitch.txt = f" {pitch:.2f} nm"
        self.params.skew.txt = f" {skew:.2f}°"
        self.params.structure.txt = f" {int(npf)}_{start:.1f}"
    
    def _plot_properties(self, props: "pd.DataFrame"):
        if props is None:
            return None
        x = np.asarray(props[H.splDistance])
        pitch_color = "lime"
        skew_color = "gold"
        
        self.plot[0].layers.clear()
        self.plot[0].add_curve(x, props[H.yPitch], color=pitch_color)
        
        self.plot[1].layers.clear()
        self.plot[1].add_curve(x, props[H.skewAngle], color=skew_color)

        self.plot.xlim = (x[0] - 2, x[-1] + 2)
        return None

@magicclass(widget_type="collapsible")
class GlobalPropertiesWidget(MagicTemplate):
    
    def __post_init__(self):
        self._init_text()
        
    @magicclass(widget_type="groupbox", labels=False, name="lattice parameters")
    class params(MagicTemplate):
        
        @magicclass(layout="horizontal", labels=False)
        class params1(MagicTemplate):
            
            @magicclass(labels=False)
            class pitch(MagicTemplate):
                """Longitudinal pitch length (interval between monomers)"""
                lbl = field("pitch", widget_type="Label")
                txt = vfield("", enabled=False)
                
            @magicclass(labels=False)
            class skew(MagicTemplate):
                """Skew angle """
                lbl = field("skew angle", widget_type="Label")
                txt = vfield("", enabled=False)
                
            @magicclass(labels=False)
            class structure(MagicTemplate):
                lbl = field("structure", widget_type="Label")
                txt = vfield("", enabled=False)
        
        @magicclass(layout="horizontal", labels=False)
        class params2(MagicTemplate):
        
            @magicclass(labels=False)
            class radius(MagicTemplate):
                """radius"""
                lbl = field("radius", widget_type="Label")
                txt = vfield("", enabled=False)
                
            @magicclass(labels=False)
            class polarity(MagicTemplate):
                """polarity of MT"""
                lbl = field("polarity", widget_type="Label")
                pol = vfield(Ori.none)

    def _init_text(self):
        self.params.params1.pitch.txt = " -- nm"
        self.params.params1.skew.txt = " -- °"
        self.params.params1.structure.txt = " -- "
        self.params.params2.radius.txt = " -- nm"
        self.params.params2.polarity.pol = Ori.none
        
    def _set_text(self, pitch, skew, npf, start, radius, pol):
        self.params.params1.pitch.txt = f" {pitch:.2f} nm"
        self.params.params1.skew.txt = f" {skew:.2f}°"
        self.params.params1.structure.txt = f" {int(npf)}_{start:.1f}"
        self.params.params2.radius.txt = f" {radius:.2f} nm"
        self.params.params2.polarity.pol = pol
    
    @params.params2.polarity.pol.connect
    def _update_polarity_of_spline(self):
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        i = parent.SplineControl.num.value
        spl = parent.tomogram.splines[i]
        spl.orientation = self.params.params2.polarity.pol