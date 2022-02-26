from magicclass import (
    magicclass,
    field,
    vfield,
    MagicTemplate
    )
from magicclass.ext.pyqtgraph import QtMultiPlotCanvas


@magicclass(widget_type="collapsible")
class LocalProperties(MagicTemplate):
    """Local profiles."""
    
    @magicclass(widget_type="groupbox", layout="horizontal", labels=False, name="structural parameters")
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
        
        def _init_text(self):
            self.pitch.txt = " -- nm"
            self.skew.txt = " -- °"
            self.structure.txt = " -- "
            
        def _set_text(self, pitch, skew, npf, start):
            self.pitch.txt = f" {pitch:.2f} nm"
            self.skew.txt = f" {skew:.2f}°"
            self.structure.txt = f" {int(npf)}_{start:.1f}"

    plot = field(QtMultiPlotCanvas,
                 name="Plot", 
                 options={"nrows": 2, 
                          "ncols": 1, 
                          "sharex": True, 
                          "tooltip": "Plot of local properties"}
                 )
