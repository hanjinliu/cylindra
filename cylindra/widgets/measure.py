from typing import TYPE_CHECKING
from enum import Enum

from magicclass import do_not_record, magicclass, abstractapi, vfield, field, MagicTemplate
from magicclass.types import Bound
from magicclass.ext.pyqtgraph import QtImageCanvas, mouse_event

import numpy as np
import impy as ip

if TYPE_CHECKING:
    from .main import CylindraMainWidget

class MeasureMode(Enum):
    none = "none"
    spacing_rise = "spacing/rise"
    skew_npf = "skew/npf"

@magicclass(widget_type="groupbox")
class Parameters(MagicTemplate):
    """
    Cylinder paramters.
    
    Attributes
    ----------
    spacing : str
        Lattice spacing.
    rise : str
        Rise angle (degree).
    skew : str
        Skew angle (degree).
    npf : str
        Number of protofilaments.
    """
    radius = vfield("").with_options(enabled=False)
    spacing = vfield("").with_options(enabled=False)
    rise = vfield("").with_options(enabled=False)
    skew = vfield("").with_options(enabled=False)
    npf = vfield("").with_options(enabled=False)
    
    def __init__(self):
        self._radius = None
        self._spacing = None
        self._rise = None
        self._skew = None
        self._npf = None
    
    @radius.post_get_hook
    def _get_radius(self, value):
        return self._radius
    
    @radius.pre_set_hook
    def _set_radius(self, value):
        self._radius = value
        if value is None:
            return "-- nm"
        else:
            return f"{value:.2f} nm"
    
    @spacing.post_get_hook
    def _get_spacing(self, value):
        return self._spacing
    
    @spacing.pre_set_hook
    def _set_spacing(self, value):
        self._spacing = value
        if value is None:
            return "-- nm"
        else:
            return f"{value:.2f} nm"
    
    @rise.post_get_hook
    def _get_rise(self, value):
        return self._rise
    
    @rise.pre_set_hook
    def _set_rise(self, value):
        self._rise = value
        if value is None:
            return "--°"
        else:
            return f"{value:.2f}°"
    
    @skew.post_get_hook
    def _get_skew(self, value):
        return self._skew
    
    @skew.pre_set_hook
    def _set_skew(self, value):
        self._skew = value
        if value is None:
            return "--°"
        else:
            return f"{value:.2f}°"
    
    @npf.post_get_hook
    def _get_npf(self, value):
        return self._npf
    
    @npf.pre_set_hook
    def _set_npf(self, value):
        self._npf = value
        if value is None:
            return "--"
        else:
            return f"{int(value)}"

@magicclass(layout="horizontal")
class SpectraMeasurer(MagicTemplate):
    canvas = field(QtImageCanvas)
    
    def __init__(self) -> None:
        self._layer_axial = None
        self._layer_angular = None
        self._mode = MeasureMode.none
        self._image = None

    @magicclass(properties={"min_width": "200"})
    class SidePanel(MagicTemplate):
        parameters = abstractapi()
        load_spline = abstractapi()
        select_axial_peak = abstractapi()
        select_angular_peak = abstractapi()
        log_scale = abstractapi()
    
    parameters = SidePanel.field(Parameters)
    log_scale = SidePanel.vfield(False).with_options(tooltip="Check to use log power spectra.")
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        value = MeasureMode(value)
        
        # update button texts
        if value is MeasureMode.none:
            self.SidePanel[2].text = "Select axial peak"
            self.SidePanel[3].text = "Select angular peak"
        elif value is MeasureMode.spacing_rise:
            self.SidePanel[2].text = "Selecting ..."
            self.SidePanel[3].text = "Select angular peak"
        else:
            self.SidePanel[2].text = "Select axial peak"
            self.SidePanel[3].text = "Select ..."
        self._mode = value
    
    def _get_parent(self) -> "CylindraMainWidget":
        from .main import CylindraMainWidget
        
        return self.find_ancestor(CylindraMainWidget, cache=True)
    
    def _get_current_index(self, *_) -> int:
        parent = self._get_parent()
        return parent.SplineControl.num
    
    @SidePanel.wraps
    def load_spline(self, idx: Bound[_get_current_index]):
        self.canvas.mouse_clicked.disconnect(self._on_mouse_clicked, missing_ok=True)
        parent = self._get_parent()
        tomo = parent.tomogram
        self.parameters.radius = tomo.splines[idx].radius
        polar = tomo.straighten_cylindric(idx)
        pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")

        self.canvas.layers.clear()
        self._image = pw.value
        self.canvas.image = self._image
        self._on_log_scale_changed(self.log_scale)
        
        center = np.ceil(np.array(pw.shape) / 2 - 0.5)
        self.canvas.add_infline(center[::-1], 0, color="yellow")
        self.canvas.add_infline(center[::-1], 90, color="yellow")
        
        self.canvas.mouse_clicked.connect(self._on_mouse_clicked, unique=True)
    
    @SidePanel.wraps
    @do_not_record
    def select_axial_peak(self):
        self.mode = MeasureMode.spacing_rise
    
    @SidePanel.wraps
    @do_not_record
    def select_angular_peak(self):
        self.mode = MeasureMode.skew_npf
    
    @log_scale.connect
    def _on_log_scale_changed(self, value):
        if value:
            self.canvas.image = self._image
        else:
            self.canvas.image = np.log(self._image + 1e-12)
    
    def _on_mouse_clicked(self, e: mouse_event.MouseClickEvent):
        if self.mode == MeasureMode.none:
            return
        a0, y0 = e.pos()
        shape = self.canvas.image.shape
        ycenter, acenter = np.ceil(np.array(shape) / 2 - 0.5)
        afreq = (a0 - acenter) / shape[1]
        yfreq = (y0 - ycenter) / shape[0]
        
        parent = self._get_parent()
        scale = parent.tomogram.scale
        
        if self.mode == MeasureMode.spacing_rise:
            self.parameters.spacing = abs(1.0 / yfreq * scale)
            self.parameters.rise = np.rad2deg(np.arctan(-afreq / yfreq))
            
            if self._layer_axial in self.canvas.layers:
                self.canvas.layers.remove(self._layer_axial)
            self._layer_axial = self.canvas.add_scatter([a0], [y0], color="cyan", symbol="+", size=12)
            
        elif self.mode == MeasureMode.skew_npf:
            _p = self.parameters
            self.parameters.skew = np.rad2deg(np.arctan(yfreq / afreq * 2 * _p.spacing / _p.radius))
            self.parameters.npf = int(round(a0 - acenter))

            if self._layer_angular in self.canvas.layers:
                self.canvas.layers.remove(self._layer_angular)
            self._layer_angular = self.canvas.add_scatter([a0], [y0], color="lime", symbol="+", size=12)
        self.mode = MeasureMode.none
