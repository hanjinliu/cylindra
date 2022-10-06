from typing import TYPE_CHECKING
from magicclass import magicclass, field, vfield, MagicTemplate, do_not_record, bind_key
from magicclass.types import Optional
from magicclass.ext.pyqtgraph import QtImageCanvas
import impy as ip

from ..utils import map_coordinates
from ..const import GVar, nm

if TYPE_CHECKING:
    from .main import CylindraMainWidget

@magicclass
class SplineSweeper(MagicTemplate):
    def _get_available_binsize(self, widget=None) -> "list[int]":
        from .main import CylindraMainWidget
        try:
            parent = self.find_ancestor(CylindraMainWidget)
            return parent._get_available_binsize(widget)
        except Exception:
            return []

    show_what = vfield(options={"choices": ["Y-projection", "R-projection", "CFT"]})
    depth = vfield(32.0, options={"min": 1.0, "max": 200.0})
    binsize = vfield(options={"choices": _get_available_binsize})
    radius = vfield(Optional[nm], label="Radius (nm)", options={"text": "Use spline radius", "options": {"max": 100.0}})
    
    canvas = field(QtImageCanvas, options={"lock_contrast_limits": True})

    @property
    def parent(self) -> "CylindraMainWidget":
        from .main import CylindraMainWidget
        return self.find_ancestor(CylindraMainWidget)
    
    @magicclass(layout="horizontal")
    class controller(MagicTemplate):
        """
        Control spline positions.
        
        Attributes
        ----------
        spline_id : int
            Current spline ID to analyze.
        pos : nm
            Position along the spline in nm.
        """
        def _get_spline_id(self, widget=None) -> "list[tuple[str, int]]":
            from .main import CylindraMainWidget
            try:
                parent = self.find_ancestor(CylindraMainWidget)
                return parent._get_splines(widget)
            except Exception:
                return []
            
        spline_id = vfield(label="Spline", options={"choices": _get_spline_id})
        pos = field(nm, label="Position from the tip (nm)", widget_type="FloatSlider", options={"max": 0}, record=False)
        
        @bind_key("Up")
        @do_not_record
        def _next_pos(self):
            self.pos.value = min(self.pos.value + 1, self.pos.max)
        
        @bind_key("Down")
        @do_not_record
        def _prev_pos(self):
            self.pos.value = max(self.pos.value - 1, self.pos.min)
            
    @do_not_record
    def update_widget_state(self):
        tomo = self.parent.tomogram
        if tomo is None:
            return None
        self.parent.tomogram.splines
        parent = self.parent
        tomo = parent.tomogram
        self._spline_changed(self.controller.spline_id)
        self._update_canvas()
        return None
    
    @controller.spline_id.connect
    def _spline_changed(self, idx: int):
        try:
            spl = self.parent.tomogram.splines[idx]
            self.controller.pos.max = max(spl.length(), 0)
        except Exception:
            pass
    
    @show_what.connect
    @depth.connect
    @binsize.connect
    @radius.connect
    @controller.spline_id.connect
    @controller.pos.connect
    def _on_widget_state_changed(self):
        if self.visible:
            self._update_canvas()
        return None

    def _update_canvas(self):
        _type = self.show_what
        idx = self.controller.spline_id
        pos = self.controller.pos.value
        if _type == "R-projection":
            polar = self._current_cylindrical_img(idx, pos, self.depth).proj("r")
            img = polar.value
        elif _type == "Y-projection":
            block = self._current_cartesian_img(idx, pos, self.depth).proj("y")
            img = block.value
        elif _type == "CFT":
            polar = self._current_cylindrical_img(idx, pos, self.depth)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw /= pw.max()
            img = pw.value
        else:
            raise RuntimeError
        self.canvas.image = img
        return None
    
    @show_what.connect
    def _update_clims(self):
        img = self.canvas.image
        if img is not None:
            self.canvas.contrast_limits = (img.min(), img.max())
        return None
    
    def _show_global_r_proj(self):
        """Show radial projection of cylindrical image along current spline."""        
        i = self.controller.num.value
        polar = self.parent.tomogram.straighten_cylindric(i).proj("r")
        self.canvas.image = polar.value
        self.canvas.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        return None
    
    def _show_global_ft(self, i):
        """View Fourier space along current spline."""  
        polar: ip.ImgArray = self.parent.tomogram.straighten_cylindric(i)
        pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
        pw /= pw.max()
    
        self.canvas.image = pw.value
        self.canvas.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        return None

    def _current_cartesian_img(self, idx: int, pos: nm, depth: nm) -> ip.ImgArray:
        """Return local Cartesian image at the current position."""
        tomo = self.parent.tomogram
        spl = tomo.splines[idx]
        length_px = tomo.nm2pixel(depth, binsize=self.binsize)
        if radius := self.radius:
            width_px = tomo.nm2pixel(2 * radius * GVar.outer, binsize=self.binsize)
        else:
            if r := spl.radius:
                width_px = tomo.nm2pixel(2 * r * GVar.outer, binsize=self.binsize)
            else:
                return ip.zeros((1, 1, 1), axes="zyx")
        
        coords = spl.translate(
            [-tomo.multiscale_translation(self.binsize)] * 3
        ).local_cartesian(
            shape=(width_px, width_px), 
            n_pixels=length_px,
            u=pos / spl.length(),
            scale=tomo.scale * self.binsize
        )
        img = tomo._get_multiscale_or_original(self.binsize)
        out = map_coordinates(img, coords, order=1)
        out = ip.asarray(out, axes="zyx")
        out.set_scale(img)
        out.scale_unit = img.scale_unit
        return out
    
    def _current_cylindrical_img(self, idx: int, pos: int, depth: nm):
        """Return cylindric-transformed image at the current position"""
        tomo = self.parent.tomogram
        ylen = tomo.nm2pixel(depth, binsize=self.binsize)
        spl = tomo.splines[idx]
        
        if radius := self.radius:
            rmin = tomo.nm2pixel(radius * GVar.inner, binsize=self.binsize)
            rmax = tomo.nm2pixel(radius * GVar.outer, binsize=self.binsize)
        else:
            if r := spl.radius:
                rmin = tomo.nm2pixel(r * GVar.inner, binsize=self.binsize)
                rmax = tomo.nm2pixel(r * GVar.outer, binsize=self.binsize)
            else:
                return ip.zeros((1, 1, 1), axes="rya")
        
        coords = spl.translate(
            [-tomo.multiscale_translation(self.binsize)] * 3
        ).local_cylindrical(
            r_range=(rmin, rmax), 
            n_pixels=ylen, 
            u=pos / spl.length(),
            scale=tomo.scale * self.binsize,
        )
        img = tomo._get_multiscale_or_original(self.binsize)
        polar = map_coordinates(img, coords, order=1)
        polar = ip.asarray(polar, axes="rya")  # radius, y, angle
        polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
        polar.scale_unit = img.scale_unit
        return polar
