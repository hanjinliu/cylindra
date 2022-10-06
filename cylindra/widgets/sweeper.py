from typing import TYPE_CHECKING
from magicclass import magicclass, field, vfield, MagicTemplate, do_not_record, bind_key, set_design
from magicclass.types import Bound
from magicclass.ext.pyqtgraph import QtImageCanvas
import impy as ip

from ..utils import map_coordinates
from ..const import GVar, nm

if TYPE_CHECKING:
    from .main import CylindraMainWidget
    from ..components import CylSpline

@magicclass
class SplineSweeper(MagicTemplate):
    show_what = vfield(options={"choices": ["R-projection", "Y-projection", "CFT"]})
    depth = vfield(32.0, options={"min": 1.0, "max": 200.0})
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
        num : int
            Spline number in current tomogram.
        pos : int
            Position along the spline.
        """
        num = field(int, label="Spline No.", options={"max": 0}, record=False)
        pos = field(int, label="Position", options={"max": 0}, record=False)
        
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
        
        self.controller.num.max = max(tomo.n_splines - 1, 0)
        self.controller.num.value = 0
        self._num_changed(self.controller.num.value)
        self._update_canvas()
        return None
    
    @controller.num.connect
    def _num_changed(self, i: int):
        try:
            nanchors = len(self.parent.tomogram.splines[i].anchors)
            self.controller.pos.max = max(nanchors - 1, 0)
        except Exception:
            pass
    
    @show_what.connect
    @depth.connect
    @controller.num.connect
    @controller.pos.connect
    def _update_canvas(self):
        _type = self.show_what
        num = self.controller.num.value
        pos = self.controller.pos.value
        if _type == "R-projection":
            polar = self._current_cylindrical_img(num, pos, self.depth).proj("r")
            img = polar.value
        elif _type == "Y-projection":
            block = self._current_cartesian_img(num, pos, self.depth).proj("y")
            img = block.value
        elif _type == "CFT":
            polar = self._current_cylindrical_img(num, pos, self.depth)
            pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
            pw /= pw.max()
            img = pw.value
        else:
            raise RuntimeError
        self.canvas.image = img
        self.canvas.text_overlay.update(visible=True, text=f"{num}-{pos}", color="lime")
    
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
    
    def _show_current_ft(self, i: Bound[controller.num], j: Bound[controller.pos]):
        """View Fourier space of local cylindrical coordinate system at current position."""        
        polar = self._current_cylindrical_img(i, j, self.depth)
        pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
        pw /= pw.max()
        
        self.canvas.image = pw.value
        self.canvas.text_overlay.update(visible=True, text=f"{i}-{j}", color="lime")
        return None
    
    def _show_global_ft(self, i: Bound[controller.num]):
        """View Fourier space along current spline."""  
        polar: ip.ImgArray = self.parent.tomogram.straighten_cylindric(i)
        pw = polar.power_spectra(zero_norm=True, dims="rya").proj("r")
        pw /= pw.max()
    
        self.canvas.image = pw.value
        self.canvas.text_overlay.update(visible=True, text=f"{i}-global", color="magenta")
        return None

    def _current_cartesian_img(self, i: int, j: int, depth: nm) -> ip.ImgArray:
        """Return local Cartesian image at the current position."""
        tomo = self.parent.tomogram
        spl = tomo._splines[i]
        
        length_px = tomo.nm2pixel(depth)
        width_px = tomo.nm2pixel(2 * spl.radius * GVar.outer)
        
        coords = spl.local_cartesian(
            shape=(width_px, width_px), 
            n_pixels=length_px,
            u=spl.anchors[j],
            scale=tomo.scale
        )
        img = tomo.image
        out = map_coordinates(img, coords, order=1)
        out = ip.asarray(out, axes="zyx")
        out.set_scale(img)
        out.scale_unit = img.scale_unit
        return out
    
    def _current_cylindrical_img(self, i: int, j: int, depth: nm):
        """Return cylindric-transformed image at the current position"""
        tomo = self.parent.tomogram
        ylen = tomo.nm2pixel(depth)
        spl = tomo._splines[i]
        
        rmin = tomo.nm2pixel(spl.radius*GVar.inner)
        rmax = tomo.nm2pixel(spl.radius*GVar.outer)
        
        coords = spl.local_cylindrical(
            r_range=(rmin, rmax), 
            n_pixels=ylen, 
            u=spl.anchors[j],
            scale=tomo.scale
        )
        img = tomo.image
        polar = map_coordinates(img, coords, order=1)
        polar = ip.asarray(polar, axes="rya")  # radius, y, angle
        polar.set_scale(r=img.scale.x, y=img.scale.x, a=img.scale.x)
        polar.scale_unit = img.scale_unit
        return polar
