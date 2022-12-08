from typing import List, TYPE_CHECKING
import numpy as np
import impy as ip

from magicclass import (
    magicclass,
    field,
    do_not_record,
    MagicTemplate,
    bind_key,
    set_design,
    abstractapi,
)
from magicclass.types import Bound
from magicclass.ext.pyqtgraph import QtImageCanvas

from cylindra.utils import roundint, centroid, map_coordinates
from cylindra.const import nm, GlobalVariables as GVar, Mode

if TYPE_CHECKING:
    from cylindra.components import CylTomogram


@magicclass
class SplineFitter(MagicTemplate):
    # Manually fit cylinders with spline curve using longitudinal projections
    
    canvas = field(QtImageCanvas, options={"lock_contrast_limits": True})
        
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
        fit = abstractapi()
        
        @bind_key("Up")
        @do_not_record
        def _next_pos(self):
            self.pos.value = min(self.pos.value + 1, self.pos.max)
        
        @bind_key("Down")
        @do_not_record
        def _prev_pos(self):
            self.pos.value = max(self.pos.value - 1, self.pos.min)
            
    def _get_shifts(self, _=None):
        i = self.controller.num.value
        return np.round(self.shifts[i], 3)
    
    @controller.wraps
    @set_design(text="Fit")
    def fit(self, shifts: Bound[_get_shifts], i: Bound[controller.num]):
        """Fit current spline."""
        shifts = np.asarray(shifts)
        parent = self._get_parent()
        _scale = parent.tomogram.scale
        spl = self.splines[i]
        
        min_cr = GVar.minCurvatureRadius
        spl.shift_coa(
            shifts=shifts*self.binsize*_scale,
            min_radius=min_cr, 
            weight_ramp=(min_cr/10, 0.5),
        )
        spl.make_anchors(max_interval=self.max_interval)
        self.fit_done = True
        self._cylinder_changed()
        parent._update_splines_in_images()
        parent._need_save = True
        return None
    
    def __post_init__(self):
        self.shifts: List[np.ndarray] = None
        self.canvas.min_height = 160
        self.fit_done = True
        self.canvas.add_infline(pos=[0, 0], angle=90, color="lime", lw=2)
        self.canvas.add_infline(pos=[0, 0], angle=0, color="lime", lw=2)
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        cos = np.cos(theta)
        sin = np.sin(theta)
        self.canvas.add_curve(cos, sin, color="lime", lw=2, ls="--")
        self.canvas.add_curve(2*cos, 2*sin, color="lime", lw=2, ls="--")
        self.controller.max_height = 50
        self.controller.height = 50
        
        @self.canvas.mouse_clicked.connect
        def _(e):
            if "left" not in e.buttons():
                return
            self.fit_done = False
            x, z = e.pos()
            self._update_cross(x, z)
    
    def _get_parent(self):
        from .main import CylindraMainWidget
        return self.find_ancestor(CylindraMainWidget)
    
    def _update_cross(self, x: float, z: float):
        i = self.controller.num.value
        j = self.controller.pos.value
        if self.shifts[i] is None:
            return
        itemv = self.canvas.layers[0]
        itemh = self.canvas.layers[1]
        item_circ_inner = self.canvas.layers[2]
        item_circ_outer = self.canvas.layers[3]
        itemv.pos = [x, z]
        itemh.pos = [x, z]
        
        tomo = self._get_parent().tomogram
        r_max: nm = GVar.fitWidth/2
        nbin = max(roundint(r_max/tomo.scale/self.binsize/2), 8)
        prof = self.subtomograms[j].radial_profile(center=[z, x], nbin=nbin, r_max=r_max)
        imax = np.argmax(prof)
        imax_sub = centroid(prof, imax-5, imax+5)
        r_peak = (imax_sub+0.5)/nbin*r_max/tomo.scale/self.binsize
        
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        item_circ_inner.xdata = r_peak * GVar.inner * np.cos(theta) + x
        item_circ_inner.ydata = r_peak * GVar.inner * np.sin(theta) + z
        item_circ_outer.xdata = r_peak * GVar.outer * np.cos(theta) + x
        item_circ_outer.ydata = r_peak * GVar.outer * np.sin(theta) + z
        
        lz, lx = self.subtomograms.sizesof("zx")
        self.shifts[i][j, :] = z - lz/2 + 0.5, x - lx/2 + 0.5
        return None
    
    def _load_parent_state(self, max_interval: nm):
        self.max_interval = max_interval
        parent = self._get_parent()
        tomo = parent.tomogram
        for i in range(tomo.n_splines):
            spl = tomo.splines[i]
            spl.make_anchors(max_interval=self.max_interval)
            
        self.shifts = [None] * tomo.n_splines
        self.binsize = parent.layer_image.metadata["current_binsize"]
        self.controller.num.max = tomo.n_splines - 1
        self.controller.num.value = 0
        self._cylinder_changed()
        
    @controller.num.connect
    def _cylinder_changed(self):
        i = self.controller.num.value
        self.controller.pos.value = 0
        parent = self._get_parent()
        imgb = parent.layer_image.data
        tomo: CylTomogram = parent.tomogram
        
        spl = tomo.splines[i]
        self.splines = tomo.splines
        npos = spl.anchors.size
        self.shifts[i] = np.zeros((npos, 2))
        
        length_px = tomo.nm2pixel(GVar.fitLength, binsize=self.binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=self.binsize)
        
        mole = spl.anchors_to_molecules()
        coords = mole.cartesian_at(
            index=slice(None),
            shape=(width_px, length_px, width_px), 
            scale=tomo.scale*self.binsize
        )
        out: list[ip.ImgArray] = []
        for crds in coords:
            out.append(map_coordinates(imgb, crds, order=1, mode=Mode.constant, cval=np.mean))
        subtomo: ip.ImgArray = ip.asarray(np.stack(out, axis=0), axes="pzyx")
        self.subtomograms = subtomo.proj("y")[ip.slicer.x[::-1]]
        
        self.canvas.image = self.subtomograms[0]
        self.controller.pos.max = npos - 1
        self.canvas.xlim = (0, self.canvas.image.shape[1])
        self.canvas.ylim = (0, self.canvas.image.shape[0])
        lz, lx = self.subtomograms.sizesof("zx")
        self._update_cross(lx/2 - 0.5, lz/2 - 0.5)

        return None
    
    @controller.pos.connect
    def _position_changed(self):
        i = self.controller.num.value
        j = self.controller.pos.value
        self.canvas.image = self.subtomograms[j]
        if self.shifts is not None and self.shifts[i] is not None:
            y, x = self.shifts[i][j]
        else:
            y = x = 0
        lz, lx = self.subtomograms.shape[-2:]
        self._update_cross(x + lx/2 - 0.5, y + lz/2 - 0.5)
