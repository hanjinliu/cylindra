from typing import List
import numpy as np
from magicclass import magicclass, MagicTemplate, field
from magicclass.ext.pyqtgraph import QtMultiImageCanvas
from ..const import GVar, H
from ..utils import no_verbose, load_rot_subtomograms, Projections


@magicclass(widget_type="groupbox", name="Spline Control")
class SplineControl(MagicTemplate):
    """MT sub-regions"""
    def __post_init__(self):
        self.projections: List[Projections] = []
        self.pos.min_width = 70
        self.canvas.min_height = 200
        self.canvas.max_height = 230
        self.canvas[0].lock_contrast_limits = True
        self.canvas[0].title = "XY-Projection"
        self.canvas[1].lock_contrast_limits = True
        self.canvas[1].title = "XZ-Projection"
        self.canvas[2].lock_contrast_limits = True
        self.canvas[2].title = "Rot. average"
        
    def _get_splines(self, widget=None) -> List[int]:
        """Get list of spline objects for categorical widgets."""
        from .main import MTPropsWidget
        try:
            tomo = self.find_ancestor(MTPropsWidget).tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]
    
    num = field(int, widget_type="ComboBox", options={"choices": _get_splines}, name="Spline No.", record=False)
    pos = field(int, widget_type="Slider", options={"max": 0, "tooltip": "Position along a MT."}, name="Position", record=False)
    canvas = field(QtMultiImageCanvas, name="Figure", options={"nrows": 1, "ncols": 3, "tooltip": "Projections"})
    
    @num.connect
    def _num_changed(self):
        from .main import MTPropsWidget
        i = self.num.value
        parent = self.find_ancestor(MTPropsWidget)
        tomo = parent.tomogram
        spl = tomo.splines[i]
        
        if spl.localprops is not None:
            n_anc = len(spl.localprops)
        elif spl._anchors is not None:
            n_anc = len(spl._anchors)
        else:
            self.pos.value = 0
            self.pos.max = 0
            return
        
        self.pos.max = n_anc - 1
        
        # update plots in pyqtgraph
        parent.Local_Properties._plot_properties(spl.localprops)
        
        # calculate projection
        if spl.localprops is not None:
            npf_list = spl.localprops[H.nPF]
        elif spl.globalprops is not None:
            npf_list = [spl.globalprops[H.nPF]] * spl.anchors.size
        else:
            return None

        binsize = tomo.metadata["binsize"]
        imgb = parent.layer_image.data
        
        # TODO: remove load_rot_subtomograms
        spl.scale *= binsize
        
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        out = load_rot_subtomograms(imgb, length_px, width_px, spl)
        
        spl.scale /= binsize
        
        projections = []
        for img, npf in zip(out, npf_list):    
            proj = Projections(img)
            proj.rotational_average(npf)
            projections.append(proj)
        
        self.projections = projections
        self._pos_changed()
        return None

    @pos.connect
    def _pos_changed(self):
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        tomo = parent.tomogram
        i = self.num.value
        j = self.pos.value
        npaths = len(tomo.splines)
        if 0 == npaths:
            return
        if 0 < npaths <= i:
            i = 0
        spl = tomo.splines[i]
        
        binsize = parent.tomogram.metadata["binsize"]
        with no_verbose():
            proj = self.projections[j]
            for ic in range(3):
                self.canvas[ic].layers.clear()
            self.canvas[0].image = proj.yx
            self.canvas[1].image = proj.zx
            self.canvas[2].image = proj.zx_ave
        
        # Update text overlay
        self.canvas[0].text_overlay.text = f"{i}-{j}"
        self.canvas[0].text_overlay.color = "lime"
        
        if spl.radius is None:
            return None
        lz, ly, lx = np.array(proj.shape)
        
        if parent._last_ft_size is None:
            ylen = 25/binsize/tomo.scale
        else:
            ylen = parent._last_ft_size/2/binsize/tomo.scale
        
        # draw a square in YX-view
        ymin, ymax = ly/2 - ylen - 0.5, ly/2 + ylen + 0.5
        r_px = spl.radius/tomo.scale/binsize
        r = r_px * GVar.outer
        xmin, xmax = -r + lx/2 - 0.5, r + lx/2 + 0.5
        self.canvas[0].add_curve([xmin, xmin, xmax, xmax, xmin], 
                                 [ymin, ymax, ymax, ymin, ymin], color="lime")

        # draw two circles in ZX-view
        theta = np.linspace(0, 2*np.pi, 360)
        r = r_px * GVar.inner
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = r_px * GVar.outer
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        
        # update pyqtgraph
        if parent.Local_Properties._y_pitch is not None:
            x = spl.localprops[H.splDistance][j]
            parent.Local_Properties._plot_spline_position(x)
        return None
    
    def _reset_contrast_limits(self):
        for i in range(3):
            img = self.canvas[i].image
            if img is not None:
                self.canvas[i].contrast_limits = [img.min(), img.max()]
        return None