from typing import List
import numpy as np
import impy as ip
from magicclass import magicclass, MagicTemplate, field, vfield, set_options
from magicclass.types import Bound
from magicclass.ext.pyqtgraph import QtMultiImageCanvas

from ..const import GVar, Ori, H, Mode
from ..utils import map_coordinates, Projections
from ..components.tomogram import MtSpline

@magicclass(widget_type="groupbox")
class SplineControl(MagicTemplate):
    """MT sub-regions"""
    def __post_init__(self):
        self.projections: List[Projections] = []
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
    
    num = vfield(int, widget_type="ComboBox", options={"choices": _get_splines, "tooltip": "Spline in current tomogram.", "label": "Spline No."}, record=False)
    pos = vfield(int, widget_type="Slider", options={"max": 0, "tooltip": "Position along a MT.", "label": "Position"}, record=False)
    canvas = field(QtMultiImageCanvas, name="Figure", options={"nrows": 1, "ncols": 3, "tooltip": "Projections"})
    
    @magicclass(layout="horizontal")
    class footer(MagicTemplate):
        focus = vfield(False, options={"text": "focus on", "tooltip": "Keep focus of viewer camera on the current spline position"}, record=False)
        def set_PF_number(self): ...
        def set_orientation(self): ...
    
    @footer.wraps
    @set_options(labels=False)
    def set_PF_number(self, i: Bound[num], npf: int = 13):
        """Manually update protofilament number."""
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        if parent.tomogram is None or i is None:
            return None
        spl: MtSpline = parent.tomogram.splines[i]
        if spl.localprops is not None:
            spl.localprops[H.nPF].values[:] = npf
            parent._update_local_properties_in_widget()
        if spl.globalprops is not None:
            spl.globalprops[H.nPF] = npf
            parent._update_global_properties_in_widget()
        if self.canvas[0].image is not None:
            parent.Sample_subtomograms()
        return None
        
    @footer.wraps
    @set_options(labels=False, orientation={"widget_type": "RadioButtons"})
    def set_orientation(self, i: Bound[num], orientation: Ori = Ori.none):
        """Manually set polarity."""
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        if parent.tomogram is None or i is None:
            return None
        spl: MtSpline = parent.tomogram.splines[i]
        spl.orientation = orientation
        parent.GlobalProperties.params.params2.polarity.txt = str(orientation)
        return None
    
    @num.connect
    def _num_changed(self):
        from .main import MTPropsWidget
        i = self.num
        if i is None:
            return
        parent = self.find_ancestor(MTPropsWidget)
        tomo = parent.tomogram
        spl = tomo.splines[i]
        
        if spl.localprops is not None:
            n_anc = len(spl.localprops)
        else:
            parent.LocalProperties._init_text()
            parent.LocalProperties._init_plot()
            if spl._anchors is not None:
                n_anc = len(spl._anchors)
            else:
                self.pos = 0
                self["pos"].max = 0
                return
            
        self["pos"].max = n_anc - 1
        self._load_projection()
        self._update_canvas()
        return None
    
    def _load_projection(self):
        from .main import MTPropsWidget
        i = self.num
        parent = self.find_ancestor(MTPropsWidget)
        tomo = parent.tomogram
        spl = tomo.splines[i]
        
        # update plots in pyqtgraph, if properties exist
        parent.LocalProperties._plot_properties(spl.localprops)
        
        # calculate projection
        if spl.localprops is not None:
            npf_list = spl.localprops[H.nPF]
        elif spl.globalprops is not None:
            npf_list = [spl.globalprops[H.nPF]] * spl.anchors.size
        else:
            npf_list = [0] * spl.anchors.size
            # return None

        binsize = tomo.metadata["binsize"]
        imgb = parent.layer_image.data
        
        length_px = tomo.nm2pixel(tomo.subtomo_length/binsize)
        width_px = tomo.nm2pixel(tomo.subtomo_width/binsize)
        
        mole = spl.anchors_to_molecules()
        coords = mole.cartesian((width_px, length_px, width_px), spl.scale*binsize)
        out: List[ip.ImgArray] = []
        with ip.silent():
            for crds in coords:
                mapped = map_coordinates(imgb, crds, order=1, mode=Mode.constant, cval=np.mean)
                out.append(ip.asarray(mapped, axes="zyx"))
        
            projections = []
            for img, npf in zip(out, npf_list):
                proj = Projections(img)
                if npf > 1:
                    proj.rotational_average(npf)
                projections.append(proj)
        
        self.projections = projections
        return None

    @pos.connect
    def _update_canvas(self):
        from .main import MTPropsWidget
        parent = self.find_ancestor(MTPropsWidget)
        tomo = parent.tomogram
        binsize = tomo.metadata["binsize"]
        i = self.num
        j = self.pos
        
        if not self.projections or i is None or j is None:
            return
        spl = tomo.splines[i]
        # Set projections
        with ip.silent():
            proj = self.projections[j]
            for ic in range(3):
                self.canvas[ic].layers.clear()
            self.canvas[0].image = proj.yx
            self.canvas[1].image = proj.zx
            if proj.zx_ave is not None:
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
                                 [ymin, ymax, ymax, ymin, ymin], 
                                 color="lime")

        # draw two circles in ZX-view
        theta = np.linspace(0, 2*np.pi, 360)
        r = r_px * GVar.inner
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        r = r_px * GVar.outer
        self.canvas[1].add_curve(r*np.cos(theta) + lx/2, r*np.sin(theta) + lz/2, color="lime")
        
        # update pyqtgraph
        if spl.localprops is not None:
            x = spl.localprops[H.splDistance][j]
            parent.LocalProperties._plot_spline_position(x)
        else:
            parent.LocalProperties._init_plot()
    
    def _reset_contrast_limits(self):
        for i in range(3):
            img = self.canvas[i].image
            if img is not None:
                self.canvas[i].contrast_limits = [img.min(), img.max()]
        return None
