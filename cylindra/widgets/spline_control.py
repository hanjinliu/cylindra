import numpy as np
import impy as ip
from magicclass import magicclass, MagicTemplate, field, vfield, set_options, set_design, abstractapi
from magicclass.types import Bound, OneOf
from magicclass.ext.pyqtgraph import QtMultiImageCanvas

from cylindra.const import GlobalVariables as GVar, Ori, PropertyNames as H, Mode
from cylindra.utils import map_coordinates, Projections

@magicclass(widget_type="groupbox", name="Spline Control")
class SplineControl(MagicTemplate):
    """
    Control and visualization along splines
    
    Attributes
    ----------
    num : int
        Splines in current tomogram.
    pos : int
        Position along the spline.
    canvas : QtMultiImageCanvas
        2-D projections of subtomogram at current position.
    """
    def __post_init__(self):
        self.projections: list[Projections] = []
        
        self.canvas.min_height = 200
        self.canvas.max_height = 230
        self.canvas[0].lock_contrast_limits = True
        self.canvas[0].title = "XY-Projection"
        self.canvas[1].lock_contrast_limits = True
        self.canvas[1].title = "XZ-Projection"
        self.canvas[2].lock_contrast_limits = True
        self.canvas[2].title = "Rot. average"
        
        self.canvas.enabled = False
        
    def _get_splines(self, widget=None) -> list[int]:
        """Get list of spline objects for categorical widgets."""
        from .main import CylindraMainWidget
        try:
            tomo = self.find_ancestor(CylindraMainWidget).tomogram
        except Exception:
            return []
        if tomo is None:
            return []
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    num = vfield(OneOf[_get_splines], label="Spline No.", record=False)
    pos = vfield(int, widget_type="Slider", label="Position", record=False).with_options(max=0)
    canvas = field(QtMultiImageCanvas, name="Figure", ).with_options(nrows=1, ncols=3)
    
    @magicclass(layout="horizontal")
    class footer(MagicTemplate):
        focus = vfield(False, record=False).with_options(text="focus on", tooltip="Keep focus of viewer camera on the current spline position")
        set_pf_number = abstractapi()
        set_orientation = abstractapi()
    
    @footer.wraps
    @set_design(text="Set PF number")
    @set_options(labels=False)
    def set_pf_number(self, i: Bound[num], npf: int = 13):
        """Manually update protofilament number."""
        from .main import CylindraMainWidget
        parent = self.find_ancestor(CylindraMainWidget)
        if parent.tomogram is None or i is None:
            return None
        spl = parent.tomogram.splines[i]
        if spl.localprops is not None:
            spl.localprops[H.nPF].values[:] = npf
            parent._update_local_properties_in_widget()
        if spl.globalprops is not None:
            spl.globalprops[H.nPF] = npf
            parent._update_global_properties_in_widget()
        if self.canvas[0].image is not None:
            parent.sample_subtomograms()
        return None
        
    @footer.wraps
    @set_design(text="Set orientation")
    @set_options(labels=False, orientation={"widget_type": "RadioButtons"})
    def set_orientation(self, i: Bound[num], orientation: Ori = Ori.none):
        """Manually set polarity."""
        from .main import CylindraMainWidget
        parent = self.find_ancestor(CylindraMainWidget)
        if parent.tomogram is None or i is None:
            return None
        spl = parent.tomogram.splines[i]
        spl.orientation = orientation
        parent.GlobalProperties.params.params2.polarity.txt = str(orientation)
        parent._set_orientation_marker(i)
        return None
    
    @num.connect
    def _num_changed(self):
        from cylindra.widgets.main import CylindraMainWidget
        i = self.num
        if i is None:
            return
        parent = self.find_ancestor(CylindraMainWidget)
        tomo = parent.tomogram
        if i >= len(tomo.splines):
            return
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
        from cylindra.widgets.main import CylindraMainWidget
        i = self.num
        parent = self.find_ancestor(CylindraMainWidget)
        tomo = parent.tomogram
        if i >= len(tomo.splines):
            return
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

        binsize = parent.layer_image.metadata["current_binsize"]
        imgb = parent.tomogram.get_multiscale(binsize)
        
        length_px = tomo.nm2pixel(GVar.fitLength, binsize=binsize)
        width_px = tomo.nm2pixel(GVar.fitWidth, binsize=binsize)
        
        mole = spl.anchors_to_molecules()
        if binsize > 1:
            mole = mole.translate(-parent.tomogram.multiscale_translation(binsize))
        coords = mole.cartesian_at(
            index=slice(None),
            shape=(width_px, length_px, width_px), 
            scale=tomo.scale * binsize
        )
        out: list[ip.ImgArray] = []
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
        from .main import CylindraMainWidget
        parent = self.find_ancestor(CylindraMainWidget)
        tomo = parent.tomogram
        binsize = parent.layer_image.metadata["current_binsize"]
        i = self.num
        j = self.pos
        if i >= len(tomo.splines):
            return
        if not self.projections or i is None or j is None:
            for ic in range(3):
                self.canvas[ic].layers.clear()
            return
        spl = tomo.splines[i]
        # Set projections
        proj = self.projections[j]
        for ic in range(3):
            self.canvas[ic].layers.clear()
        self.canvas[0].image = proj.yx
        self.canvas[1].image = proj.zx
        if proj.zx_ave is not None:
            self.canvas[2].image = proj.zx_ave
        else:
            del self.canvas[2].image
        
        # Update text overlay
        self.canvas[0].text_overlay.text = f"{i}-{j}"
        self.canvas[0].text_overlay.color = "lime"
        
        if spl.radius is None:
            return None
        lz, ly, lx = np.array(proj.shape)
        
        if parent._current_ft_size is None:
            ylen = 25/binsize/tomo.scale
        else:
            ylen = parent._current_ft_size/2/binsize/tomo.scale
        
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
