from typing import Any
from magicgui.widgets import RangeSlider
from magicclass import magicclass, MagicTemplate, set_options, field, vfield, FieldGroup
from magicclass.types import Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.vispy import VispyPlotCanvas
import numpy as np
import impy as ip

from ..components import CylTomogram, CylinderModel, CylSpline
from ..const import nm, GVar, H
from ..utils import roundint


class CylinderOffsets(FieldGroup):
    """
    Widget for specifying cylinder tilts.

    Attributes
    ----------
    yoffset : float
        Offset of axial direction.
    aoffset : float
        Offset of angular direction.
    """
    yoffset = vfield(0.0, options={"min": -10, "max": 10})
    aoffset = vfield(0.0, options={"min": -np.pi, "max": np.pi})

    @property
    def value(self):
        return self.yoffset, self.aoffset

class CylinderParameters(FieldGroup):
    interval = vfield((GVar.yPitchMin + GVar.yPitchMax) / 2, options={"min": 0.0, "max": GVar.yPitchMax * 2})
    skew = vfield((GVar.minSkew + GVar.maxSkew) / 2, options={"min": GVar.minSkew, "max": GVar.maxSkew})
    rise = vfield(0.0)
    npf = vfield((GVar.nPFmin + GVar.nPFmax) // 2, options={"min": GVar.nPFmin, "max": GVar.nPFmax})
    radius = vfield(1.0)
    offsets = CylinderOffsets(layout="horizontal")
    
    def as_kwargs(self) -> dict[str, Any]:
        return {
            H.yPitch: self.interval,
            H.skewAngle: self.skew,
            H.riseAngle: self.rise,
            H.nPF: self.npf,
            "radius": self.radius,
            "offsets": self.offsets.value,
        }

@magicclass
class CylinderSimulator(MagicTemplate):
    @magicclass
    class CylinderModelViewer(MagicTemplate):
        parameters = CylinderParameters()
        canvas = field(VispyPlotCanvas)

        def __init__(self) -> None:
            self._model: CylinderModel = None

        def _set_model(self, model: CylinderModel):
            self._model = model
            return self._update_canvas()
            
        def _update_canvas(self, spl=None):
            self.canvas.layers.clear()
            if spl is None:
                spl = CylSpline().fit_voa([[0, 0, 0], [0, 100, 0]])
            mole = self._model.to_molecules(spl)
            self.canvas.add_scatter(mole.pos[:, 2], mole.pos[:, 1], size=1.0, edge_color="lime", symbol="o")
            return None

    @property
    def parent_widget(self):
        from .main import CylindraMainWidget
        return self.find_ancestor(CylindraMainWidget)
    
    @dask_thread_worker(progress={"desc": "Creating an image"})
    @set_options(bin_size={"options": {"min": 1, "max": 8}})
    def create_empty_image(self, size: tuple[nm, nm, nm] = (100., 200., 100.), scale: nm = 0.25, bin_size: list[int] = [1]):
        parent = self.parent_widget
        shape = tuple(roundint(s / scale) for s in size)
        img = ip.random.normal(size=shape, axes="zyx", name="simulated image")  # TODO: just for now
        img.scale_unit = "nm"
        bin_size = list(set(bin_size))  # delete duplication
        tomo = CylTomogram.from_image(img, scale=scale, binsize=bin_size)
        
        parent._macro_offset = len(parent.macro)
        parent.tomogram = tomo
        return thread_worker.to_callback(parent._send_tomogram_to_viewer, False)
    
    def _get_current_index(self, *_) -> int:
        parent = self.parent_widget
        return parent.SplineControl.num

    def set_current_spline(self, idx: Bound[_get_current_index]):
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        if spl.radius is not None and spl.globalprops is not None:
            model = tomo.get_cylinder_model(idx)
        else:
            params = self.CylinderModelViewer.parameters.as_kwargs()
            model = tomo.get_cylinder_model(idx, **params)
        self.CylinderModelViewer._set_model(model)
        return None

    @CylinderModelViewer.parameters.connect
    def _on_param_changed(self):
        return self.set_current_spline(self._get_current_index())
