from typing import Any, TYPE_CHECKING, Tuple
from magicgui.widgets import RangeSlider
from magicclass import (
    magicclass, magicmenu, MagicTemplate, set_design, set_options, field, 
    vfield, FieldGroup, impl_preview
)
from magicclass.types import Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.vispy import Vispy3DCanvas
import numpy as np
import impy as ip

from ..components import CylTomogram, CylinderModel, CylSpline, indexer as Idx
from ..const import nm, GVar, H
from ..utils import roundint

if TYPE_CHECKING:
    from magicclass.ext.vispy import layer3d as layers

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
    yoffset = vfield(0.0, options={"min": -10, "max": 10}, label="y")
    aoffset = vfield(0.0, options={"min": -np.pi, "max": np.pi}, label="θ")

    @property
    def value(self):
        return self.yoffset, self.aoffset

_INTERVAL = (GVar.yPitchMin + GVar.yPitchMax) / 2
_NPF = (GVar.nPFmin + GVar.nPFmax) // 2
_RADIUS = _INTERVAL * _NPF / 2 / np.pi

class CylinderParameters(FieldGroup):
    """
    Parameters that define a regular cylindric structure.

    Attributes
    ----------
    interval : nm
        Interval between adjacent molecules in the axial (y) direction.
    skew : float
        Skew angle in degree.
    rise : float
        Rise angle in degree.
    npf : int
        Number of protofilamnet.
    radius : nm
        Radius of cylinder.
    offsets : tuple of float
        Offsets of axial and angular coordinates.
    """    
    interval = vfield(_INTERVAL, options={"min": 0.0, "max": GVar.yPitchMax * 2, "step": 0.2}, label="interval (nm)")
    skew = vfield((GVar.minSkew + GVar.maxSkew) / 2, options={"min": GVar.minSkew, "max": GVar.maxSkew}, label="skew (deg)")
    rise = vfield(0.0, options={"min": -90.0, "max": 90.0, "step": 0.5}, label="rise (deg)")
    npf = vfield(_NPF, options={"min": GVar.nPFmin, "max": GVar.nPFmax}, label="nPF")
    radius = vfield(_RADIUS, options={"min": 0.5, "max": 50.0, "step": 0.5}, label="radius (nm)")
    offsets = CylinderOffsets()
    
    def as_kwargs(self) -> dict[str, Any]:
        return {
            H.yPitch: self.interval,
            H.skewAngle: self.skew,
            H.riseAngle: self.rise,
            H.nPF: self.npf,
            "radius": self.radius,
            "offsets": self.offsets.value,
        }
    
    def update(self, other: dict[str, Any], **kwargs) -> None:
        kwargs = dict(**other, **kwargs)
        with self.changed.blocked():
            for k, v in kwargs:
                setattr(self, k, v)
        self.changed.emit(self)
        return None

@magicclass(widget_type="split", labels=False, layout="horizontal")
class CylinderSimulator(MagicTemplate):
    @magicmenu
    class Menu(MagicTemplate):
        def create_empty_image(self): ...
        def set_current_spline(self): ...
        def load_spline_parameters(self): ...
        def show_layer_control(self): ...
        def open_operator(self): ...

    @magicclass(widget_type="split", labels=False)
    class Left(MagicTemplate):
        parameters = CylinderParameters()
    
    @property
    def parameters(self):
        return self.Left.parameters

    canvas = field(Vispy3DCanvas)

    def __post_init__(self) -> None:
        self._model: CylinderModel = None
        self._spline: CylSpline = None
        self._spline_arrow: layers.Arrows3D = None
        self._points: layers.Points3D = None
        self._selections: layers.Points3D = None
        self.parameters.max_width = 230

    def _set_model(self, model: CylinderModel, spl: CylSpline):
        self._model = model
        self._spline = spl
        mole = model.to_molecules(spl)
        
        if self._points is None:
            self._points = self.canvas.add_points(mole.pos, size=2.0, face_color="lime", edge_color="lime")
            self._selections = self.canvas.add_points(
                [[0, 0, 0]], size=2.0, face_color=[0, 0, 0, 0], edge_color="cyan", edge_width=1.5, spherical=False
            )
            self._spline_arrow = self.canvas.add_arrows(np.expand_dims(spl.partition(100), axis=0), arrow_size=15, width=2.0)
            self._points.signals.size.connect_setattr(self._selections, "size")
            self._selections.visible = False
        else:
            self._points.data = mole.pos
        return None
    
    @magicclass(popup_mode="below")
    class Operator(MagicTemplate):
        yrange = vfield(Tuple[int, int], widget_type=RangeSlider)
        arange = vfield(Tuple[int, int], widget_type=RangeSlider, options={"value": (0, 100)})
        n_allev = vfield(1, options={"min": 0, "max": 20, "label": "number of alleviation"})
        show_selection = vfield(True, record=False)
        
        def _set_shape(self, ny, na):
            self["yrange"].max = ny
            self["arange"].max = na
        
        @yrange.connect
        @arange.connect
        def _on_range_changed(self):
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            parent._select_molecules(self.yrange, self.arange)
            return None
        
        @show_selection.connect
        def _on_show_selection_changed(self, show: bool):
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            parent._selections.visible = show
            return None
        
        @set_options(shift={"min": -1.0, "max": 1.0, "step": 0.01, "label": "shift (nm)"})
        @set_design(text="Expansion/Compaction", font_color="lime")
        @impl_preview(auto_call=True)
        def expand(
            self,
            shift: nm,
            yrange: Bound[yrange],
            arange: Bound[arange],
            n_allev: Bound[n_allev] = 1,
        ):
            """Expand the selected molecules."""
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            shift_arr, sl = self._fill_shift(yrange, arange, shift)
            new_model = parent.model.expand(shift, sl)
            if n_allev > 0:
                new_model = new_model.alleviate(shift_arr != 0, niter=n_allev)
            parent.model = new_model
            return None
        
        @set_options(skew={"min": -45.0, "max": 45.0, "step": 0.01, "label": "skew (deg)"})
        @set_design(text="Screw", font_color="lime")
        @impl_preview(auto_call=True)
        def screw(
            self, 
            skew: float,
            yrange: Bound[yrange], 
            arange: Bound[arange],
            n_allev: Bound[n_allev] = 1,
        ):
            """Screw (change the skew angles of) the selected molecules."""
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            shift, sl = self._fill_shift(yrange, arange, skew)
            new_model = parent.model.screw(skew, sl)
            if n_allev > 0:
                new_model = new_model.alleviate(shift != 0, niter=n_allev)
            parent.model = new_model
            return None
        
        @set_options(radius={"min": -1.0, "max": 1.0, "step": 0.01, "label": "radius (nm)"})
        @set_design(text="Dilation/Erosion", font_color="lime")
        @impl_preview(auto_call=True)
        def dilate(
            self,
            radius: nm,
            yrange: Bound[yrange],
            arange: Bound[arange],
            n_allev: Bound[n_allev] = 1,
        ):
            """Dilate (increase the local radius of) the selected molecules."""
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            shift, sl = self._fill_shift(yrange, arange, radius)
            new_model = parent.model.dilate(radius, sl)
            if n_allev > 0:
                new_model = new_model.alleviate(shift != 0, niter=n_allev)
            parent.model = new_model
            return None
        
        @expand.during_preview
        @screw.during_preview
        @dilate.during_preview
        def _prev_context(self):
            """Temporarily update the layers."""
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            original = parent.model
            yield
            parent.model = original
        
        def _fill_shift(self, yrange, arange, val: float):
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            shift = np.zeros(parent.model.shape, dtype=np.float32)
            ysl = slice(*yrange)
            asl = slice(*arange)
            shift[ysl, asl] = val
            return shift, Idx[ysl, asl]
        
    @property
    def parent_widget(self):
        from .main import CylindraMainWidget
        return self.find_ancestor(CylindraMainWidget, cache=True)
    
    @property
    def model(self) -> CylinderModel:
        """Current cylinder model."""
        return self._model
    
    @model.setter
    def model(self, model: CylinderModel):
        """Set new model and simulate molecules with the same spline."""
        return self._set_model(model, self._spline)
    
    @Menu.wraps
    @dask_thread_worker(progress={"desc": "Creating an image"})
    @set_options(bin_size={"options": {"min": 1, "max": 8}})
    def create_empty_image(
        self, 
        size: tuple[nm, nm, nm] = (100., 200., 100.), 
        scale: nm = 0.25,
        bin_size: list[int] = [4],
    ):
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
    
    def _select_molecules(self, yrange: tuple[int, int], arange: tuple[int, int]):
        points = self._points.data
        npf = self.parameters.npf
        ysl = slice(*yrange)
        asl = slice(*arange)
        selected_points = points.reshape(-1, npf, 3)[ysl, asl].reshape(-1, 3)
        self._selections.data = selected_points
        self._selections.visible = True
    
    def _deselect_molecules(self):
        self._selections.data = np.zeros((1, 3), dtype=np.float32)
        self._selections.visible = False

    @Menu.wraps
    def set_current_spline(self, idx: Bound[_get_current_index]):
        """Use the current parameters and the spline to construct a model and molecules."""
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        params = self.parameters.as_kwargs()
        model = tomo.get_cylinder_model(idx, **params)
        spl.radius = params.pop("radius")
        params.pop("offsets")
        self._set_model(model, spl)
        
        return None
    
    @Menu.wraps
    def load_spline_parameters(self, idx: Bound[_get_current_index]):
        """Copy the spline parameters in the viewer."""
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        props = spl.globalprops
        if props is None:
            raise ValueError("Global property is not calculated yet.")
        params = {
            H.yPitch: props[H.yPitch],
            H.skewAngle: props[H.skewAngle],
            H.riseAngle: props[H.riseAngle],
            H.nPF: props[H.nPF],
            "radius": spl.radius,
        }
        self.parameters.update(params)
        return None

    @Menu.wraps
    def show_layer_control(self):
        points = self._points
        if points is None:
            raise ValueError("No layer found in this viewer.")
        cnt = self._points.widgets.as_container()
        self.Left.append(cnt)
        return None
    
    @Menu.wraps
    def open_operator(self):
        self.Operator._set_shape(*self.model.shape)
        self.Operator.show()
        return None
        
    @Left.parameters.connect
    def _on_param_changed(self):
        idx = self._get_current_index()
        
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        params = self.parameters.as_kwargs()
        model = tomo.get_cylinder_model(idx, **params).add_shift(self.model.displace)
        spl.radius = params.pop("radius")
        params.pop("offsets")
        self.model = model
        
        op = self.Operator
        self._select_molecules(op.yrange, op.arange)  # update selection coordinates
