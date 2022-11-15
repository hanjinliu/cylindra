from typing import Any, TYPE_CHECKING, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt

from magicgui.widgets import RangeSlider
from magicclass import (
    do_not_record, magicclass, magicmenu, MagicTemplate, set_design, set_options, field, 
    vfield, impl_preview, confirm
)
from magicclass.types import Bound, OneOf, Optional
from magicclass.utils import thread_worker
from magicclass.widgets import Separator
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.vispy import Vispy3DCanvas

import numpy as np
import impy as ip

from acryo import TomogramSimulator

from cylindra.components import CylTomogram, CylinderModel, CylSpline, indexer as Idx, RadonModel
from cylindra.const import nm, GVar, H
from cylindra.utils import roundint
from cylindra.widgets.widget_utils import FileFilter

if TYPE_CHECKING:
    from magicclass.ext.vispy import layer3d as layers

INTERPOLATION_CHOICES = (("nearest", 0), ("linear", 1), ("cubic", 3))
SAVE_MODE_CHOICES = (("separate mrc", "mrc"), ("separate tif", "tif"), ("single tif (TZYX)", "stack"))

_INTERVAL = (GVar.yPitchMin + GVar.yPitchMax) / 2
_NPF = (GVar.nPFmin + GVar.nPFmax) // 2
_RADIUS = _INTERVAL * _NPF / 2 / np.pi

class CylinderParameters:
    """Parameters for cylinder model."""

    interval: nm = _INTERVAL
    skew: float = (GVar.minSkew + GVar.maxSkew) / 2
    rise: float = 0.0
    npf: int = _NPF
    radius: nm = _RADIUS
    offsets: "tuple[nm, float]" = (0.0, 0.0)
    
    def update(self, other: dict[str, Any] = {}, **kwargs) -> None:
        """Update parameters"""
        kwargs = dict(**other, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        return None
    
    def asdict(self) -> dict[str, Any]:
        """Return parameters as a dictionary."""
        return {
            "interval": self.interval,
            "skew": self.skew,
            "rise": self.rise,
            "npf": self.npf,
            "radius": self.radius,
            "offsets": self.offsets,
        }

# Main widget class
@magicclass(widget_type="split", labels=False, layout="horizontal")
class CylinderSimulator(MagicTemplate):
    @magicmenu
    class Menu(MagicTemplate):
        def create_empty_image(self): ...
        def set_current_spline(self): ...
        def load_spline_parameters(self): ...
        sep0 = Separator()
        def simulate_tomogram(self): ...
        def simulate_tomogram_batch(self): ...
        sep1 = Separator()
        def save_image(self): ...
        def send_moleclues_to_viewer(self): ...
        def show_layer_control(self): ...

    def __post_init__(self) -> None:
        self._model: CylinderModel = None
        self._parameters = CylinderParameters()
        self._spline: CylSpline = None
        self._spline_arrow: layers.Arrows3D = None
        self._points: layers.Points3D = None
        self._selections: layers.Points3D = None

    def _set_model(self, model: CylinderModel, spl: CylSpline):
        self._model = model
        self._spline = spl
        mole = model.to_molecules(spl)
        
        if self._points is None:
            self._points = self.canvas.add_points(mole.pos, size=2.0, face_color="lime", edge_color="lime")
            self._selections = self.canvas.add_points(
                [[0, 0, 0]], size=2.0, face_color=[0, 0, 0, 0], edge_color="cyan", edge_width=1.5, spherical=False
            )
            self._selections.visible = False
            arrow_data = np.expand_dims(spl.partition(100), axis=0)
            self._spline_arrow = self.canvas.add_arrows(arrow_data, arrow_size=15, width=2.0)
            self._points.signals.size.connect_setattr(self._selections, "size")
        else:
            self._points.data = mole.pos
            self._select_molecules(self.Operator.yrange, self.Operator.arange)
        self._molecules = mole
        return None
    
    @magicclass
    class Operator(MagicTemplate):
        """
        Apply local structural changes to the molecules.

        Attributes
        ----------
        yrange : tuple of int
            Selected range in axial direction.
        arange : tuple of int
            Selected range in angular direction (selected protofilaments).
        n_allev : int
            Number of iteration of alleviation.
        show_selection : bool
            Check to show all the selected molecules
        """
        yrange = vfield(Tuple[int, int], label="axial", widget_type=RangeSlider, record=False)
        arange = vfield(Tuple[int, int], label="angular", widget_type=RangeSlider, options={"value": (0, 100)}, record=False)
        n_allev = vfield(1, label="alleviate", options={"min": 0, "max": 20}, record=False)
        show_selection = vfield(True, label="show selected molecules", record=False)
        
        def __post_init__(self):
            self.min_width = 300

        def _update_slider_lims(self, ny: int, na: int):
            amax_old = self["arange"].max
            self["yrange"].max = ny
            self["arange"].max = na
            if self.arange[1] == amax_old:
                self.arange = (self.arange[0], na)
        
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
        
        def update_model(self): ...
        def expand(self): ...
        def screw(self): ...
        def dilate(self): ...
        
        def _fill_shift(self, yrange, arange, val: float):
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            shift = np.zeros(parent.model.shape, dtype=np.float32)
            ysl = slice(*yrange)
            asl = slice(*arange)
            shift[ysl, asl] = val
            return shift, Idx[ysl, asl]
        
    canvas = field(Vispy3DCanvas)  # the 3D viewer

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
    @confirm(text="You may have unsaved data. Continue?", condition="self.parent_widget._need_save")
    @set_options(
        size={"label": "image size of Z, Y, X (nm)"},
        scale={"label": "pixel scale (nm/pixel)"},
        bin_size={"options": {"min": 1, "max": 8}}
    )
    def create_empty_image(
        self, 
        size: tuple[nm, nm, nm] = (100., 200., 100.), 
        scale: nm = 0.25,
        bin_size: list[int] = [4],
    ):
        """
        Create an empty image with the given size and scale, and send it to the viewer.

        Parameters
        ----------
        size : tuple[nm, nm, nm], default is (100., 200., 100.)
            Size of the image in nm, of (Z, Y, X).
        scale : nm, default is 0.25
            Pixel size of the image.
        bin_size : list of int, default is [4]
            List of binning to be applied to the image for visualization.
        """
        parent = self.parent_widget
        shape = tuple(roundint(s / scale) for s in size)
        img = ip.random.normal(size=shape, axes="zyx", name="simulated image")  # TODO: just for now
        img.scale_unit = "nm"
        bin_size = sorted(list(set(bin_size)))  # delete duplication
        tomo = CylTomogram.from_image(img, scale=scale, binsize=bin_size)
        parent._macro_offset = len(parent.macro)
        parent.tomogram = tomo
        return thread_worker.to_callback(parent._send_tomogram_to_viewer, False)
    
    def _get_current_index(self, *_) -> int:
        parent = self.parent_widget
        return parent.SplineControl.num
    
    def _select_molecules(self, yrange: tuple[int, int], arange: tuple[int, int]):
        points = self._points.data
        npf = self._parameters.npf
        ysl = slice(*yrange)
        asl = slice(*arange)
        try:
            selected_points = points.reshape(-1, npf, 3)[ysl, asl].reshape(-1, 3)
            self._selections.data = selected_points
        except Exception:
            # A little bit hacky, but points data shape sometimes mismatches with the model shape.
            pass
        self._selections.visible = True

    @Menu.wraps
    def save_image(self, path: Path, dtype: OneOf[np.int8, np.int16, np.float32] = np.float32):
        """Save the current image to a file."""
        img = self.parent_widget.tomogram.image.compute()
        if np.dtype(dtype).kind == "i":
            amax = max(-img.min(), img.max())
            img = (img / amax * np.iinfo(dtype).max)
        img.imsave(path, dtype=dtype)
        return None

    @Menu.wraps
    def set_current_spline(self, idx: Bound[_get_current_index]):
        """Use the current parameters and the spline to construct a model and molecules."""
        self._spline = self.parent_widget.tomogram.splines[idx]
        self.canvas.layers.clear()
        self._points = None
        self._spline_arrow = None
        self.update_model(idx, **self._parameters.asdict())
        return None
    
    @Menu.wraps
    def load_spline_parameters(self, idx: Bound[_get_current_index]):
        """Copy the spline parameters in the viewer."""
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        props = spl.globalprops
        if props is None:
            raise ValueError("Global property is not calculated yet.")
        self._parameters.update(
            interval=props[H.yPitch],
            skew=props[H.skewAngle],
            rise=props[H.riseAngle],
            npf=props[H.nPF],
            radius=spl.radius,
        )
        return None

    @Menu.wraps
    def send_moleclues_to_viewer(self):
        """Send the current molecules to the viewer."""
        mole = self._molecules
        if mole is None:
            raise ValueError("Molecules are not generated yet.")
        self.parent_widget.add_molecules(mole, name="Simulated")
        return None
        
    @Menu.wraps
    @do_not_record
    def show_layer_control(self):
        """Open layer control widget."""
        points = self._points
        if points is None:
            raise ValueError("No layer found in this viewer.")
        cnt = self._points.widgets.as_container()
        cnt.native.setParent(self.native, cnt.native.windowFlags())
        cnt.show()
        return None

    @Operator.wraps
    @impl_preview(auto_call=True)
    @set_options(
        interval={"min": 0.2, "max": GVar.yPitchMax * 2, "step": 0.01, "label": "interval (nm)"},
        skew={"min": GVar.minSkew, "max": GVar.maxSkew, "label": "skew (deg)"},
        rise={"min": -90.0, "max": 90.0, "step": 0.5, "label": "rise (deg)"},
        npf={"min": GVar.nPFmin, "max": GVar.nPFmax, "label": "nPF"},
        radius={"min": 0.5, "max": 50.0, "step": 0.5, "label": "radius (nm)"},
        offsets={"options": {"min": -30.0, "max": 30.0}, "label": "offsets (nm, rad)"},
    )
    @set_design(text="Update model parameters", font_color="lime")
    def update_model(
        self,
        idx: Bound[_get_current_index],
        interval: nm = CylinderParameters.interval,
        skew: float = CylinderParameters.skew,
        rise: float = CylinderParameters.rise,
        npf: int = CylinderParameters.npf,
        radius: nm = CylinderParameters.radius,
        offsets: Tuple[float, float] = CylinderParameters.offsets,
    ):
        """
        Update cylinder model with new parameters.
        
        Local structural displacement will be deleted because this function may change the number
        of molecules. This function should be called first.

        Parameters
        ----------
        idx : int
            Index of spline used in the viewer.
        interval : nm
            Axial interval between molecules.
        skew : float
            Skew angle.
        rise : float
            Rise angle.
        npf : int
            Number of protofilaments.
        radius : nm
            Radius of the cylinder.
        offsets : tuple of float
            Offset of the starting molecule.
        """
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        self._parameters.update(
            interval=interval, skew=skew, rise=rise, npf=npf, radius=radius, offsets=offsets
        )
        kwargs = {H.yPitch: interval, H.skewAngle: skew, H.riseAngle: rise, H.nPF: npf}
        model = tomo.get_cylinder_model(idx, offsets=offsets, radius=radius, **kwargs)
        self.model = model
        spl.radius = radius
        
        op = self.Operator
        op._update_slider_lims(*self.model.shape)
        self._select_molecules(op.yrange, op.arange)  # update selection coordinates
        return None

    @update_model.during_preview
    def _during_update_model(self):
        op = self.Operator
        old_model = self.model
        old_max = op["yrange"].max, op["arange"].max
        yield
        self.model = old_model
        op["yrange"].max, op["arange"].max = old_max
        return None

    def _prep_radon(
        self,
        path: Path,
        tilt_range: Tuple[float, float] = (-60.0, 60.0),
        n_tilt: int = 61,
        order: int = 3,
    ) -> Tuple[RadonModel, ip.ImgArray]:
        parent = self.parent_widget
        tomo = parent.tomogram
        template = ip.imread(path)
        scale_ratio = template.scale.x / tomo.scale
        template = template.rescale(scale_ratio)
        
        # noise-free tomogram generation from the current cylinder model
        model = self.model
        mole = model.to_molecules(self._spline)
        scale = tomo.scale
        simulator = TomogramSimulator(order=order, scale=scale)
        simulator.add_molecules(molecules=mole, image=template)
        simulated_image = ip.asarray(simulator.simulate(tomo.image.shape), like=template)
        parent.log.print_html(f"Tomogram of shape {tuple(simulated_image.shape)!r} is generated.")
        
        # tilt ranges to array
        radon_model = RadonModel(
            range=(tilt_range[0], tilt_range[1], n_tilt),
            height=simulated_image.shape[0],
            order=order,
        )
        
        return radon_model, radon_model.transform(simulated_image)
        
    @Menu.wraps
    @set_options(
        path={"label": "Template image", "filter": FileFilter.IMAGE},
        tilt_range={"label": "Tilt range (deg)", "widget_type": "FloatRangeSlider", "min": -90.0, "max": 90.0},
        bin_size={"options": {"min": 1, "max": 10}},
        nsr={"min": 0.0, "max": 4.0, "step": 0.1},
    )
    @thread_worker(progress={"desc": "Simulating a tomogram"})
    def simulate_tomogram(
        self,
        path: Path,
        nsr: float = 2.0,
        tilt_range: Tuple[float, float] = (-60.0, 60.0),
        n_tilt: int = 61,
        bin_size: list[int] = [4],
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        filter: bool = True,
    ):
        """
        Simulate a tomographic image using the current model.
        
        Parameters
        ----------
        path : Path
            Path to the image used for the template.
        nsr : float
            Noise-to-signal ratio.
        tilt_range : tuple of float
            Minimum and maximum tilt angles.
        n_tilt : int
            Number of tilt angles.
        bin_size : list of int
            List of binning to be applied to the image for visualization.
        interpolation : int
            Interpolation method used during the simulation.
        filter : bool, default is True
            Apply low-pass filter on the reference image (does not affect image data itself).
        """
        parent = self.parent_widget
        template = ip.imread(path)
        scale_ratio = template.scale.x / parent.tomogram.scale
        template = template.rescale(scale_ratio)
        
        radon_model, sino = self._prep_radon(path, tilt_range, n_tilt, interpolation)
        
        # add noise
        if nsr > 0:
            imax = template.max()
            sino += ip.random.normal(scale=imax * nsr, size=sino.shape, axes=sino.axes)
        
        # back projection
        parent.log.print_html("Running inverse Radon transformation.")
        rec = radon_model.inverse_transform(sino)
        tomo = CylTomogram.from_image(rec, binsize=bin_size)
        parent.tomogram = tomo
        return thread_worker.to_callback(parent._send_tomogram_to_viewer, filter)

    def _directory_not_empty(self):
        path = Path(self["simulate_tomogram_batch"].mgui.save_path.value)
        try:
            next(path.glob("*"))
        except StopIteration:
            return True
        return False
        
    @Menu.wraps
    @set_options(
        path={"label": "Template image", "filter": FileFilter.IMAGE},
        save_path={"label": "Save directory", "mode": "d"},
        tilt_range={"label": "Tilt range (deg)", "widget_type": "FloatRangeSlider", "min": -90.0, "max": 90.0},
        nsr={"label": "N/S ratio", "options": {"min": 0.0, "max": 4.0, "step": 0.1}},
    )
    @confirm(text="Directory already exists. Overwrite?", condition=_directory_not_empty)
    @thread_worker(progress={"desc": "Simulating tomograms", "total": "len(nsr) + 1"})
    def simulate_tomogram_batch(
        self,
        path: Path,
        save_path: Path,
        nsr: list[float] = [2.0],
        tilt_range: Tuple[float, float] = (-60.0, 60.0),
        n_tilt: int = 61,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        save_mode: OneOf[SAVE_MODE_CHOICES] = "mrc",
        seed: Optional[int] = None,
    ):
        """
        Simulate a tomographic image using the current model.
        
        Parameters
        ----------
        path : Path
            Path to the image used for the template.
        nsr : float
            Noise-to-signal ratio.
        tilt_range : tuple of float
            Minimum and maximum tilt angles.
        n_tilt : int
            Number of tilt angles.
        interpolation : int
            Interpolation method used during the simulation.
        save_mode : ("mrc", "tif", "stack")
            Specify how to save images.
        """
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir()
        if save_mode not in ("mrc", "tif", "stack"):
            raise ValueError(f"Invalid save mode {save_mode!r}.")
        
        parent = self.parent_widget
        template = ip.imread(path)
        scale_ratio = template.scale.x / parent.tomogram.scale
        template = template.rescale(scale_ratio)
        
        radon_model, sino = self._prep_radon(path, tilt_range, n_tilt, interpolation)
        
        # plot some of the results
        @thread_worker.to_callback
        def _on_radon_finished():
            if n_tilt < 3:
                return
            degs = radon_model.range.asarray()
            _, ny, nx = sino.shape
            ysize = max(4 / nx * ny, 4)
            with parent.log.set_plt(rc_context={"font.size": 15}):
                _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, ysize))
                for i, idx in enumerate([0, n_tilt//2, -1]):
                    axes[i].imshow(sino[idx], cmap="gray")
                    axes[i].set_title(f"deg = {degs[idx]:.1f}")
                    axes[i].set_axis_off()
                
                plt.show()
        
        yield _on_radon_finished

        # plot some of the results
        @thread_worker.to_callback
        def _on_iradon_finished(rec: ip.ImgArray, title: str):
            with parent.log.set_plt():
                plt.imshow(rec.proj("z"), cmap="gray")
                plt.title(title)
                plt.show()
                
        # add noise and save image
        recs: list[ip.ImgArray] = []
        rng = ip.random.default_rng(seed)
        for val in nsr:
            imax = sino.max()
            sino_noise = sino + rng.normal(scale=imax * val, size=sino.shape, axes=sino.axes)
            rec = radon_model.inverse_transform(sino_noise)
            recs.append(rec)
            yield _on_iradon_finished(rec, f"N/S = {val:.1f}")
        
        if save_mode in ("mrc", "tif"):
            for i, rec in enumerate(recs):
                rec.imsave(save_path / f"image-{i}.{save_mode}")
        else:
            stack: ip.ImgArray = np.stack(recs, axis="t")
            stack.imsave(save_path / "images.tif")

        nsr_info = {i: val for i, val in enumerate(nsr)}
        js = {"settings": radon_model.dict(), "nsr": nsr_info}
        with open(save_path / "simulation_info.json", "w") as f:
            json.dump(js, f, indent=4, separators=(", ", ": "))
        return None 

    @Operator.wraps
    @set_options(shift={"min": -1.0, "max": 1.0, "step": 0.01, "label": "shift (nm)"})
    @set_design(text="Expansion/Compaction", font_color="lime")
    @impl_preview(auto_call=True)
    def expand(
        self,
        shift: nm,
        yrange: Bound[Operator.yrange],
        arange: Bound[Operator.arange],
        n_allev: Bound[Operator.n_allev] = 1,
    ):
        """Expand the selected molecules."""
        shift_arr, sl = self.Operator._fill_shift(yrange, arange, shift)
        new_model = self.model.expand(shift, sl)
        if n_allev > 0:
            new_model = new_model.alleviate(shift_arr != 0, niter=n_allev)
        self.model = new_model
        return None
    
    @Operator.wraps
    @set_options(skew={"min": -45.0, "max": 45.0, "step": 0.05, "label": "skew (deg)"})
    @set_design(text="Screw", font_color="lime")
    @impl_preview(auto_call=True)
    def screw(
        self, 
        skew: float,
        yrange: Bound[Operator.yrange], 
        arange: Bound[Operator.arange],
        n_allev: Bound[Operator.n_allev] = 1,
    ):
        """Screw (change the skew angles of) the selected molecules."""
        shift, sl = self.Operator._fill_shift(yrange, arange, skew)
        new_model = self.model.screw(np.deg2rad(skew), sl)
        if n_allev > 0:
            new_model = new_model.alleviate(shift != 0, niter=n_allev)
        self.model = new_model
        return None
    
    @Operator.wraps
    @set_options(radius={"min": -1.0, "max": 1.0, "step": 0.1, "label": "radius (nm)"})
    @set_design(text="Dilation/Erosion", font_color="lime")
    @impl_preview(auto_call=True)
    def dilate(
        self,
        radius: nm,
        yrange: Bound[Operator.yrange],
        arange: Bound[Operator.arange],
        n_allev: Bound[Operator.n_allev] = 1,
    ):
        """Dilate (increase the local radius of) the selected molecules."""
        shift, sl = self.Operator._fill_shift(yrange, arange, radius)
        new_model = self.model.dilate(radius, sl)
        if n_allev > 0:
            new_model = new_model.alleviate(shift != 0, niter=n_allev)
        self.model = new_model
        return None
    
    @expand.during_preview
    @screw.during_preview
    @dilate.during_preview
    def _prev_context(self):
        """Temporarily update the layers."""
        original = self.model
        yield
        self.model = original