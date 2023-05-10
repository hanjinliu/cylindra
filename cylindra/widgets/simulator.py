from typing import Any, TYPE_CHECKING, Annotated
import json
import matplotlib.pyplot as plt

from magicgui.widgets import RangeSlider
from magicclass import (
    abstractapi,
    magicclass,
    magicmenu,
    MagicTemplate,
    set_design,
    field,
    vfield,
    impl_preview,
    confirm,
)
from magicclass.types import Bound, OneOf, Optional, Path
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.widgets import Separator
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.vispy import Vispy3DCanvas

import numpy as np
import impy as ip

from acryo import Molecules, TomogramSimulator, pipe
from scipy.spatial.transform import Rotation

from cylindra.components import (
    CylTomogram,
    CylinderModel,
    CylSpline,
    indexer as Idx,
)
from cylindra.const import nm, GlobalVariables as GVar, PropertyNames as H
from cylindra.utils import roundint, ceilint
from cylindra.widgets.widget_utils import FileFilter

if TYPE_CHECKING:
    from magicclass.ext.vispy import layer3d as layers

INTERPOLATION_CHOICES = (("nearest", 0), ("linear", 1), ("cubic", 3))

_INTERVAL = (GVar.yPitchMin + GVar.yPitchMax) / 2
_NPF = (GVar.nPFmin + GVar.nPFmax) // 2
_RADIUS = _INTERVAL * _NPF / 2 / np.pi

_TiltRange = Annotated[
    tuple[float, float],
    {
        "label": "Tilt range (deg)",
        "widget_type": "FloatRangeSlider",
        "min": -90.0,
        "max": 90.0,
    },
]

_NSRatio = Annotated[
    list[float],
    {"label": "N/S ratio", "options": {"min": 0.0, "max": 4.0, "step": 0.1}},
]

_ImageSize = Annotated[tuple[nm, nm, nm], {"label": "image size of Z, Y, X (nm)"}]

_Logger = getLogger("cylindra")


def _simulate_tomogram_iter(nsr):
    n = len(nsr)
    yield f"(0/{n + 1}) Simulating projections"
    for i in range(n):
        yield f"({i + 1}/{n + 1}) Back-projection of {i}-th image"


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
@magicclass(widget_type="scrollable", labels=False)
class CylinderSimulator(MagicTemplate):
    @magicmenu(name="Viewer")
    class ViewerMenu(MagicTemplate):
        """Receive image or spline data from the viewer"""

        create_empty_image = abstractapi()
        set_current_spline = abstractapi()
        load_spline_parameters = abstractapi()
        create_straight_line = abstractapi()
        sep0 = field(Separator)
        send_moleclues_to_viewer = abstractapi()

    @magicmenu(name="Simulate")
    class SimulateMenu(MagicTemplate):
        """Simulate using current model."""

        simulate_tomogram = abstractapi()
        simulate_tilt_series = abstractapi()

    @magicmenu(name="Transform")
    class TransformMenu(MagicTemplate):
        """Transform the cylinder lattice."""

        update_model = abstractapi()
        expand = abstractapi()
        screw = abstractapi()
        dilate = abstractapi()

    def __post_init__(self) -> None:
        self._model: CylinderModel = None
        self._parameters = CylinderParameters()
        self._spline: CylSpline = None
        self._spline_arrow: layers.Arrows3D = None
        self._points: layers.Points3D = None
        self._selections: layers.Points3D = None
        self._layer_control = None
        self._simulate_shape = (0, 0, 0)
        self._simulate_scale = 0.0
        self.canvas.min_height = 300

    def _set_model(self, model: CylinderModel, spl: CylSpline):
        self._model = model
        self._spline = spl
        mole = model.to_molecules(spl)

        if self._points is None:
            self.canvas.layers.clear()
            self._points = self.canvas.add_points(
                mole.pos, size=2.0, face_color="lime", edge_color="lime"
            )
            self._selections = self.canvas.add_points(
                [[0, 0, 0]],
                size=2.0,
                face_color=[0, 0, 0, 0],
                edge_color="cyan",
                edge_width=1.5,
                spherical=False,
            )
            self._selections.visible = False
            arrow_data = np.expand_dims(spl.partition(100), axis=0)
            self._spline_arrow = self.canvas.add_arrows(
                arrow_data, arrow_size=15, width=2.0
            )
            self._points.signals.size.connect_setattr(self._selections, "size")
            nz, ny, nx = self._simulate_shape
            for z in [0, nz]:
                arr = (
                    np.array(
                        [[z, 0, 0], [z, 0, nx], [z, ny, nx], [z, ny, 0], [z, 0, 0]]
                    )
                    * self._simulate_scale
                )
                self.canvas.add_curve(arr, color="gray")
            for y, x in [(0, 0), (0, nx), (ny, nx), (ny, 0)]:
                arr = np.array([[0, y, x], [nz, y, x]]) * self._simulate_scale
                self.canvas.add_curve(arr, color="gray")
        else:
            self._points.data = mole.pos
            self._select_molecules(self.Operator.yrange, self.Operator.arange)

        self._molecules = mole
        return None

    def _set_shape_and_scale(self, shape: tuple[int, int, int], scale: nm):
        self._simulate_shape = shape
        self._simulate_scale = scale
        return None

    @magicclass(record=False)
    class Operator(MagicTemplate):
        """
        Select/adjust molecules in the model.

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

        yrange = vfield(tuple[int, int], label="axial", widget_type=RangeSlider)
        arange = vfield(
            tuple[int, int], label="angular", widget_type=RangeSlider
        ).with_options(value=(0, 100))

        @magicclass(
            properties={"margin": (0, 0, 0, 0)}, layout="horizontal", record=False
        )
        class Col(MagicTemplate):
            n_allev = abstractapi()
            show_selection = abstractapi()

        n_allev = Col.vfield(1, label="alleviate").with_options(min=0, max=20)
        show_selection = Col.vfield(True, label="show selected molecules")

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

        def _fill_shift(self, yrange, arange, val: float):
            parent = self.find_ancestor(CylinderSimulator, cache=True)
            shift = np.zeros(parent.model.shape, dtype=np.float32)
            ysl = slice(*yrange)
            asl = slice(*arange)
            shift[ysl, asl] = val
            return shift, Idx[ysl, asl]

    # the 3D viewer of the cylinder model
    canvas = field(Vispy3DCanvas)

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

    @ViewerMenu.wraps
    @thread_worker.with_progress(desc="Creating an image")
    @set_design(text="Create an empty image")
    @confirm(
        text="You may have unsaved data. Continue?",
        condition="self.parent_widget._need_save",
    )
    def create_empty_image(
        self,
        size: _ImageSize = (60.0, 200.0, 60.0),
        scale: Annotated[nm, {"label": "pixel scale (nm/pixel)"}] = 0.25,
    ):  # fmt: skip
        """
        Create an empty image with the given size and scale, and send it to the viewer.

        Parameters
        ----------
        size : tuple[nm, nm, nm], default is (100., 200., 100.)
            Size of the image in nm, of (Z, Y, X).
        scale : nm, default is 0.25
            Pixel size of the image.
        """
        parent = self.parent_widget
        shape = tuple(roundint(s / scale) for s in size)

        # update simulation parameters
        self._set_shape_and_scale(shape, scale)

        binsize = ceilint(0.96 / scale)
        # NOTE: zero-filled image breaks contrast limit calculation, and bad for
        # visual detection of the image edges.
        img = ip.zeros(shape, axes="zyx", name="simulated image")
        img.scale_unit = "nm"
        val = 100 * binsize**3
        px10nm = roundint(10 / scale)
        img[:, ::px10nm, ::px10nm] = val
        img[:, 0, :] = img[:, -1, :] = img[:, :, 0] = img[:, :, -1] = val / 2
        tomo = CylTomogram.from_image(img, scale=scale, binsize=binsize)
        parent._macro_offset = len(parent.macro)
        parent.tomogram = tomo
        return thread_worker.to_callback(parent._send_tomogram_to_viewer, False)

    def _get_current_index(self, *_) -> int:
        parent = self.parent_widget
        return parent.SplineControl.num

    def _get_shape(self, *_) -> tuple[int, int, int]:
        return self._simulate_shape

    def _get_scale(self, *_) -> float:
        return self._simulate_scale

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

    def _set_spline(self, spl: CylSpline) -> None:
        self._spline = spl
        self.canvas.layers.clear()
        self._points = None
        self._spline_arrow = None
        self._update_model_from_spline(spl, **self._parameters.asdict())
        return None

    @ViewerMenu.wraps
    @set_design(text="Set current spline")
    def set_current_spline(self, idx: Bound[_get_current_index]):
        """Use the current parameters and the spline to construct a model and molecules."""
        return self._set_spline(self.parent_widget.tomogram.splines[idx])

    @ViewerMenu.wraps
    @set_design(text="Load spline parameters")
    def load_spline_parameters(self, idx: Bound[_get_current_index]):
        """Copy the spline parameters in the viewer."""
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]
        props = spl.globalprops
        if props is None:
            raise ValueError("Global property is not calculated yet.")
        self._parameters.update(
            interval=props[H.yPitch][0],
            skew=props[H.skewAngle][0],
            rise=props[H.riseAngle][0],
            npf=props[H.nPF][0],
            radius=spl.radius,
        )
        return None

    @ViewerMenu.wraps
    @set_design(text="Create a straight line")
    def create_straight_line(
        self,
        length: nm = 150.0,
        size: _ImageSize = (60.0, 200.0, 60.0),
        scale: Annotated[nm, {"label": "pixel scale (nm/pixel)"}] = 0.25,
        yxrotation: Annotated[float, {"max": 90, "step": 1}] = 0.0,
        zxrotation: Annotated[float, {"max": 90, "step": 1}] = 0.0,
    ):
        """
        Create a straight line as a cylinder spline.

        Parameters
        ----------
        length : nm, default is 150.0
            Length if the straight line in nm.
        size : (nm, nm, nm), (60.0, 200.0, 60.0)
            Size of the tomogram in which the spline will reside.
        scale : nm, default is 0.25
            Scale of pixel in nm/pixel.
        yxrotation : float, optional
            Rotation in YX plane. This rotation will be applied before ZX rotation.
        zxrotation : float, optional
            Rotation in ZX plane. This rotation will be applied before YX rotation.
        """
        yxrot = Rotation.from_rotvec([np.deg2rad(yxrotation), 0.0, 0.0])
        zxrot = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(zxrotation)])
        start_shift = zxrot.apply(yxrot.apply(np.array([0.0, -length / 2, 0.0])))
        end_shift = zxrot.apply(yxrot.apply(np.array([0.0, length / 2, 0.0])))
        center = np.array(size) / 2
        shape = tuple(roundint(s / scale) for s in size)
        spl = CylSpline.line(start_shift + center, end_shift + center)
        self._set_shape_and_scale(shape, scale)
        self._set_spline(spl)
        self.show()
        return None

    @ViewerMenu.wraps
    @set_design(text="Send molecules to viewer")
    def send_moleclues_to_viewer(self):
        """Send the current molecules to the viewer."""
        mole = self._molecules
        if mole is None:
            raise ValueError("Molecules are not generated yet.")
        self.parent_widget.add_molecules(mole, name="Simulated", source=self._spline)
        return None

    @TransformMenu.wraps
    @impl_preview(auto_call=True)
    @set_design(text="Update model parameters")
    def update_model(
        self,
        interval: Annotated[nm, {"min": 0.2, "max": GVar.yPitchMax * 2, "step": 0.01, "label": "interval (nm)"}] = CylinderParameters.interval,
        skew: Annotated[float, {"min": GVar.minSkew, "max": GVar.maxSkew, "label": "skew (deg)"}] = CylinderParameters.skew,
        rise: Annotated[float, {"min": -90.0, "max": 90.0, "step": 0.5, "label": "rise (deg)"}] = CylinderParameters.rise,
        npf: Annotated[int, {"min": GVar.nPFmin, "max": GVar.nPFmax, "label": "nPF"}] = CylinderParameters.npf,
        radius: Annotated[nm, {"min": 0.5, "max": 50.0, "step": 0.5, "label": "radius (nm)"}] = CylinderParameters.radius,
        offsets: Annotated[tuple[float, float], {"options": {"min": -30.0, "max": 30.0}, "label": "offsets (nm, rad)"}] = CylinderParameters.offsets,
    ):  # fmt: skip
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
        spl = self._spline
        self._update_model_from_spline(spl, interval, skew, rise, npf, radius, offsets)
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

    def _update_model_from_spline(
        self,
        spl: CylSpline,
        interval,
        skew,
        rise,
        npf,
        radius,
        offsets,
    ):
        self._parameters.update(
            interval=interval,
            skew=skew,
            rise=rise,
            npf=npf,
            radius=radius,
            offsets=offsets,
        )
        kwargs = {H.yPitch: interval, H.skewAngle: skew, H.riseAngle: rise, H.nPF: npf}
        model = spl.cylinder_model(offsets=offsets, radius=radius, **kwargs)
        self.model = model

        op = self.Operator
        op._update_slider_lims(*self.model.shape)
        self._select_molecules(op.yrange, op.arange)  # update selection coordinates
        return None

    def _prep_radon(
        self,
        path: Path,
        degrees: np.ndarray,
        scale: float,
        shape: tuple[int, int, int],
        order: int = 3,
    ) -> tuple[ip.ImgArray, Molecules]:
        template = pipe.from_file(path)

        # noise-free tomogram generation from the current cylinder model
        model = self.model
        mole = model.to_molecules(self._spline)
        simulator = TomogramSimulator(order=order, scale=scale)
        simulator.add_molecules(molecules=mole, image=template)
        tilt_series = simulator.simulate_tilt_series(degrees=degrees, shape=shape)
        tilt_series = ip.asarray(tilt_series, axes=["degree", "y", "x"]).set_scale(
            y=scale, x=scale
        )
        return tilt_series, mole

    @SimulateMenu.wraps
    @dask_thread_worker.with_progress(descs=_simulate_tomogram_iter)
    @set_design(text="Simulate tomogram")
    def simulate_tomogram(
        self,
        template_path: Path.Read[FileFilter.IMAGE],
        save_dir: Annotated[Path.Dir, {"label": "Save at"}],
        nsr: _NSRatio = [2.0],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: int = 61,
        shape: Bound[_get_shape] = None,
        scale: Bound[_get_scale] = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):  # fmt: skip
        """
        Simulate tomographic images using the current model and save the images.

        This function projects the template image to each tilt series, adding
        Gaussian noise, and back-projects the noisy tilt series to the tomogram.

        Parameters
        ----------
        template_path : Path
            Path to the image used for the template.
        save_dir : Path
            Path to the directory where the images will be saved.
        nsr : list of float
            Noise-to-signal ratio.
        tilt_range : tuple of float
            Minimum and maximum tilt angles.
        n_tilt : int
            Number of tilt angles.
        interpolation : int
            Interpolation method used during the simulation.
        seed : int, optional
            Random seed used for the Gaussian noise.
        """
        if scale is None:
            scale = self._get_scale()
        if shape is None:
            shape = self._get_shape()
        save_dir = Path(save_dir)
        parent = self.parent_widget
        degrees = np.linspace(*tilt_range, n_tilt)
        sino, mole = self._prep_radon(
            template_path, degrees, scale, shape, interpolation
        )

        yield _on_radon_finished(sino, degrees)

        # add noise and save image
        if not save_dir.exists():
            save_dir.mkdir()
            _Logger.print(f"Directory created at {save_dir}.")

        nsr_info = {i: val for i, val in enumerate(nsr)}
        js = {
            "tilt_degree_min": tilt_range[0],
            "tilt_degree_max": tilt_range[1],
            "n_tilt": len(degrees),
            "interpolation": interpolation,
            "random_seed": seed,
            "tomogram_shape": shape,
            "central_axis": "y",  # NOTE: may change in the future
            "ns_ratio": nsr_info,
        }
        with open(save_dir / "simulation_info.json", "w") as f:
            json.dump(js, f, indent=4, separators=(", ", ": "))
        self._spline.to_json(save_dir / "spline.json")
        mole.to_csv(save_dir / "molecules.csv")
        macro_str = str(parent._format_macro(parent.macro[parent._macro_offset :]))
        fp = save_dir / "script.py"
        fp.write_text(macro_str)

        rng = ip.random.default_rng(seed)
        for i, nsr_val in enumerate(nsr):
            imax = sino.max()
            sino_noise = sino + rng.normal(
                scale=imax * nsr_val, size=sino.shape, axes=sino.axes
            )
            rec = sino_noise.iradon(
                degrees,
                central_axis="y",
                height=shape[0],
                order=interpolation,
            ).set_scale(zyx=scale, unit="nm")
            yield _on_iradon_finished(rec, f"N/S = {nsr_val:.1f}")

            file_name = save_dir / f"image-{i}.mrc"
            rec.imsave(file_name)
            _Logger.print(f"Image saved at {file_name}.")

        return None

    @SimulateMenu.wraps
    @dask_thread_worker.with_progress(desc="Simulating tilt series...")
    @set_design(text="Simulation tilt series")
    def simulate_tilt_series(
        self,
        template_path: Path.Read[FileFilter.IMAGE],
        save_dir: Annotated[Path.Dir, {"label": "Save at"}],
        nsr: _NSRatio = [2.0],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: int = 61,
        shape: Bound[_get_shape] = None,
        scale: Bound[_get_scale] = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):  # fmt: skip
        """
        Simulate tilt series using the current model and save the images.

        Parameters
        ----------
        template_path : Path
            Path to the image used for the template.
        save_dir : Path
            Path to the directory where the images will be saved. If None, the images
            will be added to the viewer.
        nsr : list of float
            Noise-to-signal ratio.
        tilt_range : tuple of float
            Minimum and maximum tilt angles.
        n_tilt : int
            Number of tilt angles.
        interpolation : int
            Interpolation method used during the simulation.
        seed : int, optional
            Random seed used for the Gaussian noise.
        """
        if scale is None:
            scale = self._get_scale()
        if shape is None:
            shape = self._get_shape()
        parent = self.parent_widget
        degrees = np.linspace(*tilt_range, n_tilt)
        sino, mole = self._prep_radon(
            template_path, degrees, scale, shape, interpolation
        )
        # add noise and save image
        if not save_dir.exists():
            save_dir.mkdir()
            _Logger.print(f"Directory created at {save_dir}.")

        nsr_info = {i: val for i, val in enumerate(nsr)}
        js = {
            "tilt_degree_min": tilt_range[0],
            "tilt_degree_max": tilt_range[1],
            "n_tilt": len(degrees),
            "interpolation": interpolation,
            "random_seed": seed,
            "tomogram_shape": shape,
            "central_axis": "y",  # NOTE: may change in the future
            "ns_ratio": nsr_info,
        }
        with open(save_dir / "simulation_info.json", "w") as f:
            json.dump(js, f, indent=4, separators=(", ", ": "))
        self._spline.to_json(save_dir / "spline.json")
        mole.to_csv(save_dir / "molecules.csv")
        macro_str = str(parent._format_macro(parent.macro[parent._macro_offset :]))
        fp = save_dir / "script.py"
        fp.write_text(macro_str)

        rng = ip.random.default_rng(seed)
        for i, nsr_val in enumerate(nsr):
            imax = sino.max()
            sino_noise = sino + rng.normal(
                scale=imax * nsr_val, size=sino.shape, axes=sino.axes
            )
            file_name = save_dir / f"tilt_series-{i}.mrc"
            sino_noise.set_axes("zyx").set_scale(zyx=scale, unit="nm").imsave(file_name)
            _Logger.print(f"Tilt series saved at {file_name}.")

        return None

    @TransformMenu.wraps
    @set_design(text="Expansion/Compaction")
    @impl_preview(auto_call=True)
    def expand(
        self,
        shift: Annotated[nm, {"min": -1.0, "max": 1.0, "step": 0.01, "label": "shift (nm)"}],
        yrange: Bound[Operator.yrange],
        arange: Bound[Operator.arange],
        n_allev: Bound[Operator.n_allev] = 1,
    ):  # fmt: skip
        """Expand the selected molecules."""
        shift_arr, sl = self.Operator._fill_shift(yrange, arange, shift)
        new_model = self.model.expand(shift, sl)
        if n_allev > 0:
            new_model = new_model.alleviate(shift_arr != 0, niter=n_allev)
        self.model = new_model
        return None

    @TransformMenu.wraps
    @set_design(text="Screw")
    @impl_preview(auto_call=True)
    def screw(
        self,
        skew: Annotated[float, {"min": -45.0, "max": 45.0, "step": 0.05, "label": "skew (deg)"}],
        yrange: Bound[Operator.yrange],
        arange: Bound[Operator.arange],
        n_allev: Bound[Operator.n_allev] = 1,
    ):  # fmt: skip
        """Screw (change the skew angles of) the selected molecules."""
        shift, sl = self.Operator._fill_shift(yrange, arange, skew)
        new_model = self.model.screw(np.deg2rad(skew), sl)
        if n_allev > 0:
            new_model = new_model.alleviate(shift != 0, niter=n_allev)
        self.model = new_model
        return None

    @TransformMenu.wraps
    @set_design(text="Dilation/Erosion")
    @impl_preview(auto_call=True)
    def dilate(
        self,
        radius: Annotated[nm, {"min": -1.0, "max": 1.0, "step": 0.1, "label": "radius (nm)"}],
        yrange: Bound[Operator.yrange],
        arange: Bound[Operator.arange],
        n_allev: Bound[Operator.n_allev] = 1,
    ):  # fmt: skip
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

    @create_empty_image.started.connect
    def _show_simulator(self):
        self.show()


@thread_worker.to_callback
def _on_radon_finished(sino: ip.ImgArray, degrees: np.ndarray):
    n_tilt = len(degrees)
    if n_tilt < 3:
        return
    _, ny, nx = sino.shape
    ysize = max(4 / nx * ny, 4)
    with _Logger.set_plt(rc_context={"font.size": 15}):
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, ysize))
        for i, idx in enumerate([0, n_tilt // 2, -1]):
            axes[i].imshow(sino[idx], cmap="gray")
            axes[i].set_title(f"deg = {degrees[idx]:.1f}")
            axes[i].set_axis_off()

        plt.show()
    return None


@thread_worker.to_callback
def _on_iradon_finished(rec: ip.ImgArray, title: str):
    with _Logger.set_plt(rc_context={"font.size": 15}):
        plt.imshow(rec.proj("z"), cmap="gray")
        plt.title(title)
        plt.show()
    return None
