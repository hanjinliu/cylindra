from typing import Any, TYPE_CHECKING, Annotated
import json
import matplotlib.pyplot as plt

import macrokit as mk
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
from magicgui.widgets import RangeSlider
from magicclass.types import Optional, Path
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
SELF = mk.Mock("self")
SIMULATION_INFO_FILE_NAME = "simulation_info.txt"

_TiltRange = Annotated[
    tuple[float, float],
    {
        "label": "Tilt range (deg)",
        "widget_type": "FloatRangeSlider",
        "min": -90.0,
        "max": 90.0,
    },
]

_NSRatio = Annotated[float, {"label": "N/S ratio", "min": 0.0, "max": 4.0, "step": 0.1}]

_NSRatios = Annotated[
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

    spacing: nm = 1.0
    skew: float = 0.0
    rise: float = 0.0
    npf: int = 1
    radius: nm = 10.0
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
            "spacing": self.spacing,
            "skew": self.skew,
            "rise": self.rise,
            "npf": self.npf,
            "radius": self.radius,
            "offsets": self.offsets,
        }


# Main widget class
@magicclass(widget_type="scrollable", labels=False)
class CylinderSimulator(MagicTemplate):
    @magicmenu(name="Create")
    class CreateMenu(MagicTemplate):
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
        simulate_tomogram_and_open = abstractapi()
        simulate_tilt_series = abstractapi()

    @magicmenu(name="Transform")
    class TransformMenu(MagicTemplate):
        """Transform the cylinder lattice."""

        update_model = abstractapi()
        expand = abstractapi()
        screw = abstractapi()
        dilate = abstractapi()

    def __post_init__(self) -> None:
        self._model: "CylinderModel | None" = None
        self._parameters = CylinderParameters()
        self._spline: "CylSpline | None" = None
        self._spline_arrow: "layers.Arrows3D | None" = None
        self._points: "layers.Points3D | None" = None
        self._selections: "layers.Points3D | None" = None
        self._layer_control = None
        self._simulate_shape = (0, 0, 0)
        self._simulate_scale = 0.0
        self._molecules: "Molecules | None" = None
        self.canvas.min_height = 300

    @property
    def parameters(self) -> CylinderParameters:
        """Parameters for cylinder model."""
        return self._parameters

    def _set_model(self, model: CylinderModel, spl: CylSpline):
        self._model = model
        self._spline = spl
        mole = model.to_molecules(spl)

        if self._points is None:
            self.canvas.layers.clear()
            self._points = self.canvas.add_points(
                mole.pos, size=GVar.point_size, face_color="lime", edge_color="lime"
            )
            self._selections = self.canvas.add_points(
                [[0, 0, 0]],
                size=GVar.point_size,
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
        allev : int
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
            allev = abstractapi()
            show_selection = abstractapi()

        allev = Col.vfield(True, label="alleviate")
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
        from cylindra.widgets.main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)

    @property
    def model(self) -> CylinderModel:
        """Current cylinder model."""
        return self._model

    @model.setter
    def model(self, model: CylinderModel):
        """Set new model and simulate molecules with the same spline."""
        return self._set_model(model, self._spline)

    @CreateMenu.wraps
    @thread_worker.with_progress(desc="Creating an image")
    @set_design(text="Create an empty image")
    @confirm(
        text="You may have unsaved data. Continue?",
        condition=SELF.parent_widget._need_save,
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
        return parent._send_tomogram_to_viewer.with_args(tomo)

    def _get_current_index(self, *_) -> int:
        parent = self.parent_widget
        return parent.SplineControl.num

    def _get_shape(self, *_) -> tuple[int, int, int]:
        return self._simulate_shape

    def _get_scale(self, *_) -> float:
        return self._simulate_scale

    def _select_molecules(self, yrange: tuple[int, int], arange: tuple[int, int]):
        points = self._points.data
        npf = self.parameters.npf
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
        with self.macro.blocked():
            self.update_model(**self.parameters.asdict())
        return None

    @CreateMenu.wraps
    @set_design(text="Set current spline")
    def set_current_spline(self, idx: Annotated[int, {"bind": _get_current_index}]):
        """Use the current parameters and the spline to construct a model and molecules."""
        return self._set_spline(self.parent_widget.tomogram.splines[idx])

    @CreateMenu.wraps
    @set_design(text="Load spline parameters")
    def load_spline_parameters(self, idx: Annotated[int, {"bind": _get_current_index}]):
        """Copy the spline parameters in the viewer."""
        tomo = self.parent_widget.tomogram
        spl = tomo.splines[idx]

        if not spl.props.has_glob([H.spacing, H.skew, H.rise, H.npf, H.radius]):
            raise ValueError("Global property is not calculated yet.")

        self.parameters.update(
            interval=spl.props.get_glob(H.spacing),
            skew=spl.props.get_glob(H.skew),
            rise=spl.props.get_glob(H.rise),
            npf=spl.props.get_glob(H.npf),
            radius=spl.props.get_glob(H.radius),
        )
        return None

    @CreateMenu.wraps
    @set_design(text="Create a straight line")
    def create_straight_line(
        self,
        length: nm = 150.0,
        size: _ImageSize = (60.0, 200.0, 60.0),
        scale: Annotated[nm, {"label": "pixel scale (nm/pixel)"}] = 0.25,
        yxrotation: Annotated[
            float, {"max": 90, "step": 1, "label": "Rotation in YX plane (deg)"}
        ] = 0.0,
        zxrotation: Annotated[
            float, {"max": 90, "step": 1, "label": "Rotation in ZX plane (deg)"}
        ] = 0.0,
    ):  # fmt: off
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

    @CreateMenu.wraps
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
        spacing: Annotated[nm, {"min": 0.2, "max": 100.0, "step": 0.01, "label": "spacing (nm)"}] = 1.0,
        skew: Annotated[float, {"min": -45.0, "max": 45.0, "label": "skew (deg)"}] = 0.0,
        rise: Annotated[float, {"min": -90.0, "max": 90.0, "step": 0.5, "label": "rise (deg)"}] = 0.0,
        npf: Annotated[int, {"min": 1, "label": "number of PF"}] = 1,
        radius: Annotated[nm, {"min": 0.5, "max": 50.0, "step": 0.5, "label": "radius (nm)"}] = 10.0,
        offsets: Annotated[tuple[float, float], {"options": {"min": -30.0, "max": 30.0}, "label": "offsets (nm, rad)"}] = CylinderParameters.offsets,
    ):  # fmt: skip
        """
        Update cylinder model with new parameters.

        Local structural displacement will be deleted because this function may change the number
        of molecules. This function should be called first.

        Parameters
        ----------
        spacing : nm
            Axial spacing between molecules.
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
        self.parameters.update(
            spacing=spacing,
            skew=skew,
            rise=rise,
            npf=npf,
            radius=radius,
            offsets=offsets,
        )
        # NOTE: these parameters are hard-coded for microtubule for now.
        kwargs = {
            H.spacing: spacing,
            H.skew: skew,
            H.rise: rise,
            H.npf: npf,
        }
        model = self._spline.cylinder_model(offsets=offsets, radius=radius, **kwargs)
        self.model = model

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
        nsr: _NSRatios = [1.5],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: Annotated[int, {"label": "Number of tilts"}] = 21,
        shape: Annotated[Any, {"bind": _get_shape}] = None,
        scale: Annotated[nm, {"bind": _get_scale}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
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
            Noise-to-signal ratio. It is defined by N/S, where S is the maximum
            value of the true monomer density and N is the standard deviation of
            the Gaussian noise. Duplicate values are allowed, which is useful
            for simulation of multiple images with the same noise level.
        tilt_range : tuple of float
            Minimum and maximum tilt angles in degree.
        n_tilt : int
            Number of tilt angles between minimum and maximum angles.
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
        nsr = list(round(float(_nsr), 4) for _nsr in nsr)
        parent = self.parent_widget
        degrees = np.linspace(*tilt_range, n_tilt)
        sino, mole = self._prep_radon(
            template_path, degrees, scale, shape, interpolation
        )

        yield _on_radon_finished.with_args(sino, degrees)

        # add noise and save image
        if not save_dir.exists():
            save_dir.mkdir()
            _Logger.print(f"Directory created at {save_dir}.")

        js = {
            "tilt_degree_min": tilt_range[0],
            "tilt_degree_max": tilt_range[1],
            "n_tilt": len(degrees),
            "interpolation": interpolation,
            "random_seed": seed,
            "tomogram_shape": shape,
            "central_axis": "y",  # NOTE: may change in the future
            "ns_ratio": list(nsr),
        }
        with open(save_dir / SIMULATION_INFO_FILE_NAME, "w") as f:
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
            yield _on_iradon_finished.with_args(rec.proj("z"), f"N/S = {nsr_val:.1f}")

            file_name = save_dir / f"image-{i}.mrc"
            rec.imsave(file_name)
            _Logger.print(f"Image saved at {file_name}.")

        return None

    @SimulateMenu.wraps
    @dask_thread_worker.with_progress(desc="Simulating tilt series...")
    @set_design(text="Simulate tomogram and open")
    @confirm(
        text="You may have unsaved data. Run anyway?",
        condition=SELF.parent_widget._need_save,
    )
    def simulate_tomogram_and_open(
        self,
        template_path: Path.Read[FileFilter.IMAGE],
        nsr: _NSRatio = 1.5,
        bin_size: Annotated[list[int], {"options": {"min": 1, "max": 32}}] = [1],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: Annotated[int, {"label": "Number of tilts"}] = 21,
        shape: Annotated[Any, {"bind": _get_shape}] = None,
        scale: Annotated[nm, {"bind": _get_scale}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):  # fmt: skip
        """
        Simulate a tomogram and open the image immediately.

        This function projects the template image to each tilt series, adding
        Gaussian noise, and back-projects the noisy tilt series to the tomogram.

        Parameters
        ----------
        template_path : Path
            Path to the image used for the template.
        nsr : list of float
            Noise-to-signal ratio. It is defined by N/S, where S is the maximum
            value of the true monomer density and N is the standard deviation of
            the Gaussian noise.
        bin_size : list of int
            Bin sizes used to create multi-scaled images from the simulated image.
        tilt_range : tuple of float
            Minimum and maximum tilt angles in degree.
        n_tilt : int
            Number of tilt angles between minimum and maximum angles.
        interpolation : int
            Interpolation method used during the simulation.
        seed : int, optional
            Random seed used for the Gaussian noise.
        """
        if scale is None:
            scale = self._get_scale()
        if shape is None:
            shape = self._get_shape()

        nsr = round(float(nsr), 4)
        parent = self.parent_widget
        degrees = np.linspace(*tilt_range, n_tilt)
        sino, _ = self._prep_radon(template_path, degrees, scale, shape, interpolation)

        yield _on_radon_finished.with_args(sino, degrees)

        rng = ip.random.default_rng(seed)
        imax = sino.max()
        sino_noise = sino + rng.normal(
            scale=imax * nsr, size=sino.shape, axes=sino.axes
        )
        rec = sino_noise.iradon(
            degrees,
            central_axis="y",
            height=shape[0],
            order=interpolation,
        ).set_scale(zyx=scale, unit="nm")
        yield _on_iradon_finished.with_args(rec.proj("z"), f"N/S = {nsr:.1f}")

        rec.name = "Simulated tomogram"
        tomo = CylTomogram.from_image(
            rec, scale=scale, tilt_range=tilt_range, binsize=bin_size
        )
        return parent._send_tomogram_to_viewer.with_args(tomo)

    @SimulateMenu.wraps
    @dask_thread_worker.with_progress(desc="Simulating tilt series...")
    @set_design(text="Simulate tilt series")
    def simulate_tilt_series(
        self,
        template_path: Path.Read[FileFilter.IMAGE],
        save_dir: Annotated[Path.Dir, {"label": "Save at"}],
        nsr: _NSRatios = [1.5],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: Annotated[int, {"label": "Number of tilts"}] = 21,
        shape: Annotated[Any, {"bind": _get_shape, "widget_type": "EmptyWidget"}] = None,
        scale: Annotated[nm, {"bind": _get_scale}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
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
            Noise-to-signal ratio. It is defined by N/S, where S is the maximum
            value of the true monomer density and N is the standard deviation of
            the Gaussian noise.
        tilt_range : tuple of float
            Minimum and maximum tilt angles in degree.
        n_tilt : int
            Number of tilt angles between minimum and maximum angles.
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
        with open(save_dir / SIMULATION_INFO_FILE_NAME, "w") as f:
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

    # TODO: should be like
    #   yrange: tuple[nm, nm]
    #   arange: tuple[float, float]
    # but magicgui TupleEdit doesn't support `bind`. Use Any for now.
    @TransformMenu.wraps
    @set_design(text="Expansion/Compaction")
    @impl_preview(auto_call=True)
    def expand(
        self,
        exp: Annotated[nm, {"min": -1.0, "max": 1.0, "step": 0.01, "label": "expansion (nm)"}],
        yrange: Annotated[Any, {"bind": Operator.yrange, "widget_type": "EmptyWidget"}],
        arange: Annotated[Any, {"bind": Operator.arange, "widget_type": "EmptyWidget"}],
        allev: Annotated[bool, {"bind": Operator.allev}] = True,
    ):  # fmt: skip
        """Expand the selected molecules."""
        shift, sl = self.Operator._fill_shift(yrange, arange, exp)
        new_model = self.model.expand(exp, sl)
        if allev > 0:
            new_model = new_model.alleviate(shift != 0)
        self.model = new_model
        return None

    @TransformMenu.wraps
    @set_design(text="Screw")
    @impl_preview(auto_call=True)
    def screw(
        self,
        skew: Annotated[float, {"min": -45.0, "max": 45.0, "step": 0.05, "label": "skew (deg)"}],
        yrange: Annotated[Any, {"bind": Operator.yrange, "widget_type": "EmptyWidget"}],
        arange: Annotated[Any, {"bind": Operator.arange, "widget_type": "EmptyWidget"}],
        allev: Annotated[bool, {"bind": Operator.allev}] = True,
    ):  # fmt: skip
        """Screw (change the skew angles of) the selected molecules."""
        shift, sl = self.Operator._fill_shift(yrange, arange, skew)
        new_model = self.model.screw(np.deg2rad(skew), sl)
        if allev > 0:
            new_model = new_model.alleviate(shift != 0)
        self.model = new_model
        return None

    @TransformMenu.wraps
    @set_design(text="Dilation/Erosion")
    @impl_preview(auto_call=True)
    def dilate(
        self,
        radius: Annotated[nm, {"min": -10.0, "max": 10.0, "step": 0.1, "label": "radius (nm)"}],
        yrange: Annotated[Any, {"bind": Operator.yrange, "widget_type": "EmptyWidget"}],
        arange: Annotated[Any, {"bind": Operator.arange, "widget_type": "EmptyWidget"}],
        allev: Annotated[bool, {"bind": Operator.allev}] = True,
    ):  # fmt: skip
        """Dilate (increase the local radius of) the selected molecules."""
        shift, sl = self.Operator._fill_shift(yrange, arange, radius)
        new_model = self.model.dilate(radius, sl)
        if allev > 0:
            new_model = new_model.alleviate(shift != 0)
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


@thread_worker.callback
def _on_radon_finished(sino: ip.ImgArray, degrees: np.ndarray):
    n_tilt = len(degrees)
    if n_tilt < 3:
        return
    _, ny, nx = sino.shape
    ysize = max(4 / nx * ny, 4)
    with _Logger.set_plt():
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, ysize))
        for i, idx in enumerate([0, n_tilt // 2, -1]):
            axes[i].imshow(sino[idx], cmap="gray")
            axes[i].set_title(f"deg = {degrees[idx]:.1f}")
            axes[i].set_axis_off()
        plt.tight_layout()
        plt.show()
    return None


@thread_worker.callback
def _on_iradon_finished(rec: ip.ImgArray, title: str):
    with _Logger.set_plt():
        plt.imshow(rec, cmap="gray")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    return None
