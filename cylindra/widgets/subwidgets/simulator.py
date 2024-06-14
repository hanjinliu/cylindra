import weakref
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterator

import impy as ip
import numpy as np
import polars as pl
from acryo import TomogramSimulator, pipe
from magicclass import (
    abstractapi,
    confirm,
    do_not_record,
    field,
    impl_preview,
    magicclass,
    magicmenu,
    magictoolbar,
    set_design,
    setup_function_gui,
    vfield,
)
from magicclass.ext.dask import dask_thread_worker
from magicclass.logging import getLogger
from magicclass.types import ExprStr, Optional, Path
from magicclass.utils import thread_worker
from magicclass.widgets import Separator
from magicgui.widgets import FunctionGui, Label, RangeSlider
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from cylindra._napari import MoleculesLayer
from cylindra.components import (
    CylinderModel,
    CylindricSliceConstructor,
    CylSpline,
    CylTomogram,
    indexer,
)
from cylindra.const import (
    INTERPOLATION_CHOICES,
    PREVIEW_LAYER_NAME,
    FileFilter,
    nm,
)
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.utils import ceilint, roundint
from cylindra.widget_utils import POLARS_NAMESPACE, capitalize
from cylindra.widgets._annotated import MoleculesLayerType, _as_layer_name, assert_layer
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    import napari
    from napari.layers import Layer

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
_ImageSize = Annotated[
    tuple[nm, nm, nm],
    {"label": "image size of Z, Y, X (nm)", "options": {"max": 10000.0}},
]
_Point3D = Annotated[
    tuple[nm, nm, nm],
    {"label": "coordinates of Z, Y, X (nm)", "options": {"max": 10000.0}},
]
PROJECT_NAME = "simulation-project.tar"
SIMULATION_MODEL_KEY = "simulation-model"
SIMULATED_IMAGE_NAME = "Simulated tomogram"
_Logger = getLogger("cylindra")


def _simulate_tomogram_iter(nsr):
    n = len(nsr)
    yield f"(0/{n + 1}) Simulating projections"
    for i in range(n):
        yield f"({i + 1}/{n + 1}) Back-projection of {i}-th image"


def _simulate_tomogram_from_tilt_iter():
    yield "(0/2) Reading tilt series"
    yield "(1/2) Back-projection"


@magicclass(labels=False, record=False, layout="horizontal")
class Component(ChildWidget):
    layer_name = vfield(str).with_options(enabled=False)
    path = vfield(str).with_options(enabled=False)

    def __init__(self, path: str, layer: MoleculesLayer):
        self._layer_ref = weakref.ref(layer)
        self._path = str(path)

    def __post_init__(self):
        self.path = self._path
        self.layer_name = self._layer_ref().name

    @property
    def layer(self) -> MoleculesLayer | None:
        return self._layer_ref()

    @set_design(text="✕")
    def remove_me(self):
        """Remove this component"""
        parent = self.find_ancestor(ComponentList)
        idx = parent.index(self)
        del parent[idx]
        parent._on_children_change()


@magicclass(widget_type="scrollable", record=False)
class ComponentList(ChildWidget):
    """List of components"""

    def __post_init__(self):
        self._empty_label = Label(value="No components added.")
        self.append(self._empty_label)

    def _iter_components(self) -> Iterator[Component]:
        for wdt in self:
            if isinstance(wdt, Component):
                yield wdt

    def _as_input(self) -> list[tuple[str, Path]]:
        return [(comp.layer_name, Path(comp.path)) for comp in self._iter_components()]

    def reset_choices(self, *_):
        to_remove = []
        for i, comp in enumerate(self._iter_components()):
            if layer := comp.layer:
                comp.layer_name = layer.name
            else:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.pop(i)
        self._on_children_change()

    def _on_children_change(self):
        self._empty_label.visible = len(list(self._iter_components())) == 0


@magicclass(use_native_menubar=False)
class Simulator(ChildWidget):
    @magicmenu(name="Create")
    class CreateMenu(ChildWidget):
        create_empty_image = abstractapi()
        create_straight_line = abstractapi()
        create_image_with_straight_line = abstractapi()

    @magicmenu(name="Simulate")
    class SimulateMenu(ChildWidget):
        simulate_tomogram = abstractapi()
        simulate_tomogram_from_tilt_series = abstractapi()
        simulate_tomogram_and_open = abstractapi()
        simulate_tilt_series = abstractapi()
        simulate_projection = abstractapi()

    @magictoolbar
    class SimulatorTools(ChildWidget):
        add_component = abstractapi()
        sep0 = field(Separator)
        generate_molecules = abstractapi()
        expand = abstractapi()
        twist = abstractapi()
        dilate = abstractapi()
        displace = abstractapi()

    component_list = field(ComponentList, name="components")

    def _prep_radon(
        self,
        components: list[tuple[str, Path]],
        degrees: NDArray[np.floating],
        order: int = 3,
    ) -> ip.ImgArray:
        # noise-free tomogram generation from the current cylinder model
        main = self._get_main()
        tomo = main.tomogram
        scale = tomo.scale
        shape = tomo.image.shape
        simulator = TomogramSimulator(order=order, scale=scale)
        for layer_name, temp_path in components:
            mole = main.mole_layers[layer_name].molecules
            simulator.add_molecules(mole, pipe.from_file(temp_path))
        tilt_series = simulator.simulate_tilt_series(degrees=degrees, shape=shape)
        tilt_series = ip.asarray(
            tilt_series, axes=["degree", "y", "x"], name="Simulated"
        )
        return tilt_series.set_scale(y=scale, x=scale)

    def _get_proper_molecules_layers(self, *_):
        out = list[MoleculesLayer]()
        for layer in self._get_main().mole_layers:
            if layer.source_spline is None:
                continue
            mole = layer.molecules
            cols = mole.features.columns
            if Mole.nth in cols and Mole.pf in cols and mole.count() > 0:
                out.append(layer)
        return out

    _ModeledMoleculesLayer = Annotated[
        MoleculesLayer,
        {"choices": _get_proper_molecules_layers, "validator": _as_layer_name},
    ]

    @set_design(icon="fluent:cloud-add-16-filled", location=SimulatorTools)
    @do_not_record
    def add_component(
        self,
        layer: MoleculesLayerType,
        template_path: Path.Read[FileFilter.IMAGE],
    ):
        """
        Add a set of template and a molecules as a simulation component.

        A component defines which molecules corresponds to what template image.
        Multiple components can be added to simulate a tomogram with different
        materials.

        Parameters
        ----------
        layer : MoleculesLayer
            Layer to be used for simulation.
        template_path : Path
            Path to the template image that will be used to simulate the
            corresponding molecules layer.
        """
        layer = assert_layer(layer, self.parent_viewer)
        self.component_list.append(Component(template_path, layer))
        self.component_list._on_children_change()

    @set_design(text=capitalize, location=CreateMenu)
    @thread_worker.with_progress(desc="Creating an image")
    @confirm(
        text="You have an opened image. Run anyway?",
        condition="not self._get_main().tomogram.is_dummy",
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
        size : (nm, nm, nm), default (60., 200., 60.)
            Size of the image in nm, of (Z, Y, X).
        scale : nm, default 0.25
            Pixel size of the image.
        """
        main = self._get_main()
        shape = tuple(roundint(s / scale) for s in size)

        binsize = ceilint(0.96 / scale)
        # NOTE: zero-filled image breaks contrast limit calculation, and bad for
        # visual detection of the image edges.
        tomo = CylTomogram.dummy(scale=scale, binsize=binsize, shape=shape)
        main._macro_offset = len(main.macro)
        yield main._send_tomogram_to_viewer.with_args(tomo)
        main._reserved_layers.image.bounding_box.visible = True

    @set_design(text=capitalize, location=CreateMenu)
    def create_straight_line(
        self,
        start: _Point3D,
        end: _Point3D,
    ):
        """
        Create a straight line as a spline.

        Parameters
        ----------
        start : (nm, nm, nm)
            Start point of the line.
        end : (nm, nm, nm)
            End point of the line.
        """
        spl = CylSpline.line(start, end)
        main = self._get_main()
        main.tomogram.splines.append(spl)
        main._add_spline_instance(spl)
        return None

    @set_design(text=capitalize, location=CreateMenu)
    @thread_worker.with_progress(desc="Creating an image")
    @confirm(
        text="You have an opened image. Run anyway?",
        condition="not self._get_main().tomogram.is_dummy",
    )
    def create_image_with_straight_line(
        self,
        length: nm = 150.0,
        size: _ImageSize = (60.0, 200.0, 60.0),
        scale: Annotated[nm, {"label": "pixel scale (nm/pixel)"}] = 0.25,
        yxrotation: Annotated[float, {"max": 90, "step": 1, "label": "Rotation in YX plane (deg)"}] = 0.0,
        zxrotation: Annotated[float, {"max": 90, "step": 1, "label": "Rotation in ZX plane (deg)"}] = 0.0,
    ):  # fmt: skip
        """
        Create a straight line as a cylinder spline.

        Parameters
        ----------
        length : nm, default 150.0
            Length if the straight line in nm.
        size : (nm, nm, nm), (60.0, 200.0, 60.0)
            Size of the tomogram in which the spline will reside.
        scale : nm, default 0.25
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
        yield from self.create_empty_image.arun(size=size, scale=scale)
        cb = thread_worker.callback(self.create_straight_line)
        yield cb.with_args(start_shift + center, end_shift + center)

    def _get_spline_idx(self, *_) -> int:
        return self._get_main()._get_spline_idx()

    @set_design(icon="fluent:select-object-skew-20-regular", location=SimulatorTools)
    def generate_molecules(
        self,
        spline: Annotated[int, {"bind": _get_spline_idx}] = 0,
        spacing: Annotated[nm, {"min": 0.2, "max": 100.0, "step": 0.01, "label": "spacing (nm)"}] = 1.0,
        twist: Annotated[float, {"min": -45.0, "max": 45.0, "label": "twist (deg)"}] = 0.0,
        start: Annotated[int, {"min": -50, "max": 50, "label": "start"}] = 0,
        npf: Annotated[int, {"min": 1, "label": "number of PF"}] = 2,
        radius: Annotated[nm, {"min": 0.5, "max": 50.0, "step": 0.5, "label": "radius (nm)"}] = 10.0,
        offsets: tuple[float, float] = (0.0, 0.0),
        update_glob: Annotated[bool, {"label": "update spline global properties"}] = True,
    ):  # fmt: skip
        """
        Update cylinder model with new parameters.

        Local structural displacement will be deleted because this function may change
        the number of molecules. This function should be called first.

        Parameters
        ----------
        spacing : nm
            Axial spacing between molecules.
        twist : float
            Monomer twist of the cylinder.
        start : int
            The start number.
        npf : int
            Number of protofilaments.
        radius : nm
            Radius of the cylinder.
        """
        # NOTE: these parameters are hard-coded for microtubule for now.
        main = self._get_main()
        spl = main.splines[spline]
        model = self._prep_model(spl, spacing, twist, start, npf, radius, offsets)
        mole = model.to_molecules(spl)
        name = _make_simulated_mole_name(main.parent_viewer)
        layer = main.add_molecules(mole, name=name, source=spl)
        _set_simulation_model(layer, model)
        if update_glob:
            cparams = spl.cylinder_params(
                spacing=spacing, twist=twist, start=start, npf=npf, radius=radius
            )
            spl.update_glob_by_cylinder_params(cparams)

    @impl_preview(generate_molecules, auto_call=True)
    def _preview_generate_molecules(
        self,
        spline: int,
        spacing: nm,
        twist: float,
        start: int,
        npf: int,
        radius: nm,
        offsets: tuple[float, float],
    ):
        main = self._get_main()
        spl = main.splines[spline]
        model = self._prep_model(spl, spacing, twist, start, npf, radius, offsets)
        out = model.to_molecules(spl)
        viewer = main.parent_viewer
        if PREVIEW_LAYER_NAME in viewer.layers:
            layer: Layer = viewer.layers[PREVIEW_LAYER_NAME]
            layer.data = out.pos
        else:
            layer = main.add_molecules(
                out, name=PREVIEW_LAYER_NAME, face_color="crimson"
            )
        is_active = yield
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)

    def _prep_model(
        self,
        spl: CylSpline,
        spacing: nm,
        twist: float,
        start: int,
        npf: int,
        radius: nm,
        offsets: tuple[float, float],
    ) -> CylinderModel:
        kwargs = {
            H.spacing: spacing,
            H.twist: twist,
            H.start: start,
            H.npf: npf,
            H.radius: radius,
        }
        model = spl.cylinder_model(offsets=offsets, **kwargs)
        return model

    def _get_components(self, *_):
        return self.component_list._as_input()

    @set_design(text=capitalize, location=SimulateMenu)
    @dask_thread_worker.with_progress(descs=_simulate_tomogram_iter)
    def simulate_tomogram(
        self,
        components: Annotated[Any, {"bind": _get_components}],
        save_dir: Annotated[Path.Save, {"label": "Save at"}],
        nsr: _NSRatios = [1.5],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: Annotated[int, {"label": "Number of tilts"}] = 21,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):  # fmt: skip
        """
        Simulate tomographic images using the current model and save the images.

        This function projects the template image to each tilt series, adding
        Gaussian noise, and back-projects the noisy tilt series to the tomogram.

        Parameters
        ----------
        components : list of (str, Path)
            List of tuples of layer name and path to the template image.
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
        save_dir = _norm_save_dir(save_dir)
        _assert_not_empty(components)
        nsr = [round(float(_nsr), 4) for _nsr in nsr]
        main = self._get_main()
        degrees = np.linspace(*tilt_range, n_tilt)
        sino = self._prep_radon(components, degrees, order=interpolation)

        yield _on_radon_finished.with_args(sino, degrees)

        rng = ip.random.default_rng(seed)
        imax = sino.max()
        for i, nsr_val in enumerate(nsr):
            sino_noise = sino + rng.normal(
                scale=imax * nsr_val, size=sino.shape, axes=sino.axes
            )
            rec = sino_noise.iradon(
                degrees,
                central_axis="y",
                height=main.tomogram.image.shape[0],
                order=interpolation,
            ).set_scale(zyx=main.tomogram.scale, unit="nm")
            yield _on_iradon_finished.with_args(rec.mean("z"), f"N/S = {nsr_val:.1f}")

            file_name = save_dir / f"image-{i}.mrc"
            rec.imsave(file_name)
            _Logger.print(f"Image saved at {file_name}.")

        main.save_project(save_dir / PROJECT_NAME, molecules_ext=".parquet")
        return None

    @set_design(text=capitalize, location=SimulateMenu)
    @dask_thread_worker.with_progress(descs=_simulate_tomogram_from_tilt_iter)
    @confirm(
        text="You have an opened image. Run anyway?",
        condition="not self._get_main().tomogram.is_dummy",
    )
    def simulate_tomogram_from_tilt_series(
        self,
        path: Path.Read[FileFilter.IMAGE],
        nsr: _NSRatio = 1.5,
        bin_size: Annotated[list[int], {"options": {"min": 1, "max": 32}}] = [1],
        tilt_range: _TiltRange = (-60.0, 60.0),
        height: Annotated[nm, {"label": "height (nm)"}] = 50,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):
        """
        Simulate tomographic images using a tilt series.

        Parameters
        ----------
        path : Path
            Path to the tilt series image.
        nsr : float
            Noise-to-signal ratio. It is defined by N/S, where S is the maximum
            value of the tilt series. If the input image is already noisy, you
            can set this value to zero to avoid adding more noises.
        bin_size : list of int
            Bin sizes used to create multi-scaled images from the simulated image.
        tilt_range : tuple of float
            Minimum and maximum tilt angles in degree.
        height : int
            Height of the simulated tomogram in nm.
        interpolation : int
            Interpolation method used during the simulation.
        seed : int, optional
            Random seed used for the Gaussian noise.
        """
        main = self._get_main()
        sino = ip.imread(path)
        scale = sino.scale.x
        if sino.ndim != 3:
            raise ValueError("Input image must be a 3D image.")
        degrees = np.linspace(*tilt_range, sino.shape[0])
        rng = ip.random.default_rng(seed)
        imax = sino.max()
        sino_noise = sino + rng.normal(
            scale=imax * nsr, size=sino.shape, axes=sino.axes
        )
        yield thread_worker.callback()
        rec = sino_noise.iradon(
            degrees,
            central_axis="y",
            height=roundint(height / scale),
            order=interpolation,
        ).set_scale(zyx=scale, unit="nm")
        yield _on_iradon_finished.with_args(rec.mean("z"), f"N/S = {nsr:.1f}")
        rec.name = SIMULATED_IMAGE_NAME
        tomo = CylTomogram.from_image(
            rec, scale=scale, tilt=tilt_range, binsize=bin_size
        )
        main._macro_offset = len(main.macro)
        return main._send_tomogram_to_viewer.with_args(tomo)

    @set_design(text=capitalize, location=SimulateMenu)
    @dask_thread_worker.with_progress(desc="Simulating tomogram...")
    @confirm(
        text="You have an opened image. Run anyway?",
        condition="not self._get_main().tomogram.is_dummy",
    )
    def simulate_tomogram_and_open(
        self,
        components: Annotated[Any, {"bind": _get_components}],
        nsr: _NSRatio = 1.5,
        bin_size: Annotated[list[int], {"options": {"min": 1, "max": 32}}] = [1],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: Annotated[int, {"label": "Number of tilts"}] = 21,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):  # fmt: skip
        """
        Simulate a tomogram and open the image immediately.

        This function projects the template image to each tilt series, adding
        Gaussian noise, and back-projects the noisy tilt series to the tomogram.

        Parameters
        ----------
        components : list of (str, Path)
            List of tuples of layer name and path to the template image.
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
        nsr = round(float(nsr), 4)
        _assert_not_empty(components)
        main = self._get_main()
        degrees = np.linspace(*tilt_range, n_tilt)
        mole_layers = [main.mole_layers[layer_name] for layer_name, _ in components]
        sources = [layer.source_spline for layer in mole_layers]
        sino = self._prep_radon(components, degrees, order=interpolation)

        yield _on_radon_finished.with_args(sino, degrees)

        rng = ip.random.default_rng(seed)
        imax = sino.max()
        sino_noise = sino + rng.normal(
            scale=imax * nsr, size=sino.shape, axes=sino.axes
        )
        rec = sino_noise.iradon(
            degrees,
            central_axis="y",
            height=main.tomogram.image.shape[0],
            order=interpolation,
        ).set_scale(zyx=sino.scale.x, unit="nm")
        yield _on_iradon_finished.with_args(rec.mean("z"), f"N/S = {nsr:.1f}")

        rec.name = SIMULATED_IMAGE_NAME
        tomo = CylTomogram.from_image(
            rec, scale=sino.scale.x, tilt=tilt_range, binsize=bin_size
        )
        tomo.splines.extend(sources)
        yield main._send_tomogram_to_viewer.with_args(tomo)

        @thread_worker.callback
        def _on_return():
            for layer, source_spline in zip(mole_layers, sources, strict=True):
                main.parent_viewer.add_layer(layer)
                if source_spline is not None:
                    layer.source_component = source_spline
            if len(main.splines) > 0:
                main._update_splines_in_images()

        return _on_return

    @set_design(text=capitalize, location=SimulateMenu)
    @dask_thread_worker.with_progress(desc="Simulating tilt series...")
    def simulate_tilt_series(
        self,
        components: Annotated[Any, {"bind": _get_components}],
        save_dir: Annotated[Path.Save, {"label": "Save at"}],
        tilt_range: _TiltRange = (-60.0, 60.0),
        n_tilt: Annotated[int, {"label": "Number of tilts"}] = 21,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
    ):  # fmt: skip
        """
        Simulate tilt series using the current model and save the images.

        Parameters
        ----------
        components : list of (str, Path)
            List of tuples of layer name and path to the template image.
        save_dir : Path
            Directory path where the tilt series will be saved.
        tilt_range : tuple of float
            Minimum and maximum tilt angles in degree.
        n_tilt : int
            Number of tilt angles between minimum and maximum angles.
        interpolation : int
            Interpolation method used during the simulation.
        """
        save_dir = _norm_save_dir(save_dir)
        _assert_not_empty(components)
        degrees = np.linspace(*tilt_range, n_tilt)
        sino = self._prep_radon(components, degrees, order=interpolation)
        scale = sino.scale.x
        save_path = save_dir / "image.mrc"
        sino.set_axes("zyx").set_scale(zyx=scale, unit="nm").imsave(save_path)
        _Logger.print(f"Tilt series saved at {save_path}.")
        self._get_main().save_project(save_dir / PROJECT_NAME, molecules_ext=".parquet")
        return None

    @set_design(text=capitalize, location=SimulateMenu)
    @dask_thread_worker.with_progress(desc="Simulating 2D projections...")
    def simulate_projection(
        self,
        components: Annotated[Any, {"bind": _get_components}],
        save_dir: Annotated[Path.Save, {"label": "Save at"}],
        nsr: _NSRatios = [1.5],
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        seed: Optional[Annotated[int, {"min": 0, "max": 1e8}]] = None,
    ):  # fmt: skip
        """
        Simulate a projection without tilt (cryo-EM-like image).

        Parameters
        ----------
        components : list of (str, Path)
            List of tuples of layer name and path to the template image.
        save_dir : Path
            Path to the directory where the images will be saved.
        nsr : list of float
            Noise-to-signal ratio. It is defined by N/S, where S is the maximum
            value of the true monomer density and N is the standard deviation of
            the Gaussian noise. Duplicate values are allowed, which is useful
            for simulation of multiple images with the same noise level.
        interpolation : int
            Interpolation method used during the simulation.
        seed : int, optional
            Random seed used for the Gaussian noise.
        """
        save_dir = _norm_save_dir(save_dir)
        _assert_not_empty(components)
        proj = self._prep_radon(components, np.zeros(1), order=interpolation)[0]
        proj = proj.set_axes("yx").set_scale(yx=proj.scale.x, unit="nm")
        yield _on_iradon_finished.with_args(proj, "Projection (noise-free)")
        rng = ip.random.default_rng(seed)
        imax = proj.max()
        for i, nsr_val in enumerate(nsr):
            proj_noise = proj + rng.normal(
                scale=imax * nsr_val, size=proj.shape, axes=proj.axes
            )
            proj_noise.imsave(save_dir / f"image-{i}.tif")
        _Logger.print(f"Projections saved at {save_dir}.")
        self._get_main().save_project(save_dir / PROJECT_NAME, molecules_ext=".parquet")
        return None

    @set_design(icon="iconoir:expand-lines", location=SimulatorTools)
    def expand(
        self,
        layer: _ModeledMoleculesLayer,
        by: Annotated[float, {"min": -100, "max": 100}] = 0.0,
        yrange: Annotated[tuple[int, int], {"widget_type": RangeSlider}] = (0, 1),
        arange: Annotated[tuple[int, int], {"widget_type": RangeSlider}] = (0, 1),
        allev: bool = False,
    ):
        """
        Apply local expansion to molecules.

        Parameters
        ----------
        layer : MoleculesLayer
            Layer to be transformed.
        by : float
            Amount of expansion in nm.
        yrange : tuple of int
            Range of Y axis to be transformed. Range is [a, b).
        arange : tuple of int
            Range of angle axis to be transformed. Range is [a, b).
        allev : bool
            Alleviation of the local expansion. If true, the surrounding molecules
            will be shifted to alleviate the local expansion.
        """
        layer = assert_layer(layer, self.parent_viewer)
        spl, model = _local_transform(
            CylinderModel.expand, layer, by, yrange, arange, allev
        )
        layer.molecules = model.to_molecules(spl, layer.molecules.features)
        _set_simulation_model(layer, model)
        return None

    @set_design(icon="mingcute:rotate-x-line", location=SimulatorTools)
    def twist(
        self,
        layer: _ModeledMoleculesLayer,
        by: Annotated[float, {"min": -100, "max": 100}] = 0.0,
        yrange: Annotated[tuple[int, int], {"widget_type": RangeSlider}] = (0, 1),
        arange: Annotated[tuple[int, int], {"widget_type": RangeSlider}] = (0, 1),
        allev: bool = False,
    ):
        """
        Apply local twist to molecules.

        Parameters
        ----------
        layer : MoleculesLayer
            Layer to be transformed.
        by : float
            Amount of twist in degree.
        yrange : tuple of int
            Range of Y axis to be transformed. Range is [a, b).
        arange : tuple of int
            Range of angle axis to be transformed. Range is [a, b).
        allev : bool
            Alleviation of the local expansion. If true, the surrounding molecules
            will be shifted to alleviate the local expansion.
        """
        layer = assert_layer(layer, self.parent_viewer)
        spl, model = _local_transform(
            CylinderModel.twist, layer, np.deg2rad(by), yrange, arange, allev
        )
        layer.molecules = model.to_molecules(spl, layer.molecules.features)
        _set_simulation_model(layer, model)
        return None

    @set_design(icon="iconoir:scale-frame-enlarge", location=SimulatorTools)
    def dilate(
        self,
        layer: _ModeledMoleculesLayer,
        by: Annotated[float, {"min": -100, "max": 100}] = 0.0,
        yrange: Annotated[tuple[int, int], {"widget_type": RangeSlider}] = (0, 1),
        arange: Annotated[tuple[int, int], {"widget_type": RangeSlider}] = (0, 1),
        allev: bool = False,
    ):
        """
        Apply local dilation to molecules.

        Parameters
        ----------
        layer : MoleculesLayer
            Layer to be transformed.
        by : float
            Amount of dilation in nm.
        yrange : tuple of int
            Range of Y axis to be transformed. Range is [a, b).
        arange : tuple of int
            Range of angle axis to be transformed. Range is [a, b).
        allev : bool
            Alleviation of the local expansion. If true, the surrounding molecules
            will be shifted to alleviate the local expansion.
        """
        layer = assert_layer(layer, self.parent_viewer)
        spl, model = _local_transform(
            CylinderModel.dilate, layer, by, yrange, arange, allev
        )
        layer.molecules = model.to_molecules(spl, layer.molecules.features)
        _set_simulation_model(layer, model)
        return None

    @set_design(icon="fluent:arrow-move-20-filled", location=SimulatorTools)
    def displace(
        self,
        layer: _ModeledMoleculesLayer,
        expand: ExprStr.In[POLARS_NAMESPACE] = 0.0,
        twist: ExprStr.In[POLARS_NAMESPACE] = 0.0,
        dilate: ExprStr.In[POLARS_NAMESPACE] = 0.0,
    ):
        """
        Detailed local transformation of molecules.

        In this method, you'll have to specify the displacement for each molecule
        using polars expressions. For example, if you want to expand the molecules
        with odd numbering by 0.1 nm, you can set `expand` to
        >>> pl.when(pl.col("nth") % 2 == 0).then(0).otherwise(0.1)

        Parameters
        ----------
        layer : ModelLayer
            Layer to be transformed.
        expand : str, pl.Expr or constant
            Displacement in the longitudinal direction (nm).
        twist : str, pl.Expr or constant
            Displacement in the angular direction (degree).
        dilate : str, pl.Expr or constant
            Displacement from the center (nm).
        """
        layer = assert_layer(layer, self.parent_viewer)
        new_model = _get_shifted_model(layer, expand, twist, dilate)
        layer.molecules = new_model.to_molecules(
            layer.source_spline, layer.molecules.features
        )
        return _set_simulation_model(layer, new_model)


def _normalize_expression(expr: Any) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, str):
        out = ExprStr(expr, POLARS_NAMESPACE).eval()
        if not isinstance(out, pl.Expr):
            return pl.repeat(float(out), pl.len())
        return out
    else:
        return pl.repeat(float(expr), pl.len())


def _get_simulation_model(layer: MoleculesLayer) -> CylinderModel:
    out = layer.metadata.get(SIMULATION_MODEL_KEY)
    if out is None:
        raise ValueError("No simulation model is associated with this layer.")
    return out


def _set_simulation_model(layer: MoleculesLayer, model: CylinderModel):
    layer.metadata[SIMULATION_MODEL_KEY] = model


def _fill_shift(yrange, arange, val: float, shape):
    shift = np.zeros(shape, dtype=np.float32)
    ysl = slice(*yrange)
    asl = slice(*arange)
    shift[ysl, asl] = val
    return shift, indexer[ysl, asl]


def _assert_not_empty(components: list[tuple[str, Path]]):
    if not components:
        raise ValueError("No component is added.")


@thread_worker.callback
def _on_radon_finished(sino: ip.ImgArray, degrees: np.ndarray):
    import matplotlib.pyplot as plt

    n_tilt = len(degrees)
    if n_tilt < 3:
        return
    _, ny, nx = sino.shape
    ysize = max(4 / nx * ny, 4)
    with _Logger.set_plt():
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, ysize))
        axes: list[plt.Axes]
        for i, idx in enumerate([0, n_tilt // 2, -1]):
            axes[i].imshow(sino[idx], cmap="gray")
            axes[i].set_title(f"deg = {degrees[idx]:.1f}")
            axes[i].set_axis_off()
        plt.tight_layout()
        plt.show()
    return None


@thread_worker.callback
def _on_iradon_finished(rec: ip.ImgArray, title: str):
    import matplotlib.pyplot as plt

    with _Logger.set_plt():
        plt.imshow(rec, cmap="gray")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    return None


def _norm_save_dir(save_dir) -> Path:
    save_dir = Path(save_dir)
    if save_dir.suffix:
        raise ValueError(f"{save_dir.as_posix()} is not a directory.")

    # add noise and save image
    if not save_dir.exists():
        save_dir.mkdir()
        _Logger.print(f"Directory created at {save_dir}.")
    return save_dir


@setup_function_gui(Simulator.expand)
@setup_function_gui(Simulator.twist)
@setup_function_gui(Simulator.dilate)
def _fetch_shape(self: Simulator, gui: FunctionGui):
    @gui.layer.changed.connect
    def _on_layer_change(layer: MoleculesLayer | None):
        if layer is None:
            return
        df = layer.molecules.features
        try:
            nth = df[Mole.nth].n_unique()
            npf = df[Mole.pf].n_unique()
            if len(df) != nth * npf:
                raise ValueError
        except (KeyError, ValueError):
            gui.yrange.enabled = gui.arange.enabled = False
        else:
            gui.yrange.enabled = gui.arange.enabled = True
            gui.yrange.min = df[Mole.nth].min()
            gui.yrange.max = df[Mole.nth].max() + 1
            gui.arange.min = df[Mole.pf].min()
            gui.arange.max = df[Mole.pf].max() + 1
            gui.yrange.value = (gui.yrange.min, gui.yrange.max)
            gui.arange.value = (gui.arange.min, gui.arange.max)

    _on_layer_change(gui.layer.value)


def _local_transform(
    transformer: Callable[
        [CylinderModel, float, CylindricSliceConstructor], CylinderModel
    ],
    layer: MoleculesLayer,
    by: float,
    yrange: tuple[int, int],
    arange: tuple[int, int],
    allev: bool,
) -> tuple[CylSpline, CylinderModel]:
    model = _get_simulation_model(layer)
    spl = layer.source_spline
    if spl is None:
        raise ValueError("No spline is associated with this layer.")
    shift, indexer = _fill_shift(yrange, arange, by, model.shape)
    new_model = transformer(model, by, indexer)
    if allev:
        new_model = new_model.alleviate(shift != 0)
    return spl, new_model


@impl_preview(Simulator.expand, auto_call=True)
def _preview_local_expand(
    self: Simulator,
    layer: MoleculesLayer,
    by: float,
    yrange: tuple[int, int],
    arange: tuple[int, int],
    allev: bool,
):
    spl, model = _local_transform(
        CylinderModel.expand, layer, by, yrange, arange, allev
    )
    yield from _select_molecules(layer, yrange, arange, model, spl)


@impl_preview(Simulator.twist, auto_call=True)
def _preview_local_twist(
    self: Simulator,
    layer: MoleculesLayer,
    by: float,
    yrange: tuple[int, int],
    arange: tuple[int, int],
    allev: bool,
):
    spl, model = _local_transform(CylinderModel.twist, layer, by, yrange, arange, allev)
    yield from _select_molecules(layer, yrange, arange, model, spl)


@impl_preview(Simulator.dilate, auto_call=True)
def _preview_local_dilate(
    self: Simulator,
    layer: MoleculesLayer,
    by: float,
    yrange: tuple[int, int],
    arange: tuple[int, int],
    allev: bool,
):
    spl, model = _local_transform(
        CylinderModel.dilate, layer, by, yrange, arange, allev
    )
    yield from _select_molecules(layer, yrange, arange, model, spl)


@impl_preview(Simulator.displace, auto_call=True)
def _preview_displace(self, layer: MoleculesLayer, expand, twist, dilate):
    try:
        new_model = _get_shifted_model(layer, expand, twist, dilate)
    except Exception:
        yield
        return
    old_data = layer.data
    layer.data = new_model.to_molecules(layer.source_spline).pos
    yield
    layer.data = old_data


def _get_shifted_model(layer: MoleculesLayer, expand, twist, dilate):
    spl = layer.source_spline
    if spl is None:
        raise ValueError("No spline is associated with this layer.")
    model = _get_simulation_model(layer)
    df = layer.molecules.features
    shifts = list[NDArray[np.floating]]()
    for dx in [dilate, expand, twist]:
        _dx = _normalize_expression(dx)
        _shift = df.select(_dx).to_series().to_numpy()
        shifts.append(_shift.reshape(model.shape))
    shifts = np.stack(shifts, axis=2)
    shifts[:, :, 2] = np.deg2rad(shifts[:, :, 2])
    return model.add_shift(shifts)


def _select_molecules(
    layer: MoleculesLayer,
    yrange: tuple[int, int],
    arange: tuple[int, int],
    model: CylinderModel,
    spl: CylSpline,
):
    out = model.to_molecules(spl)
    layer.data = out.pos
    if_select = (
        out.features.select(
            pl.col(Mole.nth).is_between(*yrange, closed="left")
            & pl.col(Mole.pf).is_between(*arange, closed="left"),
        )
        .to_series()
        .to_numpy()
    )
    layer.selected_data = np.where(if_select)[0]
    layer.refresh()
    yield
    layer.data = layer.molecules.pos
    layer.selected_data = {}


def _make_simulated_mole_name(viewer: "napari.Viewer"):
    num = 0
    name = "Mole(Sim)"
    existing_names = {layer.name for layer in viewer.layers}
    while f"{name}-{num}" in existing_names:
        num += 1
    return f"{name}-{num}"
