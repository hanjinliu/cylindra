from typing import Iterator, Annotated, Any, TYPE_CHECKING

import weakref

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import impy as ip

from magicclass import (
    magicclass,
    set_design,
    field,
    vfield,
    confirm,
    abstractapi,
    impl_preview,
)
from magicclass.types import Path, Optional
from magicclass.logging import getLogger
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker

from acryo import TomogramSimulator, pipe

from cylindra._custom_layers import MoleculesLayer
from cylindra.widgets.widget_utils import capitalize
from cylindra.const import (
    FileFilter,
    nm,
    INTERPOLATION_CHOICES,
    PropertyNames as H,
    PREVIEW_LAYER_NAME,
)
from cylindra.utils import roundint, ceilint
from cylindra.components import CylTomogram
from ._child_widget import ChildWidget

if TYPE_CHECKING:
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
_ImageSize = Annotated[tuple[nm, nm, nm], {"label": "image size of Z, Y, X (nm)"}]
PROJECT_NAME = "simulation-project.tar"
_Logger = getLogger("cylindra")


def _simulate_tomogram_iter(nsr):
    n = len(nsr)
    yield f"(0/{n + 1}) Simulating projections"
    for i in range(n):
        yield f"({i + 1}/{n + 1}) Back-projection of {i}-th image"


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


@magicclass(widget_type="scrollable", record=False)
class ComponentList(ChildWidget):
    """List of components"""

    @magicclass
    class Header(ChildWidget):
        new = abstractapi()

    def _iter_components(self) -> Iterator[Component]:
        for wdt in self:
            if isinstance(wdt, Component):
                yield wdt

    def _as_input(self) -> list[tuple[str, Path]]:
        return [(comp.layer_name, Path(comp.path)) for comp in self._iter_components()]

    @set_design(text="+", location=Header)
    def new(self, layer: MoleculesLayer, template_path: Path.Read[FileFilter.IMAGE]):
        """Add a new component"""
        self.append(Component(template_path, layer))


@magicclass(record=False)
class Simulator(ChildWidget):
    component_list = field(ComponentList, name="components")

    def reset_choices(self, *_):
        to_remove = []
        for i, comp in enumerate(self.component_list._iter_components()):
            if layer := comp.layer:
                comp.layer_name = layer.name
            else:
                to_remove.append(i)
        for i in reversed(to_remove):
            self.component_list.pop(i)

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
        tilt_series = ip.asarray(tilt_series, axes=["degree", "y", "x"])
        return tilt_series.set_scale(y=scale, x=scale)

    @set_design(text=capitalize)
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
        size : (nm, nm, nm), default is (100., 200., 100.)
            Size of the image in nm, of (Z, Y, X).
        scale : nm, default is 0.25
            Pixel size of the image.
        """
        parent = self._get_main()
        shape = tuple(roundint(s / scale) for s in size)

        binsize = ceilint(0.96 / scale)
        # NOTE: zero-filled image breaks contrast limit calculation, and bad for
        # visual detection of the image edges.
        img = ip.zeros(shape, axes="zyx", name="simulated image")
        img.scale_unit = "nm"
        val = 100 * binsize**3
        img[:, 0, :] = img[:, -1, :] = img[:, :, 0] = img[:, :, -1] = val / 2
        tomo = CylTomogram.from_image(img, scale=scale, binsize=binsize)
        tomo.metadata["is_dummy"] = True
        parent._macro_offset = len(parent.macro)
        return parent._send_tomogram_to_viewer.with_args(tomo)

    def _get_spline_idx(self, *_) -> int:
        return self._get_main()._get_spline_idx()

    @set_design(text="Update model parameters")
    def update_cylinder_parameters(
        self,
        spline: Annotated[int, {"bind": _get_spline_idx}],
        spacing: Annotated[nm, {"min": 0.2, "max": 100.0, "step": 0.01, "label": "spacing (nm)"}] = 1.0,
        dimer_twist: Annotated[float, {"min": -45.0, "max": 45.0, "label": "dimer twist (deg)"}] = 0.0,
        start: Annotated[int, {"min": -50, "max": 50, "label": "start"}] = 0,
        npf: Annotated[int, {"min": 1, "label": "number of PF"}] = 1,
        radius: Annotated[nm, {"min": 0.5, "max": 50.0, "step": 0.5, "label": "radius (nm)"}] = 10.0,
    ):  # fmt: skip
        """
        Update cylinder model with new parameters.

        Local structural displacement will be deleted because this function may change the number
        of molecules. This function should be called first.

        Parameters
        ----------
        spacing : nm
            Axial spacing between molecules.
        dimer_twist : float
            Skew angle.
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
        spl.props.update_glob(
            {
                H.spacing: spacing,
                H.dimer_twist: dimer_twist,
                H.start: start,
                H.npf: npf,
                H.radius: radius,
            }
        )

    @impl_preview(update_cylinder_parameters, auto_call=True)
    def _preview_update_cylinder_parameters(
        self,
        spline: int,
        spacing: nm,
        dimer_twist: float,
        start: int,
        npf: int,
        radius: nm,
    ):
        main = self._get_main()
        spl = main.splines[spline]
        kwargs = {
            H.spacing: spacing,
            H.dimer_twist: dimer_twist,
            H.start: start,
            H.npf: npf,
            H.radius: radius,
        }
        model = spl.cylinder_model(**kwargs)
        out = model.to_molecules(spl)
        viewer = self.parent_viewer
        if PREVIEW_LAYER_NAME in viewer.layers:
            layer: Layer = viewer.layers[PREVIEW_LAYER_NAME]
            layer.data = out.pos
        else:
            layer = main.add_molecules(out, name=PREVIEW_LAYER_NAME)
            layer.face_color = "crimson"
        is_active = yield
        if not is_active and layer in viewer.layers:
            viewer.layers.remove(layer)

    def _get_components(self, *_):
        return self.component_list._as_input()

    @set_design(text=capitalize)
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
        nsr = list(round(float(_nsr), 4) for _nsr in nsr)
        main = self._get_main()
        degrees = np.linspace(*tilt_range, n_tilt)
        sino = self._prep_radon(components, degrees, order=interpolation)

        yield _on_radon_finished.with_args(sino, degrees)

        rng = ip.random.default_rng(seed)
        for i, nsr_val in enumerate(nsr):
            imax = sino.max()
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

    @set_design(text=capitalize)
    @dask_thread_worker.with_progress(desc="Simulating tomogram...")
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
        rec = sino_noise.iradon(
            degrees,
            central_axis="y",
            height=roundint(height / scale),
            order=interpolation,
        ).set_scale(zyx=scale, unit="nm")
        yield _on_iradon_finished.with_args(rec.mean("z"), f"N/S = {nsr:.1f}")

        rec.name = "Simulated tomogram"
        tomo = CylTomogram.from_image(
            rec, scale=scale, tilt=tilt_range, binsize=bin_size
        )
        return main._send_tomogram_to_viewer.with_args(tomo)

    @set_design(text=capitalize)
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
        main = self._get_main()
        degrees = np.linspace(*tilt_range, n_tilt)
        mole_layers = [main.mole_layers[layer_name] for layer_name, _ in components]
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

        rec.name = "Simulated tomogram"
        tomo = CylTomogram.from_image(
            rec, scale=sino.scale.x, tilt=tilt_range, binsize=bin_size
        )

        yield main._send_tomogram_to_viewer.with_args(tomo)

        @thread_worker.callback
        def _on_return():
            for layer in mole_layers:
                main.parent_viewer.add_layer(layer)

        return _on_return

    @set_design(text=capitalize)
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
        template_path : Path
            Path to the image used for the template.
        save_path : Path
            Path of the file where the tilt series will be saved.
        tilt_range : tuple of float
            Minimum and maximum tilt angles in degree.
        n_tilt : int
            Number of tilt angles between minimum and maximum angles.
        interpolation : int
            Interpolation method used during the simulation.
        """
        save_dir = _norm_save_dir(save_dir)
        degrees = np.linspace(*tilt_range, n_tilt)
        sino = self._prep_radon(components, degrees, order=interpolation)
        scale = sino.scale.x
        save_path = save_dir / "image.mrc"
        sino.set_axes("zyx").set_scale(zyx=scale, unit="nm").imsave(save_path)
        _Logger.print(f"Tilt series saved at {save_path}.")
        self._get_main().save_project(save_dir / PROJECT_NAME, molecules_ext=".parquet")
        return None


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


def _norm_save_dir(save_dir) -> Path:
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise ValueError(f"{save_dir.as_posix()} is not a directory.")

    # add noise and save image
    if not save_dir.exists():
        save_dir.mkdir()
        _Logger.print(f"Directory created at {save_dir}.")
    return save_dir
