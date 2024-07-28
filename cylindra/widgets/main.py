from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Annotated, Any, Literal, Sequence

import impy as ip
import macrokit as mk
import numpy as np
import polars as pl
from acryo import Molecules, SubtomogramLoader
from magicclass import (
    MagicTemplate,
    bind_key,
    box,
    confirm,
    do_not_record,
    field,
    get_function_gui,
    magicclass,
    nogui,
    set_design,
)
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.pyqtgraph import QtImageCanvas
from magicclass.logging import getLogger
from magicclass.types import Colormap as ColormapType
from magicclass.types import Optional, Path
from magicclass.undo import undo_callback
from magicclass.utils import thread_worker
from napari.layers import Layer

from cylindra import _config, _shared_doc, cylmeasure, utils, widget_utils
from cylindra._napari import LandscapeSurface, MoleculesLayer
from cylindra.components import CylSpline, CylTomogram, SplineConfig
from cylindra.const import (
    PREVIEW_LAYER_NAME,
    FileFilter,
    ImageFilter,
    Ori,
    SplineColor,
    nm,
)
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.project import CylindraProject, extract
from cylindra.widget_utils import (
    PolarsExprStr,
    PolarsExprStrOrScalar,
    add_molecules,
    capitalize,
    change_viewer_focus,
)
from cylindra.widgets import _progress_desc as _pdesc
from cylindra.widgets import subwidgets as _sw
from cylindra.widgets._accessors import MoleculesLayerAccessor
from cylindra.widgets._annotated import (
    MoleculesLayersType,
    MoleculesLayerType,
    assert_layer,
    assert_list_of_layers,
)
from cylindra.widgets._main_utils import (
    AutoSaver,
    SplineTracker,
    degrees_to_rotator,
    normalize_offsets,
    normalize_radius,
)
from cylindra.widgets._reserved_layers import ReservedLayers
from cylindra.widgets._widget_ext import (
    CheckBoxes,
    KernelEdit,
    OffsetEdit,
    ProtofilamentEdit,
    SingleRotationEdit,
)
from cylindra.widgets.sta import SubtomogramAveraging

if TYPE_CHECKING:
    import napari
    from napari.utils.events import Event

    from cylindra.components._base import BaseComponent
    from cylindra.widgets.batch import CylindraBatchWidget

DEFAULT_COLORMAP = {
    0.00: "#0B0000",  # black
    0.30: "#872D9D",  # purple
    0.68: "#FF0000",  # red
    1.00: "#FFFF00",  # yellow
}
_Logger = getLogger("cylindra")  # The GUI logger

# annotated types
_OffsetType = Annotated[
    Optional[tuple[nm, float]],
    {
        "text": "Infer offsets from spline global properties",
        "options": {"widget_type": OffsetEdit},
    },
]

_Interval = Annotated[
    Optional[nm],
    {
        "text": "Use existing anchors",
        "options": {"min": 1.0, "step": 0.5, "value": 50.0},
    },
]


def _validate_colormap(self, val):
    if isinstance(val, dict):
        val = {round(k, 3): utils.str_color(v) for k, v in val.items()}
    return val


_CmapType = Annotated[ColormapType, {"validator": _validate_colormap}]

# stylesheet
_WIDGETS_PATH = Path(__file__).parent
_STYLE = (
    _WIDGETS_PATH.joinpath("style.qss")
    .read_text()
    .replace("%QSS_PATH%", _WIDGETS_PATH.as_posix())
)
_NSPLINES = (
    "len(self.splines) if splines is None or isinstance(splines, str) else len(splines)"
)


def _choice_getter(method_name: str, dtype_kind: str = ""):
    def _get_choice(self: "CylindraMainWidget", w=None) -> list[str]:
        # don't use get_function_gui. It causes RecursionError.
        gui = self[method_name].mgui
        if gui is None or gui[0].value is None:
            return []
        features = gui[0].value.features
        if dtype_kind == "":
            return features.columns
        return [c for c in features.columns if features[c].dtype.kind in dtype_kind]

    _get_choice.__qualname__ = "CylindraMainWidget._get_choice"
    return _get_choice


############################################################################################
#   The Main Widget of cylindra
############################################################################################


@magicclass(
    widget_type="scrollable",
    stylesheet=_STYLE,
    name="cylindra",
    use_native_menubar=False,
)
@_shared_doc.update_cls
class CylindraMainWidget(MagicTemplate):
    # Main GUI class.

    # Widget for manual spline fitting
    spline_fitter = field(_sw.SplineFitter, name="_Spline fitter")
    # Widget for manual spline clipping
    spline_clipper = field(_sw.SplineClipper, name="_Spline clipper")
    # Widget for sweeping along splines
    spline_slicer = field(_sw.SplineSlicer, name="_Spline slicer")
    # Widget for pre-filtering/pre-processing
    image_processor = field(_sw.ImageProcessor, name="_Image Processor")
    # Widget for tomogram simulator
    simulator = field(_sw.Simulator, name="_Simulator")
    # Widget for measuring FFT parameters from a 2D power spectra
    spectra_inspector = field(_sw.SpectraInspector, name="_SpectraInspector")
    # Widget for subtomogram analysis
    sta = field(SubtomogramAveraging, name="_STA widget")

    mole_layers = MoleculesLayerAccessor()

    @property
    def batch(self) -> "CylindraBatchWidget":
        """Return the batch analyzer."""
        return self.AnalysisMenu.open_project_batch_analyzer()

    # Menu bar
    FileMenu = field(_sw.FileMenu, name="File")
    ImageMenu = field(_sw.ImageMenu, name="Image")
    SplinesMenu = field(_sw.SplinesMenu, name="Splines")
    MoleculesMenu = field(_sw.MoleculesMenu, name="Molecules")
    AnalysisMenu = field(_sw.AnalysisMenu, name="Analysis")
    OthersMenu = field(_sw.OthersMenu, name="Others")

    # Toolbar
    Toolbar = field(_sw.CylindraToolbar)

    # Child widgets
    GeneralInfo = field(_sw.GeneralInfo)
    # Widget for controling splines
    SplineControl = field(_sw.SplineControl)
    # Widget for summary of local properties
    LocalProperties = box.collapsible(field(_sw.LocalPropertiesWidget), text="Local Properties")  # fmt: skip
    # Widget for summary of glocal properties
    GlobalProperties = field(_sw.GlobalPropertiesWidget, name="Global Properties")  # fmt: skip
    # Widget for 2D overview of splines
    Overview = field(QtImageCanvas).with_options(tooltip="Overview of splines")  # fmt: skip

    ### methods ###

    def __init__(self):
        self._tomogram = CylTomogram.dummy(binsize=[1])
        self._reserved_layers = ReservedLayers()
        self._macro_offset: int = 1
        self._macro_image_load_offset: int = 1
        self._need_save: bool = False
        self._batch: "CylindraBatchWidget | None" = None
        self._project_dir: "Path | None" = None
        self._current_binsize: int = 1
        self.objectName()  # load napari types

    def __post_init__(self):
        self.min_width = 400
        self.LocalProperties.collapsed = False
        self.GlobalProperties.collapsed = False
        self.Overview.min_height = 300

        self.LocalProperties._props_changed.connect(
            lambda: self._update_local_properties_in_widget(replot=True)
        )

        # load all the workflows
        cfg = _config.get_config()
        for file in cfg.list_workflow_paths():
            try:
                self.OthersMenu.Workflows.append_workflow(file)
            except Exception as e:
                _Logger.exception(f"Failed to load workflow {file.stem}: {e}")

        # setup auto saver
        self._auto_saver = AutoSaver(self, sec=cfg.autosave_interval)

        # dask worker number
        if cfg.default_dask_n_workers is not None:
            if cfg.default_dask_n_workers <= 0:
                _Logger.warning("Invalid dask worker number. Set to default.")
            else:
                self.OthersMenu.configure_dask(cfg.default_dask_n_workers)

        @self.macro.on_appended.append
        def _on_appended(expr: mk.Expr):
            self._need_save = not str(expr).startswith("ui.open_image(")
            self._auto_saver.save()

        @self.macro.on_popped.append
        def _on_popped(*_):
            self._need_save = len(self.macro) >= self._macro_offset and not str(
                self.macro[-1]
            ).startswith("ui.open_image(")
            self._auto_saver.save()

        self.default_config = SplineConfig.from_file(cfg.default_spline_config_path)
        return None

    @property
    def tomogram(self) -> CylTomogram:
        """The current tomogram instance."""
        return self._tomogram

    @property
    def splines(self):
        """The spline list."""
        return self.tomogram.splines

    @property
    def default_config(self) -> SplineConfig:
        """Default spline configuration."""
        return self._default_cfg

    @default_config.setter
    def default_config(self, cfg: SplineConfig | dict[str, Any]):
        if not isinstance(cfg, SplineConfig):
            cfg = SplineConfig.from_dict(cfg, unknown="error")
        self._default_cfg = cfg
        self._refer_spline_config(cfg)

    @property
    def sub_viewer(self) -> "napari.Viewer":
        """The sub-viewer for subtomogram averages."""
        return self.sta.sub_viewer

    def _get_splines(self, widget=None) -> list[tuple[str, int]]:
        """Get list of spline objects for categorical widgets."""
        tomo = self.tomogram
        return [(f"({i}) {spl}", i) for i, spl in enumerate(tomo.splines)]

    def _get_spline_coordinates(self, coords=None) -> np.ndarray:
        """Get coordinates of the manually picked spline."""
        if coords is None:
            coords = self._reserved_layers.work.data
        out = np.round(coords, 3)
        if out.ndim != 2 or out.shape[1] != 3 or out.dtype.kind not in "iuf":
            raise ValueError("Input coordinates must be a (N, 3) numeric array.")
        return out

    def _get_available_binsize(self, _=None) -> list[int]:
        out = [x[0] for x in self.tomogram.multiscaled]
        if 1 not in out:
            out = [1, *out]
        return out

    def _get_default_config(self, config):
        if config is None:
            config = self.default_config.asdict()
        elif isinstance(config, dict):
            config = self.default_config.updated(**config).asdict()
        elif isinstance(config, SplineConfig):
            config = config.asdict()
        else:
            raise TypeError(f"Invalid config type: {type(config)}")
        return config

    def _splines_validator(self, splines) -> list[int] | Literal["all"]:
        """Validate list input, or 'all' if all splines are selected."""
        nspl = self.splines.count()
        if splines is None:
            splines = list(range(nspl))
        elif isinstance(splines, str):
            if splines == "all":
                return splines
            raise TypeError("Only 'all' is allow for a string input")
        elif splines is all:
            return "all"
        elif not hasattr(splines, "__iter__"):
            splines = [int(splines)]
        else:
            for i in splines:
                if i >= nspl:
                    raise ValueError(f"Spline index {i} is out of range.")
            splines = sorted(splines)
        if len(splines) == 0:
            raise ValueError("No spline is selected.")
        if splines == list(range(nspl)):
            # For better reusabiligy, recording as 'all' is better.
            return "all"
        return splines

    def _norm_splines(self, splines: list[int] | Literal["all"]) -> list[int]:
        if isinstance(splines, str) and splines == "all":
            return list(range(self.splines.count()))
        return splines

    _Splines = Annotated[
        list[int],
        {
            "choices": _get_splines,
            "widget_type": CheckBoxes,
            "validator": _splines_validator,
        },
    ]

    @set_design(icon="mdi:pen-add", location=Toolbar)
    @bind_key("F1")
    def register_path(
        self,
        coords: Annotated[np.ndarray, {"validator": _get_spline_coordinates}] = None,
        config: Annotated[dict[str, Any] | SplineConfig, {"validator": _get_default_config}] = None,
        err_max: Annotated[nm, {"bind": 0.5}] = 0.5,
    ):  # fmt: skip
        """Register points as a spline path."""
        if coords is None or coords.size == 0:
            raise ValueError("No points are given.")

        tomo = self.tomogram
        tomo.add_spline(coords, config=config, err_max=err_max)
        self._add_spline_instance(tomo.splines[-1])
        return undo_callback(self.delete_spline).with_args(-1)

    def _add_spline_instance(self, spl: "CylSpline"):
        # draw path
        tomo = self.tomogram
        self._add_spline_to_images(spl, len(tomo.splines) - 1)
        self._reserved_layers.work.data = []
        self._reserved_layers.prof.selected_data = set()
        self.reset_choices()
        self.SplineControl.num = len(tomo.splines) - 1
        return None

    _runner = field(_sw.Runner)
    _image_loader = _sw.ImageLoader
    _file_iterator = field(_sw.FileIterator)

    def _confirm_delete(self):
        i = self.SplineControl.num
        if i is None:
            # If user is writing the first spline, there's no spline registered.
            return False
        return self.tomogram.splines[i].has_props()

    @set_design(icon="solar:eraser-bold", location=Toolbar)
    @confirm(text="Spline has properties. Are you sure to delete it?", condition=_confirm_delete)  # fmt: skip
    @do_not_record(recursive=False)
    def clear_current(self):
        """Clear current selection."""
        if self._reserved_layers.work.data.size > 0:
            self._reserved_layers.work.data = []
        else:
            self.delete_spline(self.SplineControl.num)
        return None

    @set_design(icon="material-symbols:bomb", location=Toolbar)
    @confirm(text="Are you sure to clear all?\nYou cannot undo this.")
    @do_not_record
    def clear_all(self):
        """Clear all the splines and results."""
        self.macro.clear_undo_stack()
        self.Overview.layers.clear()
        self.tomogram.splines.clear()
        self._init_widget_state()
        self._init_layers()
        del self.macro[self._macro_image_load_offset + 1 :]
        self._need_save = False
        self.reset_choices()
        return None

    def _format_macro(self, macro: "mk.Macro | None" = None):
        if macro is None:
            macro = self.macro
        v = mk.Expr("getattr", [mk.symbol(self), "parent_viewer"])
        return macro.format([(mk.symbol(self.parent_viewer), v)])

    @do_not_record(recursive=False)
    @nogui
    def run_workflow(self, filename: str, *args, **kwargs):
        """
        Run a user-defined workflow.

        This method will run a .py file that was defined by the user from
        `Workflow > Define workflow`. *args and **kwargs follow the signature of the
        main function of the workflow.
        """
        main = _config.get_main_function(filename)
        out = main(self, *args, **kwargs)
        return out

    @set_design(text="Open", location=_image_loader)
    @dask_thread_worker.with_progress(desc="Reading image")
    @confirm(text="You may have unsaved data. Open a new tomogram?", condition="self._need_save")  # fmt: skip
    def open_image(
        self,
        path: Annotated[str | Path, {"bind": _image_loader.path}],
        scale: Annotated[nm, {"bind": _image_loader.scale.scale_value}] = None,
        tilt_range: Annotated[Any, {"bind": _image_loader.tilt_model}] = None,
        bin_size: Annotated[Sequence[int], {"bind": _image_loader.bin_size}] = [1],
        filter: Annotated[ImageFilter | None, {"bind": _image_loader.filter}] = ImageFilter.Lowpass,
        invert: Annotated[bool, {"bind": _image_loader.invert}] = False,
        eager: Annotated[bool, {"bind": _image_loader.eager}] = False
    ):  # fmt: skip
        """
        Load an image file and process it before sending it to the viewer.

        Parameters
        ----------
        path : Path
            Path to the tomogram. Must be 3-D image.
        scale : float, default 1.0
            Pixel size in nm/pixel unit.
        tilt_range : tuple of float, default None
            Range of tilt angles in degrees.
        bin_size : int or list of int, default [1]
            Initial bin size of image. Binned image will be used for visualization in the viewer.
            You can use both binned and non-binned image for analysis.
        {filter}
        invert : bool, default False
            If true, invert the intensity of the image.
        eager : bool, default False
            If true, the image will be loaded immediately. Otherwise, it will be loaded
            lazily.
        """
        img = ip.lazy.imread(path, chunks=_config.get_config().dask_chunk)
        if scale is not None:
            scale = float(scale)
            img.scale.x = img.scale.y = img.scale.z = scale
        else:
            scale = img.scale.x
        if isinstance(bin_size, int):
            bin_size = [bin_size]
        elif len(bin_size) == 0:
            raise ValueError("You must specify at least one bin size.")
        else:
            bin_size = list(set(bin_size))  # delete duplication
        tomo = CylTomogram.imread(
            path=path,
            scale=scale,
            tilt=tilt_range,
            binsize=bin_size,
            eager=eager,
        )
        self._macro_offset = len(self.macro)
        self._project_dir = None
        return self._send_tomogram_to_viewer.with_args(tomo, filter, invert=invert)

    @open_image.started.connect
    def _open_image_on_start(self):
        return self._image_loader.close()

    @set_design(text=capitalize, location=_sw.FileMenu)
    @thread_worker.with_progress(desc="Reading project", total=0)
    @confirm(text="You may have unsaved data. Open a new project?", condition="self._need_save")  # fmt: skip
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+P")
    def load_project(
        self,
        path: Path.Read[FileFilter.PROJECT],
        filter: ImageFilter | None = ImageFilter.Lowpass,
        read_image: Annotated[bool, {"label": "read image data"}] = True,
        update_config: bool = False,
    ):
        """
        Load a project file (project.json, tar file or zip file).

        Parameters
        ----------
        path : path-like or CylindraProject
            Path to the project file, or the project directory that contains a project
            file, or a CylindraProject object.
        {filter}
        read_image : bool, default True
            Whether to read image data from the project directory. If false, image data
            will be memory-mapped and will not be shown in the viewer. Unchecking this
            is useful to decrease loading time.
        update_config : bool, default False
            Whether to update the default spline configuration with the one described
            in the project.
        """
        if isinstance(path, CylindraProject):
            project = path
            project_path = project.project_path
        else:
            project = CylindraProject.from_file(path)
            project_path = project.project_path
        _Logger.print_html(
            f"<code>ui.load_project('{Path(project_path).as_posix()}', "
            f"filter={str(filter)!r}, {read_image=}, {update_config=})</code>"
        )
        if project_path is not None:
            _Logger.print(f"Project loaded: {project_path.as_posix()}")
            self._project_dir = project_path
        yield from project._to_gui(
            self,
            filter=filter,
            read_image=read_image,
            update_config=update_config,
        )

    @set_design(text=capitalize, location=_sw.FileMenu)
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+S")
    def save_project(
        self,
        path: Path.Save,
        molecules_ext: Literal[".csv", ".parquet"] = ".csv",
        save_landscape: Annotated[bool, {"label": "Save landscape layers"}] = False,
    ):
        """
        Save current project state and the results in a directory.

        The json file contains paths of images and results, parameters of splines,
        scales and version. Local and global properties will be exported as csv files.
        Molecule coordinates and features will be exported as the `molecules_ext`
        format. If results are saved at the default directory, they will be
        written as relative paths in the project json file so that moving root
        directory does not affect the loading behavior.

        Parameters
        ----------
        path : Path
            Path of json file.
        molecules_ext : str, default ".csv"
            Extension of the molecule file. Can be ".csv" or ".parquet".
        save_landscape : bool, default False
            Save landscape layers if any. False by default because landscape layers are
            usually large.
        """
        path = Path(path)
        CylindraProject.save_gui(self, path, molecules_ext, save_landscape)
        _Logger.print(f"Project saved: {path.as_posix()}")
        self._need_save = False
        self._project_dir = path
        autosave_path = _config.autosave_path()
        if autosave_path.exists():
            with suppress(Exception):
                autosave_path.unlink()
        return None

    @set_design(text=capitalize, location=_sw.FileMenu)
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+Shift+S")
    def overwrite_project(self):
        """Overwrite currently opened project."""
        if self._project_dir is None:
            raise ValueError(
                "No project is loaded. You can use `Save project` "
                "(ui.save_project(...)) to save the current state."
            )
        project = CylindraProject.from_file(self._project_dir)
        if project.molecules_info:
            ext = Path(project.molecules_info[0].name).suffix
        else:
            ext = ".csv"
        return self.save_project(self._project_dir, ext)

    @set_design(text=capitalize, location=_sw.FileMenu)
    def load_splines(self, paths: Path.Multiple[FileFilter.JSON]):
        """
        Load splines from a list of json paths.

        Parameters
        ----------
        paths : list of path-like objects
            Paths to json files that describe spline parameters in the correct format.
        """
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        splines = [CylSpline.from_json(path) for path in paths]
        self.tomogram.splines.extend(splines)
        self._update_splines_in_images()
        self.reset_choices()
        return None

    @set_design(text=capitalize, location=_sw.FileMenu)
    def load_molecules(self, paths: Path.Multiple[FileFilter.CSV]):
        """Load molecules from a csv file."""
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        for path in paths:
            mole = Molecules.from_file(path)
            name = Path(path).stem
            add_molecules(self.parent_viewer, mole, name)
        return None

    @set_design(text=capitalize, location=_sw.FileMenu)
    @do_not_record
    def save_spline(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        save_path: Path.Save[FileFilter.JSON],
    ):
        """Save splines as a json file."""
        spl = self.tomogram.splines[spline]
        spl.to_json(save_path)
        return None

    @do_not_record
    @set_design(text=capitalize, location=_sw.FileMenu)
    def save_molecules(
        self, layer: MoleculesLayerType, save_path: Path.Save[FileFilter.CSV]
    ):
        """
        Save monomer coordinates, orientation and features as a csv file.

        Parameters
        ----------
        {layer}
        save_path : Path
            Where to save the molecules.
        """
        return assert_layer(layer, self.parent_viewer).molecules.to_csv(save_path)

    @set_design(text=capitalize, location=_sw.FileMenu)
    @do_not_record
    def open_reference_image(self, path: Path.Read[FileFilter.IMAGE]):
        """
        Open an image as a reference image of the current tomogram.

        The input image is usually a denoised image created by other softwares, or
        simply a filtered image. Please note that this method does not check that the
        input image is appropriate as a reference of the current tomogram, as
        potentially any 3D image can be used.

        Parameters
        ----------
        path : path-like
            Path to the image file. The image must be 3-D.
        """
        img = ip.imread(path)
        return self._update_reference_image(img)

    @set_design(text=capitalize, location=_sw.FileMenu)
    @do_not_record
    def open_label_image(self, path: Path.Read[FileFilter.IMAGE]):
        """Open an image file as a label image of the current tomogram."""
        label = ip.imread(path)
        if label.ndim != 3:
            raise ValueError("Label image must be 3-D.")
        tr = self.tomogram.multiscale_translation(label.scale.x / self.tomogram.scale)
        label = self.parent_viewer.add_labels(
            label,
            name=label.name,
            translate=[tr, tr, tr],
            scale=list(label.scale),
            opacity=0.4,
        )
        self._reserved_layers.to_be_removed.add(label)
        return label

    @set_design(text=capitalize, location=_sw.ImageMenu)
    @dask_thread_worker.with_progress(desc=_pdesc.filter_image_fmt)
    @do_not_record
    def filter_reference_image(
        self,
        method: ImageFilter = ImageFilter.Lowpass,
    ):  # fmt: skip
        """Apply filter to enhance contrast of the reference image."""
        method = ImageFilter(method)
        if self.tomogram.is_dummy:
            return
        with utils.set_gpu():
            img = self._reserved_layers.image_data
            overlap = [min(s, 32) for s in img.shape]
            _tiled = img.tiled(chunks=(224, 224, 224), overlap=overlap)
            sigma = 1.6 / self._reserved_layers.scale
            match method:
                case ImageFilter.Lowpass:
                    img_filt = _tiled.lowpass_filter(cutoff=0.2)
                case ImageFilter.Gaussian:
                    img_filt = _tiled.gaussian_filter(sigma=sigma, fourier=True)
                case ImageFilter.DoG:
                    img_filt = _tiled.dog_filter(low_sigma=sigma, fourier=True)
                case ImageFilter.LoG:
                    img_filt = _tiled.log_filter(sigma=sigma)
                case _:  # pragma: no cover
                    raise ValueError(f"No method matches {method!r}")

        contrast_limits = np.percentile(img_filt, [1, 99.9])

        @thread_worker.callback
        def _filter_reference_image_on_return():
            self._reserved_layers.image.data = img_filt
            self._reserved_layers.image.contrast_limits = contrast_limits
            proj = self._reserved_layers.image.data.mean(axis="z")
            self.Overview.image = proj
            self.Overview.contrast_limits = contrast_limits

        return _filter_reference_image_on_return

    @thread_worker.with_progress(desc="Inverting image")
    @set_design(text=capitalize, location=_sw.ImageMenu)
    def invert_image(self):
        """Invert the intensity of the images."""
        self.tomogram.invert()
        if self._reserved_layers.is_lazy:

            @thread_worker.callback
            def _invert_image_on_return():
                return undo_callback(self.invert_image)

        else:
            img_inv = -self._reserved_layers.image.data
            cmin, cmax = np.percentile(img_inv, [1, 99.9])
            if cmin >= cmax:
                cmax = cmin + 1

            @thread_worker.callback
            def _invert_image_on_return():
                self._reserved_layers.image.data = img_inv
                self._reserved_layers.image.contrast_limits = (cmin, cmax)
                clow, chigh = self.Overview.contrast_limits
                self.Overview.image = -self.Overview.image
                self.Overview.contrast_limits = -chigh, -clow
                return undo_callback(self.invert_image)

        return _invert_image_on_return

    @set_design(text="Add multi-scale", location=_sw.ImageMenu)
    @dask_thread_worker.with_progress(desc=lambda bin_size: f"Adding multiscale (bin = {bin_size})")  # fmt: skip
    def add_multiscale(
        self,
        bin_size: Annotated[int, {"choices": list(range(2, 17))}] = 4,
    ):
        """
        Add a new multi-scale image of current tomogram.

        Parameters
        ----------
        bin_size : int, default 4
            Bin size of the new image
        """
        tomo = self.tomogram
        tomo.get_multiscale(binsize=bin_size, add=True)
        return thread_worker.callback(self.set_multiscale).with_args(bin_size)

    @set_design(text="Set multi-scale", location=_sw.ImageMenu)
    def set_multiscale(self, bin_size: Annotated[int, {"choices": _get_available_binsize}]):  # fmt: skip
        """
        Set multiscale used for image display.

        Parameters
        ----------
        bin_size: int
            Bin size of multiscaled image.
        """
        tomo = self.tomogram
        _old_bin_size = self._current_binsize
        imgb = tomo.get_multiscale(bin_size)
        factor = self._reserved_layers.scale / imgb.scale.x
        self._reserved_layers.update_image(imgb, tomo.multiscale_translation(bin_size))
        current_z = self.parent_viewer.dims.current_step[0]
        self.parent_viewer.dims.set_current_step(axis=0, value=current_z * factor)

        # update overview
        self.Overview.image = imgb.mean(axis="z")
        self.Overview.xlim = [x * factor for x in self.Overview.xlim]
        self.Overview.ylim = [y * factor for y in self.Overview.ylim]
        self._current_binsize = bin_size
        self.reset_choices()
        return undo_callback(self.set_multiscale).with_args(_old_bin_size)

    @set_design(text=capitalize, location=_sw.ImageMenu)
    def sample_subtomograms(self):
        """Sample subtomograms at the anchor points on splines"""
        self.spline_fitter.close()

        # initialize GUI
        if len(self.tomogram.splines) == 0:
            raise ValueError("No spline found.")
        spl = self.tomogram.splines[0]
        if spl.has_anchors:
            self.SplineControl["pos"].max = spl.anchors.size - 1
        self.SplineControl._num_changed()
        self._reserved_layers.work.mode = "pan_zoom"

        self._update_local_properties_in_widget()
        self._update_global_properties_in_widget()
        self._highlight_spline()

        # reset contrast limits
        self.SplineControl._reset_contrast_limits()
        return None

    def _get_spline_idx(self, *_) -> int:
        return self.SplineControl.num

    @set_design(text=capitalize, location=_sw.SplinesMenu.Orientation)
    def invert_spline(self, spline: Annotated[int, {"bind": _get_spline_idx}] = None):
        """
        Invert current displayed spline **in place**.

        Parameters
        ----------
        spline : int, optional
            ID of splines to be inverted.
        """
        if spline is None:
            return
        spl = self.tomogram.splines[spline]
        self.tomogram.splines[spline] = spl.invert()
        self._update_splines_in_images()
        self.reset_choices()

        need_resample = self.SplineControl.need_resample
        self._init_widget_state()
        if need_resample:
            self.sample_subtomograms()
        self._set_orientation_marker(spline)
        return undo_callback(self.invert_spline).with_args(spline)

    @set_design(text=capitalize, location=_sw.SplinesMenu.Orientation)
    def align_to_polarity(
        self, orientation: Literal["MinusToPlus", "PlusToMinus"] = "MinusToPlus"
    ):
        """
        Align all the splines in the direction parallel to the cylinder polarity.

        Parameters
        ----------
        orientation : Ori, default Ori.MinusToPlus
            To which direction splines will be aligned.
        """
        need_resample = self.SplineControl.need_resample
        _old_orientations = [spl.orientation for spl in self.tomogram.splines]
        self.tomogram.align_to_polarity(orientation=orientation)
        self._update_splines_in_images()
        self._init_widget_state()
        self.reset_choices()
        if need_resample:
            self.sample_subtomograms()
        for i in range(len(self.tomogram.splines)):
            self._set_orientation_marker(i)
        _new_orientations = [spl.orientation for spl in self.tomogram.splines]
        return (
            undo_callback(self._set_orientations)
            .with_args(_old_orientations, need_resample)
            .with_redo(lambda: self._set_orientations(_new_orientations))
        )

    @set_design(text=capitalize, location=_sw.SplinesMenu.Orientation)
    @thread_worker.with_progress(desc="Auto-detecting polarities...", total=_NSPLINES)
    def infer_polarity(
        self,
        splines: _Splines = None,
        depth: Annotated[nm, {"min": 5.0, "max": 500.0, "step": 5.0}] = 40,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Automatically detect the cylinder polarities.

        This function uses Fourier vorticity to detect the polarities of the splines.
        The subtomogram at the center of the spline will be sampled in the cylindric
        coordinate and the power spectra in (radius, angle) space will be calculated.
        The peak position of the `angle = nPF` line scan will be used to detect the
        polarity of the spline.

        Parameters
        ----------
        {splines}{depth}{bin_size}
        """
        tomo = self.tomogram
        _old_orientations = [spl.orientation for spl in self.tomogram.splines]
        for i in self._norm_splines(splines):
            tomo.infer_polarity(i=i, binsize=bin_size, depth=depth, update=True)
            yield
        _new_orientations = [spl.orientation for spl in self.tomogram.splines]

        @thread_worker.callback
        def _on_return():
            self._update_splines_in_images()
            for i in range(len(tomo.splines)):
                self._set_orientation_marker(i)

            self.SplineControl._update_canvas()
            return (
                undo_callback(self._set_orientations)
                .with_args(_old_orientations)
                .with_redo(lambda: self._set_orientations(_new_orientations))
            )

        return _on_return

    def _set_orientations(self, orientations: list[Ori], resample: bool = True):
        for spl, ori in zip(self.tomogram.splines, orientations, strict=True):
            spl.orientation = ori
        self._update_splines_in_images()
        self._init_widget_state()
        self.reset_choices()
        for i in range(len(self.tomogram.splines)):
            self._set_orientation_marker(i)
        if resample:
            self.sample_subtomograms()
        return None

    @set_design(text=capitalize, location=_sw.SplinesMenu)
    @bind_key("Ctrl+K, Ctrl+X")
    def clip_spline(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        lengths: Annotated[tuple[nm, nm], {"options": {"min": -1000.0, "max": 1000.0, "step": 0.1, "label": "clip length (nm)"}}] = (0.0, 0.0),
    ):  # fmt: skip
        """
        Clip selected spline at its edges by given lengths.

        Parameters
        ----------
        spline : int
            The ID of spline to be clipped.
        lengths : tuple of float, default (0., 0.)
            The length in nm to be clipped at the start and end of the spline.
        """
        if spline is None:
            return
        spl = self.tomogram.splines[spline]
        _old_spl = spl.copy()
        length = spl.length()
        start, stop = np.array(lengths) / length
        self.tomogram.splines[spline] = spl.clip(start, 1 - stop)
        self._update_splines_in_images()
        # current layer will be removed. Select another layer.
        self.parent_viewer.layers.selection = {self._reserved_layers.work}

        @undo_callback
        def out():
            self.tomogram.splines[spline] = _old_spl
            self._update_splines_in_images()

        return out

    @set_design(text=capitalize, location=_sw.SplinesMenu)
    @confirm(
        text="Spline has properties. Are you sure to delete it?",
        condition=_confirm_delete,
    )
    def delete_spline(self, i: Annotated[int, {"bind": _get_spline_idx}]):
        """Delete currently selected spline."""
        if i < 0:
            i = len(self.tomogram.splines) - 1
        spl = self.tomogram.splines.pop(i)
        self.reset_choices()

        # update layer
        features = self._reserved_layers.prof.features
        old_data = self._reserved_layers.prof.data
        self._reserved_layers.select_spline(i, len(self.tomogram.splines))
        self._update_splines_in_images()
        if self.SplineControl.need_resample and len(self.tomogram.splines) > 0:
            self.sample_subtomograms()

        @undo_callback
        def out():
            self.tomogram.splines.insert(i, spl)
            self._reserved_layers.prof.data = old_data
            self._reserved_layers.prof.features = features
            self._add_spline_to_images(spl, i)
            self._update_splines_in_images()
            self.reset_choices()

        return out

    @set_design(text=capitalize, location=_sw.SplinesMenu)
    def copy_spline(self, i: Annotated[int, {"bind": _get_spline_idx}]):
        """Make a copy of the current spline"""
        spl = self.tomogram.splines[i]
        self.tomogram.splines.append(spl.copy())
        self.reset_choices()
        self.SplineControl.num = len(self.tomogram.splines) - 1
        return undo_callback(self.delete_spline).with_args(-1)

    @set_design(text="Copy spline (new config)", location=_sw.SplinesMenu)
    def copy_spline_new_config(
        self,
        i: Annotated[int, {"bind": _get_spline_idx}],
        npf_range: Annotated[tuple[int, int], {"options": {"min": 2, "max": 100}}] = (11, 17),
        spacing_range: Annotated[tuple[nm, nm], {"options": {"step": 0.05}}] = (3.9, 4.3),
        twist_range: Annotated[tuple[float, float], {"options": {"min": -45.0, "max": 45.0, "step": 0.05}}] = (-1.0, 1.0),
        rise_range: Annotated[tuple[float, float], {"options": {"min": -45.0, "max": 45.0, "step": 0.1}}] = (0.0, 45.0),
        rise_sign: Literal[-1, 1] = -1,
        clockwise: Literal["PlusToMinus", "MinusToPlus"] = "MinusToPlus",
        thickness_inner: Annotated[nm, {"min": 0.0, "step": 0.1}] = 2.8,
        thickness_outer: Annotated[nm, {"min": 0.0, "step": 0.1}] = 2.8,
        fit_depth: Annotated[nm, {"min": 4.0, "step": 1}] = 49.0,
        fit_width: Annotated[nm, {"min": 4.0, "step": 1}] = 44.0,
        copy_props: bool = False,
    ):  # fmt: skip
        """Make a copy of the current spline with a new configuration."""
        config = locals()
        del config["i"], config["self"], config["copy_props"]
        spl = self.tomogram.splines[i]
        spl_new = spl.with_config(config, copy_props=copy_props)
        self.tomogram.splines.append(spl_new)
        self.reset_choices()
        self.SplineControl.num = len(self.tomogram.splines) - 1
        return undo_callback(self.delete_spline).with_args(-1)

    @set_design(text=capitalize, location=_sw.SplinesMenu.Fitting)
    @thread_worker.with_progress(desc="Spline Fitting", total=_NSPLINES)
    def fit_splines(
        self,
        splines: _Splines = None,
        max_interval: Annotated[nm, {"label": "max interval (nm)"}] = 30,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1.0,
        err_max: Annotated[nm, {"label": "max fit error (nm)", "step": 0.1}] = 1.0,
        degree_precision: float = 0.5,
        edge_sigma: Annotated[Optional[nm], {"text": "Do not mask image", "label": "edge σ"}] = 2.0,
        max_shift: nm = 5.0,
    ):  # fmt: skip
        """
        Fit splines to the cylinder by auto-correlation.

        Parameters
        ----------
        {splines}{max_interval}{bin_size}{err_max}
        degree_precision : float, default 0.5
            Precision of xy-tilt degree in angular correlation.
        edge_sigma : bool, default 2.0
            Check if cylindric structures are densely packed. Initial spline position
            must be "almost" fitted in dense mode.
        max_shift : nm, default 5.0
            Maximum shift to be applied to each point of splines.
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)
        with SplineTracker(widget=self, indices=splines) as tracker:
            for i in splines:
                tomo.fit(
                    i,
                    max_interval=max_interval,
                    binsize=bin_size,
                    err_max=err_max,
                    degree_precision=degree_precision,
                    edge_sigma=edge_sigma,
                    max_shift=max_shift,
                )
                yield thread_worker.callback(self._update_splines_in_images)

            @thread_worker.callback
            def out():
                self._init_widget_state()
                self._update_splines_in_images()
                return tracker.as_undo_callback()

        return out

    @set_design(text=capitalize, location=_sw.SplinesMenu.Fitting)
    @thread_worker.with_progress(desc="Spline Fitting", total=_NSPLINES)
    def fit_splines_by_centroid(
        self,
        splines: _Splines = None,
        max_interval: Annotated[nm, {"label": "max interval (nm)"}] = 30,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1.0,
        err_max: Annotated[nm, {"label": "max fit error (nm)", "step": 0.1}] = 1.0,
        max_shift: nm = 5.0,
    ):  # fmt: skip
        """
        Fit splines to the cylinder by centroid of sub-volumes.

        Parameters
        ----------
        {splines}{max_interval}{bin_size}{err_max}
        max_shift : nm, default 5.0
            Maximum shift to be applied to each point of splines.
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)
        with SplineTracker(widget=self, indices=splines) as tracker:
            for i in splines:
                tomo.fit_centroid(
                    i,
                    max_interval=max_interval,
                    binsize=bin_size,
                    err_max=err_max,
                    max_shift=max_shift,
                )
                yield thread_worker.callback(self._update_splines_in_images)

            @thread_worker.callback
            def out():
                self._init_widget_state()
                self._update_splines_in_images()
                return tracker.as_undo_callback()

        return out

    @set_design(text=capitalize, location=_sw.SplinesMenu)
    def add_anchors(
        self,
        splines: _Splines = None,
        interval: Annotated[nm, {"label": "Interval between anchors (nm)", "min": 1.0}] = 25.0,
        how: Literal["pack", "equal"] = "pack",
    ):  # fmt: skip
        """
        Add anchors to splines.

        Parameters
        ----------
        {splines}{interval}
        how : str, default "pack"
            How to add anchors.

            - "pack": (x———x———x—) Pack anchors from the starting point of splines.
            - "equal": (x——x——x——x) Equally distribute anchors between the starting
              point and the end point of splines. Actual intervals will be smaller.
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)
        with SplineTracker(widget=self, indices=splines) as tracker:
            match how:
                case "pack":
                    tomo.make_anchors(splines, interval=interval)
                case "equal":
                    tomo.make_anchors(splines, max_interval=interval)
                case _:  # pragma: no cover
                    raise ValueError(f"Unknown method: {how}")

            self._update_splines_in_images()
            return tracker.as_undo_callback()

    @set_design(text=capitalize, location=_sw.SplinesMenu.Fitting)
    @thread_worker.with_progress(desc="Refining splines", total=_NSPLINES)
    def refine_splines(
        self,
        splines: _Splines = None,
        max_interval: Annotated[nm, {"label": "maximum interval (nm)"}] = 30,
        err_max: Annotated[nm, {"label": "max fit error (nm)", "step": 0.1}] = 0.8,
        corr_allowed: Annotated[float, {"label": "correlation allowed", "max": 1.0, "step": 0.1}] = 0.9,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Refine splines using the global cylindric structural parameters.

        Parameters
        ----------
        {splines}{max_interval}{err_max}
        corr_allowed : float, default 0.9
            How many images will be used to make template for alignment. If 0.9, then
            top 90% will be used.
        {bin_size}
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)
        with SplineTracker(widget=self, indices=splines) as tracker:
            for i in splines:
                tomo.refine(
                    i,
                    max_interval=max_interval,
                    corr_allowed=corr_allowed,
                    err_max=err_max,
                    binsize=bin_size,
                )
                yield thread_worker.callback(self._update_splines_in_images)

            @thread_worker.callback
            def out():
                self._init_widget_state()
                self._update_splines_in_images()
                self._update_local_properties_in_widget()
                return tracker.as_undo_callback()

        return out

    @set_design(text="Set spline properties", location=_sw.SplinesMenu)
    def set_spline_props(
        self,
        spline: Annotated[int, {"bind": _get_spline_idx}],
        npf: Annotated[Optional[int], {"label": "number of PF", "text": "Do not update"}] = None,
        start: Annotated[Optional[int], {"label": "start number", "text": "Do not update"}] = None,
        orientation: Annotated[Optional[Literal["MinusToPlus", "PlusToMinus"]], {"text": "Do not update"}] = None,
    ):  # fmt: skip
        """
        Set spline global properties.

        This method will overwrite spline properties with the user input. You should
        not call this method unless there's a good reason to do so, e.g. the number
        of protofilaments is obviously wrong.

        Parameters
        ----------
        npf : int, optional
            If given, update the number of protofilaments.
        start : int, optional
            If given, update the start number of the spline.
        orientation : str, optional
            If given, update the spline orientation.
        """
        spl = self.tomogram.splines[spline]
        old_spl = spl.copy()
        spl.update_props(npf=npf, start=start, orientation=orientation)
        self.sample_subtomograms()
        self._update_splines_in_images()

        @undo_callback
        def out():
            self.tomogram.splines[spline] = old_spl
            self.sample_subtomograms()
            self._update_splines_in_images()

        return out

    @set_design(text=capitalize, location=_sw.MoleculesMenu.FromToSpline)
    def molecules_to_spline(
        self,
        layers: MoleculesLayersType = (),
        err_max: Annotated[nm, {"label": "Max fit error (nm)", "step": 0.1}] = 0.8,
        delete_old: Annotated[bool, {"label": "Delete old splines"}] = True,
        inherits: Annotated[Optional[list[str]], {"label": "Properties to inherit", "text": "All properties"}] = None,
        missing_ok: Annotated[bool, {"label": "Missing OK"}] = False,
        update_sources: Annotated[bool, {"label": "Update all the spline sources"}] = True,
    ):  # fmt: skip
        """
        Create splines from molecules.

        This function is useful to refine splines using results of subtomogram
        alignment. If the molecules layer alreadly has a source spline, replace
        it with the new one.
        Note that this function only works with molecules that is correctly
        assembled by such as :func:`map_monomers`.

        Parameters
        ----------
        {layers}{err_max}
        delete_old : bool, default True
            If True, delete the old spline if the molecules has one. For instance, if
            "Mole-0" has the spline "Spline-0" as the source, and a spline "Spline-1" is
            created from "Mole-0", then "Spline-0" will be deleted from the list.
        inherits : bool, optional
            Which global properties to be copied to the new one. If None, all the properties
            will be copied.
        missing_ok : bool, default False
            If False, raise an error if the source spline is not found in the tomogram.
        update_sources : bool, default True
            If True, all the molecules with the out-of-date source spline will be updated
            to the newly created splines. For instance, if "Mole-0" and "Mole-1" have the
            spline "Spline-0" as the source, and a spline "Spline-1" is created from
            "Mole-1", then the source of "Mole-1" will be updated to "Spline-1" as well.
        """
        tomo = self.tomogram
        layers = assert_list_of_layers(layers, self.parent_viewer)

        # first check missing_ok=False case
        if not missing_ok:
            for layer in layers:
                # NOTE: The source spline may not exist in the list
                if _s := layer.source_spline:
                    tomo.splines.index(_s)  # raise error here if not found

        for layer in layers:
            if _s := layer.source_spline:
                _config = _s.config
            else:
                _config = self.default_config
            _shape = (*layer.regular_shape(), 3)
            coords = layer.molecules.pos.reshape(_shape).mean(axis=1)
            spl = CylSpline(config=_config).fit(coords, err_max=err_max)
            try:
                idx = tomo.splines.index(layer.source_spline)
            except ValueError:
                tomo.splines.append(spl)
            else:
                old_spl = tomo.splines[idx]
                if inherits is None:
                    spl.props.glob = old_spl.props.glob.clone()
                else:
                    glob = old_spl.props.glob
                    spl.props.glob = {k: glob[k] for k in glob.columns if k in inherits}

                # Must be updated here, otherwise each.source_component may return
                # None since GC may delete the old spline.
                if update_sources:
                    for each in self.mole_layers:
                        if each.source_component is old_spl:
                            each.source_component = spl
                if delete_old:
                    tomo.splines[idx] = spl
                else:
                    tomo.splines.append(spl)
            layer.source_component = spl

        self.reset_choices()
        self.sample_subtomograms()
        self._update_splines_in_images()
        return None

    @set_design(text=capitalize, location=_sw.MoleculesMenu.FromToSpline)
    def protofilaments_to_spline(
        self,
        layer: MoleculesLayerType,
        err_max: Annotated[nm, {"label": "Max fit error (nm)", "step": 0.1}] = 0.8,
        ids: list[int] = (),
        config: Annotated[dict[str, Any] | SplineConfig, {"validator": _get_default_config}] = None,
    ):  # fmt: skip
        """
        Convert protofilaments to splines.

        If no IDs are given, all the molecules will be fitted to a spline, therefore
        essentially the same as manual filament picking. If IDs are given, selected
        protofilaments will be fitted to a spline separately.

        Parameters
        ----------
        {layer}{err_max}
        ids : list of int, default ()
            Protofilament IDs to be converted.
        """
        layer = assert_layer(layer, self.parent_viewer)
        tomo = self.tomogram
        mole = layer.molecules
        if len(ids) == 0:
            tomo.add_spline(mole.pos, err_max=err_max, config=config)
        for i in ids:
            sub = mole.filter(pl.col(Mole.pf) == i)
            if sub.count() == 0:
                continue
            tomo.add_spline(sub.sort(Mole.nth).pos, err_max=err_max, config=config)
        self.reset_choices()
        self._update_splines_in_images()
        return None

    @set_design(text=capitalize, location=_sw.AnalysisMenu.Radius)
    @thread_worker.with_progress(desc="Measuring Radius", total=_NSPLINES)
    def measure_radius(
        self,
        splines: _Splines = None,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        min_radius: Annotated[nm, {"min": 0.1, "step": 0.1}] = 1.0,
        max_radius: Annotated[nm, {"min": 0.1, "step": 0.1}] = 100.0,
    ):  # fmt: skip
        """
        Measure cylinder radius for each spline curve.

        Parameters
        ----------
        {splines}{bin_size}{min_radius}{max_radius}
        """
        splines = self._norm_splines(splines)
        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                self.tomogram.measure_radius(
                    i, binsize=bin_size, min_radius=min_radius, max_radius=max_radius
                )
                yield

            return tracker.as_undo_callback()

    @set_design(text=capitalize, location=_sw.AnalysisMenu.Radius)
    def set_radius(
        self,
        splines: _Splines = None,
        radius: PolarsExprStrOrScalar = 10.0,
    ):  # fmt: skip
        """
        Set radius of the splines.

        Parameters
        ----------
        {splines}
        radius : float or str expression
            Radius of the spline. If a string expression is given, it will be evaluated to get
            the polars.Expr object. The returned expression will be evaluated with the global
            properties of the spline as the context.
        """
        radius_expr = widget_utils.norm_scalar_expr(radius)
        splines = self._norm_splines(splines)
        rdict = dict[int, float]()
        for i in splines:
            _radius = self.splines[i].props.get_glob(radius_expr)
            if not isinstance(_radius, (int, float)):
                raise ValueError(
                    f"Radius must be converted into a number, got {_radius!r}."
                )
            if _radius <= 0:
                raise ValueError(f"Radius must be positive, got {_radius}.")
            rdict[i] = _radius
        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                self.splines[i].radius = rdict[i]
            return tracker.as_undo_callback()

    @set_design(text=capitalize, location=_sw.AnalysisMenu.Radius)
    @thread_worker.with_progress(desc="Measuring local radii", total=_NSPLINES)
    def measure_local_radius(
        self,
        splines: _Splines = None,
        interval: _Interval = None,
        depth: Annotated[nm, {"min": 2.0, "step": 0.5}] = 50.0,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        min_radius: Annotated[nm, {"min": 0.1, "step": 0.1}] = 1.0,
        max_radius: Annotated[nm, {"min": 0.1, "step": 0.1}] = 100.0,
        update_glob: Annotated[bool, {"text": "Also update the global radius"}] = True,
    ):  # fmt: skip
        """
        Measure radius for each local region along splines.

        Parameters
        ----------
        {splines}{interval}{depth}{bin_size}{min_radius}{max_radius}{update_glob}
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)

        @thread_worker.callback
        def _on_yield():
            self._update_local_properties_in_widget(replot=True)

        with SplineTracker(widget=self, indices=splines) as tracker:
            for i in splines:
                if interval is not None:
                    tomo.make_anchors(i=i, interval=interval)
                tomo.local_radii(
                    i=i,
                    depth=depth,
                    binsize=bin_size,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    update_glob=update_glob,
                )
                if i == splines[-1]:
                    yield _on_yield
                else:
                    yield

            return tracker.as_undo_callback()

    @set_design(text=capitalize, location=_sw.AnalysisMenu.Radius)
    def measure_radius_by_molecules(
        self,
        layers: MoleculesLayersType = (),
        interval: _Interval = None,
        depth: Annotated[nm, {"min": 2.0, "step": 0.5}] = 50.0,
        update_glob: Annotated[bool, {"text": "Also update the global radius"}] = True,
    ):  # fmt: skip
        """
        Measure local and global radius for each layer.

        Please note that the radius defined by the peak of the radial profile is not always
        the same as the radius measured by this method. If the molecules are aligned using
        a template image whose mass density is not centered, these radii may differ a lot.

        Parameters
        ----------
        {layers}{interval}{depth}{update_glob}
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)

        # check duplicated spline sources
        _splines = list[CylSpline]()
        _radius_df = list[pl.DataFrame]()
        _duplicated = list[CylSpline]()
        for layer in layers:
            spl = _assert_source_spline_exists(layer)
            if any(spl is each for each in _splines):
                _duplicated.append(spl)
            _splines.append(spl)
            mole = layer.molecules
            df = mole.features
            _radius_df.append(df.with_columns(cylmeasure.calc_radius(mole, spl)))

        if _duplicated:
            _layer_names = ", ".join(repr(l.name) for l in layers)
            raise ValueError(f"Layers {_layer_names} have duplicated spline sources.")

        indices = [self.tomogram.splines.index(spl) for spl in _splines]
        with SplineTracker(widget=self, indices=indices) as tracker:
            for i, spl, df in zip(indices, _splines, _radius_df, strict=True):
                if interval is not None:
                    self.tomogram.make_anchors(i=i, interval=interval)
                radii = list[float]()
                for pos in spl.anchors * spl.length():
                    lower, upper = pos - depth / 2, pos + depth / 2
                    pred = pl.col(Mole.position).is_between(lower, upper, closed="left")
                    radii.append(df.filter(pred)[Mole.radius].mean())
                radii = pl.Series(H.radius, radii, dtype=pl.Float32)
                if radii.is_nan().any():
                    _Logger.print_html(f"<b>Spline-{i} contains NaN radius.</b>")
                spl.props.update_loc([radii], depth, bin_size=1)
                if update_glob:
                    spl.radius = df[Mole.radius].mean()
            self._update_local_properties_in_widget(replot=True)
            return tracker.as_undo_callback()

    @set_design(text="Local CFT analysis", location=_sw.AnalysisMenu)
    @thread_worker.with_progress(desc="Local Cylindric Fourier transform", total=_NSPLINES)  # fmt: skip
    def local_cft_analysis(
        self,
        splines: _Splines = None,
        interval: _Interval = None,
        depth: Annotated[nm, {"min": 2.0, "step": 0.5}] = 50.0,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        radius: Literal["local", "global"] = "global",
        update_glob: Annotated[bool, {"text": "Also update the global properties"}] = False,
    ):  # fmt: skip
        """
        Determine local lattice parameters by local cylindric Fourier transformation.

        This method will sample subtomograms at given intervals and calculate the power
        spectra in a cylindrical coordinate. The peak position of the power spectra will
        used to determine the lattice parameters. Note that if the interval differs from
        the current spline anchors, the old local properties will be dropped.

        Parameters
        ----------
        {splines}{interval}{depth}{bin_size}
        radius : str, default "global"
            If "local", use the local radius for the analysis. If "global", use the
            global radius.
        {update_glob}
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)

        # first check radius
        match radius:
            case "global":
                for i in splines:
                    if tomo.splines[i].radius is None:
                        raise ValueError(
                            f"Global Radius of {i}-th spline is not measured yet. Please "
                            "measure the radius first from `Analysis > Radius`."
                        )
            case "local":
                for i in splines:
                    if not tomo.splines[i].props.has_loc(H.radius):
                        raise ValueError(
                            f"Local Radius of {i}-th spline is not measured yet. Please "
                            "measure the radius first from `Analysis > Radius`."
                        )
                if interval is not None:
                    raise ValueError(
                        "With `interval`, local radius values will be dropped. Please "
                        "set `radius='global'` or `interval=None`."
                    )
            case _:
                raise ValueError(f"radius must be 'local' or 'global', got {radius!r}.")

        @thread_worker.callback
        def _local_cft_analysis_on_yield(i: int):
            self._update_splines_in_images()
            if i == self.SplineControl.num:
                self.sample_subtomograms()

        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                if interval is not None:
                    tomo.make_anchors(i=i, interval=interval)
                tomo.local_cft_params(
                    i=i,
                    depth=depth,
                    binsize=bin_size,
                    radius=radius,
                    update_glob=update_glob,
                )
                yield _local_cft_analysis_on_yield.with_args(i)
            return tracker.as_undo_callback()

    @set_design(text="Global CFT analysis", location=_sw.AnalysisMenu)
    @thread_worker.with_progress(
        desc="Global Cylindric Fourier transform", total=_NSPLINES
    )
    def global_cft_analysis(
        self,
        splines: _Splines = None,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Determine cylindrical global structural parameters by Fourier transformation.

        Parameters
        ----------
        {splines}{bin_size}
        """
        tomo = self.tomogram
        splines = self._norm_splines(splines)

        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                spl = tomo.splines[i]
                if spl.radius is None:
                    tomo.measure_radius(i=i)
                tomo.global_cft_params(i=i, binsize=bin_size)
                yield

            # show all in a table
            @thread_worker.callback
            def _global_cft_analysis_on_return():
                df = (
                    self.tomogram.splines.collect_globalprops()
                    .drop(H.spline_id)
                    .to_pandas()
                    .transpose()
                )
                df.columns = [f"Spline-{i}" for i in range(len(df.columns))]
                self.sample_subtomograms()
                _Logger.print_table(df, precision=3)
                self._update_global_properties_in_widget()

                return tracker.as_undo_callback()

        return _global_cft_analysis_on_return

    def _get_reanalysis_macro(self, path: Path):
        """Get the macro expression for reanalysis in the given project path."""
        _ui_sym = mk.symbol(self)
        project = CylindraProject.from_file(path)
        with project.open_project() as _dir:
            macro_path = _dir / "script.py"
            macro_expr = extract(macro_path.read_text())
        return _filter_macro_for_reanalysis(macro_expr, _ui_sym)

    @set_design(text="Re-analyze current tomogram", location=_sw.AnalysisMenu)
    @do_not_record
    def reanalyze_image(self):
        """
        Reanalyze the current tomogram.

        This method will extract the first manual operations from current session.
        """
        _ui_sym = mk.symbol(self)
        macro_expr = self._format_macro()[self._macro_image_load_offset :]
        macro = _filter_macro_for_reanalysis(macro_expr, _ui_sym)
        self.clear_all()
        mk.Expr(mk.Head.block, macro.args[1:]).eval({_ui_sym: self})
        return self.macro.clear_undo_stack()

    @set_design(text="Re-analyze with new config", location=_sw.AnalysisMenu)
    @do_not_record
    def reanalyze_image_config_updated(self):
        """
        Reanalyze the current tomogram with newly set default spline config.

        This method is useful when you have mistakenly drawn splines with wrong spline
        config.
        """
        _ui_sym = mk.symbol(self)
        macro_expr = self._format_macro()[self._macro_image_load_offset :]
        macro = _filter_macro_for_reanalysis(macro_expr, _ui_sym)
        macro = _remove_config_kwargs(macro)
        self.clear_all()
        mk.Expr(mk.Head.block, macro.args[1:]).eval({_ui_sym: self})
        return self.macro.clear_undo_stack()

    @set_design(text="Re-analyze project", location=_sw.AnalysisMenu)
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+L")
    def load_project_for_reanalysis(self, path: Path.Read[FileFilter.PROJECT]):
        """
        Load a project file to re-analyze the data.

        This method will extract the first manual operations from a project file and
        run them. This is useful when you want to re-analyze the data with a different
        parameter set, or when there were some improvements in cylindra.
        """
        macro = self._get_reanalysis_macro(path)
        macro.eval({mk.symbol(self): self})
        return self.macro.clear_undo_stack()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @set_design(text=capitalize, location=_sw.MoleculesMenu.FromToSpline)
    @bind_key("M")
    @thread_worker.with_progress(desc="Mapping monomers", total=_NSPLINES)
    def map_monomers(
        self,
        splines: _Splines = None,
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
        offsets: _OffsetType = None,
        radius: Optional[nm] = None,
        extensions: Annotated[tuple[int, int], {"options": {"min": -100}}] = (0, 0),
        prefix: str = "Mole",
    ):  # fmt: skip
        """
        Map monomers as a regular cylindric grid assembly.

        This method uses the spline global properties.

        Parameters
        ----------
        {splines}{orientation}{offsets}
        radius : nm, optional
            Radius of the cylinder to position monomers.
        extensions : (int, int), default (0, 0)
            Number of molecules to extend. Should be a tuple of (prepend, append).
            Negative values will remove molecules.
        {prefix}
        """
        tomo = self.tomogram

        _Logger.print_html("<code>map_monomers</code>")
        _added_layers = list[MoleculesLayer]()

        @thread_worker.callback
        def _add_molecules(mol: Molecules, name: str, spl: CylSpline):
            layer = self.add_molecules(mol, name, source=spl)
            _added_layers.append(layer)
            _Logger.print(f"{name!r}: n = {len(mol)}")

        for i in self._norm_splines(splines):
            spl = tomo.splines[i]
            mol = tomo.map_monomers(
                i=i,
                orientation=orientation,
                offsets=normalize_offsets(offsets, spl),
                radius=normalize_radius(radius, spl),
                extensions=extensions,
            )

            cb = _add_molecules.with_args(mol, f"{prefix}-{i}", spl)
            yield cb
            cb.await_call()

        return self._undo_callback_for_layer(_added_layers)

    @set_design(text=capitalize, location=_sw.MoleculesMenu.FromToSpline)
    def map_monomers_with_extensions(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        n_extend: Annotated[dict[int, tuple[int, int]], {"label": "prepend/append", "widget_type": ProtofilamentEdit}] = {},
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
        offsets: _OffsetType = None,
        radius: Optional[nm] = None,
        prefix: str = "Mole",
    ):  # fmt: skip
        """
        Map monomers as a regular cylindric grid assembly.

        This method uses the spline global properties.

        Parameters
        ----------
        {spline}
        n_extend : dict[int, (int, int)]
            Number of molecules to extend. Should be mapping from the PF index to the (prepend,
            append) number of molecules to add. Remove molecules if negative values are given.
        {orientation}{offsets}
        radius : nm, optional
            Radius of the cylinder to position monomers.
        {prefix}
        """
        tomo = self.tomogram
        spl = tomo.splines[spline]
        coords = widget_utils.coordinates_with_extensions(spl, n_extend)
        mole = tomo.map_on_grid(
            i=spline,
            coords=coords,
            orientation=orientation,
            offsets=normalize_offsets(offsets, spl),
            radius=normalize_radius(radius, spl),
        )
        layer = self.add_molecules(mole, f"{prefix}-{spline}", source=spl)
        return self._undo_callback_for_layer(layer)

    @set_design(text=capitalize, location=_sw.MoleculesMenu.FromToSpline)
    def map_along_spline(
        self,
        splines: _Splines = None,
        molecule_interval: PolarsExprStrOrScalar = "col('spacing')",
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
        rotate_molecules: bool = True,
        prefix: str = "Center",
    ):  # fmt: skip
        """
        Map molecules along splines. Each molecule is rotated by skewing.

        Parameters
        ----------
        {splines}{molecule_interval}{orientation}
        rotate_molecules : bool, default True
            If True, rotate molecules by the "twist" parameter of each spline.
        {prefix}
        """
        tomo = self.tomogram
        interv_expr = widget_utils.norm_scalar_expr(molecule_interval)
        splines = self._norm_splines(splines)
        _Logger.print_html("<code>map_along_spline</code>")
        _added_layers = list[MoleculesLayer]()
        for idx in splines:
            spl = tomo.splines[idx]
            interv = spl.props.get_glob(interv_expr)
            mole = tomo.map_centers(
                i=idx,
                interval=interv,
                orientation=orientation,
                rotate_molecules=rotate_molecules,
            )
            _name = f"{prefix}-{idx}"
            layer = self.add_molecules(mole, _name, source=spl)
            _added_layers.append(layer)
            _Logger.print(f"{_name!r}: n = {mole.count()}")
        return self._undo_callback_for_layer(_added_layers)

    @set_design(text="Map alogn PF", location=_sw.MoleculesMenu.FromToSpline)
    def map_along_pf(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        molecule_interval: PolarsExprStrOrScalar = "col('spacing')",
        offsets: _OffsetType = None,
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
        prefix: str = "PF",
    ):  # fmt: skip
        """
        Map molecules along the line of a protofilament.

        Parameters
        ----------
        {spline}{molecule_interval}{offsets}{orientation}{prefix}
        """
        tomo = self.tomogram
        interv_expr = widget_utils.norm_scalar_expr(molecule_interval)
        spl = tomo.splines[spline]
        _Logger.print_html("<code>map_along_PF</code>")
        mol = tomo.map_pf_line(
            i=spline,
            interval=spl.props.get_glob(interv_expr),
            offsets=normalize_offsets(offsets, spl),
            orientation=orientation,
        )
        _name = f"{prefix}-{spline}"
        layer = self.add_molecules(mol, _name, source=spl)
        _Logger.print(f"{_name!r}: n = {len(mol)}")
        return self._undo_callback_for_layer(layer)

    @set_design(text=capitalize, location=_sw.MoleculesMenu.FromToSpline)
    def set_source_spline(
        self,
        layer: MoleculesLayerType,
        spline: Annotated[int, {"choices": _get_splines}],
    ):
        """
        Set source spline for a molecules layer.

        Parameters
        ----------
        {layer}{spline}
        """
        layer = assert_layer(layer, self.parent_viewer)
        old_spl = layer.source_component
        layer.source_component = self.tomogram.splines[spline]

        @undo_callback
        def _undo():
            layer.source_component = old_spl

        return _undo

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Combine)
    def concatenate_molecules(
        self,
        layers: MoleculesLayersType,
        name: str = "Mole-concat",
    ):  # fmt: skip
        """
        Concatenate selected molecules and create a new ones.

        Parameters
        ----------
        {layers}
        name : str, default "Mole-concat"
            Name of the new molecules layer.
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        all_molecules = Molecules.concat([layer.molecules for layer in layers])
        points = add_molecules(self.parent_viewer, all_molecules, name=name)

        # logging
        layer_names = list[str]()
        for layer in layers:
            layer.visible = False
            layer_names.append(layer.name)

        _Logger.print_html("<code>concatenate_molecules</code>")
        _Logger.print("Concatenated:", ", ".join(layer_names))
        _Logger.print(f"{points.name!r}: n = {len(all_molecules)}")
        return self._undo_callback_for_layer(points)

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Combine)
    def merge_molecule_info(
        self,
        pos: MoleculesLayerType,
        rotation: MoleculesLayerType,
        features: MoleculesLayerType,
    ):
        """
        Merge molecule info from different molecules.

        Parameters
        ----------
        pos : MoleculesLayer
            Molecules whose positions are used.
        rotation : MoleculesLayer
            Molecules whose rotations are used.
        features : MoleculesLayer
            Molecules whose features are used.
        """
        pos = assert_layer(pos, self.parent_viewer)
        rotation = assert_layer(rotation, self.parent_viewer)
        features = assert_layer(features, self.parent_viewer)
        _pos = pos.molecules
        _rot = rotation.molecules
        _feat = features.molecules
        mole = Molecules(_pos.pos, _rot.rotator, features=_feat.features)
        layer = self.add_molecules(
            mole, name="Mole-merged", source=pos.source_component
        )
        return self._undo_callback_for_layer(layer)

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Combine)
    def copy_molecules_features(
        self,
        source: MoleculesLayerType,
        destinations: MoleculesLayersType,
        column: Annotated[str, {"choices": _choice_getter("copy_molecules_features")}],
        alias: str = "",
    ):  # fmt: skip
        """
        Copy molecules features from one layer to another.

        This method is useful when a layer feature (such as seam search result) should be
        shared by multiple molecules layers that were aligned in a different parameters.

        Parameters
        ----------
        source : MoleculesLayer
            Layer whose features will be copied.
        destinations : MoleculesLayersType
            To which layers the features should be copied.
        column : str
            Column name of the feature to be copied.
        alias : str, optional
            If given, the copied feature will be renamed to this name.
        """
        source = assert_layer(source, self.parent_viewer)
        destinations = assert_list_of_layers(destinations, self.parent_viewer)
        series = source.molecules.features[column]
        if alias:
            series = series.alias(alias)
        for dest in destinations:
            dest.molecules = dest.molecules.with_features([series])
        return None

    @set_design(text="Split molecules by feature", location=_sw.MoleculesMenu)
    def split_molecules(
        self,
        layer: MoleculesLayerType,
        by: Annotated[str, {"choices": _choice_getter("split_molecules")}],
    ):
        """
        Split molecules by a feature column.

        Parameters
        ----------
        {layer}
        by : str
            Name of the feature to split by.
        """
        layer = assert_layer(layer, self.parent_viewer)
        utils.assert_column_exists(layer.molecules.features, by)
        _added_layers = list[MoleculesLayer]()
        for _key, mole in layer.molecules.groupby(by):
            new = self.add_molecules(
                mole, name=f"{layer.name}_{_key}", source=layer.source_component
            )
            _added_layers.append(new)
        return self._undo_callback_for_layer(_added_layers)

    @set_design(text=capitalize, location=_sw.MoleculesMenu)
    def register_molecules(
        self,
        coords: Annotated[np.ndarray, {"validator": _get_spline_coordinates}] = None,
    ):
        """Register manually added points as molecules."""
        if coords is None or coords.size == 0:
            raise ValueError("No points are given.")
        mole = Molecules(coords)
        return self.add_molecules(mole, name="Mole-manual")

    @set_design(text=capitalize, location=_sw.MoleculesMenu)
    def translate_molecules(
        self,
        layers: MoleculesLayersType,
        translation: Annotated[tuple[nm, nm, nm], {"options": {"min": -1000, "max": 1000, "step": 0.1}, "label": "translation Z, Y, X (nm)"}],
        internal: bool = True,
        inherit_source: Annotated[bool, {"label": "Inherit source spline"}] = True,
    ):  # fmt: skip
        """
        Translate molecule coordinates without changing their rotations.

        Output molecules layer will be named as "<original name>-Shift".

        Parameters
        ----------
        {layers}
        translation : tuple of float
            Translation (nm) of the molecules in (Z, Y, X) order. Whether the world
            coordinate or the internal coordinate is used depends on the `internal`
            argument.
        internal : bool, default True
            If true, the translation is applied to the internal coordinates, i.e.
            molecules with different rotations are translated differently.
        {inherit_source}
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        new_layers = list[MoleculesLayer]()
        for layer in layers:
            mole = layer.molecules
            if internal:
                out = mole.translate_internal(translation)
                if Mole.position in out.features.columns:
                    # update spline position feature
                    dy = translation[1]
                    out = out.with_features([pl.col(Mole.position) + dy])
            else:
                out = mole.translate(translation)
                if Mole.position in out.features.columns:
                    # spline position is not predictable.
                    out = out.drop_features([Mole.position])
            if inherit_source:
                source = layer.source_component
            else:
                source = None
            new = self.add_molecules(out, name=f"{layer.name}-Shift", source=source)
            new_layers.append(new)
        return self._undo_callback_for_layer(new_layers)

    @set_design(text=capitalize, location=_sw.MoleculesMenu)
    def rotate_molecules(
        self,
        layers: MoleculesLayersType,
        degrees: Annotated[
            list[tuple[Literal["z", "y", "x"], float]],
            {"layout": "vertical", "options": {"widget_type": SingleRotationEdit}},
        ],
        inherit_source: Annotated[bool, {"label": "Inherit source spline"}] = True,
    ):
        """
        Rotate molecules without changing their positions.

        Output molecules layer will be named as "<original name>-Rot".

        Parameters
        ----------
        {layers}
        degrees : list of (str, float)
            Rotation axes and degrees. For example, `[("z", 20), ("y", -10)]` means
            rotation by 20 degrees around the molecule Z axis and then by -10 degrees
            around the Y axis.
        {inherit_source}
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        new_layers = list[MoleculesLayer]()
        rotvec = degrees_to_rotator(degrees).as_rotvec()
        for layer in layers:
            mole = layer.molecules.rotate_by_rotvec_internal(rotvec)
            if inherit_source:
                source = layer.source_component
            else:
                source = None
            new = self.add_molecules(mole, name=f"{layer.name}-Rot", source=source)
            new_layers.append(new)
        return self._undo_callback_for_layer(new_layers)

    @set_design(text="Rename molecule layers", location=_sw.MoleculesMenu)
    @do_not_record(recursive=False)
    def rename_molecules(
        self,
        old: str,
        new: str,
        include: str = "",
        exclude: str = "",
        pattern: str = "",
    ):
        """
        Rename multiple molecules layers at once.

        Parameters
        ----------
        old : str
            Old string to be replaced.
        new : str
            New string to replace `old`.
        include : str, optional
            Delete layers whose names contain this string.
        exclude : str, optional
            Delete layers whose names do not contain this string.
        pattern : str, optional
            String pattern to match the layer names. Use `*` as wildcard.
        """
        if old == "":
            raise ValueError("`old` is not given.")
        if new == "":
            raise ValueError("`new` is not given.")
        return self.mole_layers.rename(
            old, new, include=include, exclude=exclude, pattern=pattern
        )

    @set_design(text="Delete molecule layers", location=_sw.MoleculesMenu)
    @do_not_record(recursive=False)
    def delete_molecules(
        self,
        include: str = "",
        exclude: str = "",
        pattern: str = "",
    ):
        """
        Delete molecules by the layer names.

        Parameters
        ----------
        include : str, optional
            Delete layers whose names contain this string.
        exclude : str, optional
            Delete layers whose names do not contain this string.
        pattern : str, optional
            String pattern to match the layer names. Use `*` as wildcard.
        """
        self.mole_layers.delete(include=include, exclude=exclude, pattern=pattern)

    @set_design(text=capitalize, location=_sw.MoleculesMenu)
    def filter_molecules(
        self,
        layer: MoleculesLayerType,
        predicate: PolarsExprStr,
        inherit_source: Annotated[bool, {"label": "Inherit source spline"}] = True,
    ):
        """
        Filter molecules by their features.

        Parameters
        ----------
        {layer}
        predicate : ExprStr
            A polars-style filter predicate, such as `pl.col("pf-id") == 3`
        {inherit_source}
        """
        layer = assert_layer(layer, self.parent_viewer)
        mole = layer.molecules
        out = mole.filter(widget_utils.norm_expr(predicate))
        if inherit_source:
            source = layer.source_component
        else:
            source = None
        new = self.add_molecules(out, name=f"{layer.name}-Filt", source=source)
        return self._undo_callback_for_layer(new)

    @set_design(text=capitalize, location=_sw.MoleculesMenu.View)
    @bind_key("Ctrl+K, C")
    def paint_molecules(
        self,
        layer: MoleculesLayerType,
        color_by: Annotated[str, {"choices": _choice_getter("paint_molecules")}],
        cmap: _CmapType = DEFAULT_COLORMAP,
        limits: Annotated[tuple[float, float], {"options": {"min": -20, "max": 20, "step": 0.01}}] = (4.00, 4.24),
    ):  # fmt: skip
        """
        Paint molecules by a feature.

        Parameters
        ----------
        {layer}{color_by}{cmap}{limits}
        """
        layer = assert_layer(layer, self.parent_viewer)
        info = layer.colormap_info
        layer.set_colormap(color_by, limits, cmap)

        match info:
            case str(color):
                return undo_callback(layer.face_color_setter).with_args(color)
            case info:
                return undo_callback(layer.set_colormap).with_args(
                    by=info.name, limits=info.clim, cmap=info.cmap
                )

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    @confirm(
        text="Column already exists. Overwrite?",
        condition="column_name in layer.molecules.features.columns",
    )
    def calculate_molecule_features(
        self,
        layer: MoleculesLayerType,
        column_name: str,
        expression: PolarsExprStr,
    ):
        """
        Calculate a new feature from the existing features.

        This method is identical to running `with_columns` on the features dataframe
        as a `polars.DataFrame`. For example,
        >>> ui.calculate_molecule_features(layer, "Y", "pl.col('X') + 1")
        is equivalent to
        >>> layer.features = layer.features.with_columns([(pl.col("X") + 1).alias("Y")])

        Parameters
        ----------
        {layer}
        column_name : str
            Name of the new column.
        expression : pl.Expr or str
            polars expression to calculate the new column.
        """
        layer = assert_layer(layer, self.parent_viewer)
        feat = layer.molecules.features
        expr = widget_utils.norm_expr(expression)
        new_feat = feat.with_columns(expr.alias(column_name))
        layer.features = new_feat
        self.reset_choices()  # choices regarding to features need update
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def interpolate_spline_properties(
        self,
        layer: MoleculesLayerType,
        interpolation: int = 3,
        suffix: str = "_spl",
    ):
        """
        Add new features by interpolating spline local properties.

        Parameters
        ----------
        {layer}{interpolation}
        suffix : str, default "_spl"
            Suffix of the new feature column names.
        """
        layer = assert_layer(layer, self.parent_viewer)
        spl = _assert_source_spline_exists(layer)
        feat = layer.molecules.features
        anc = spl.anchors
        interp = utils.interp(
            anc, spl.props.loc.to_numpy(), order=interpolation, axis=0
        )
        pos_nm = feat[Mole.position].to_numpy()
        values = interp(spl.y_to_position(pos_nm).clip(anc.min(), anc.max()))
        layer.molecules = layer.molecules.with_features(
            [
                pl.Series(f"{c}{suffix}", values[:, i])
                for i, c in enumerate(spl.props.loc.columns)
            ]
        )
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def calculate_lattice_structure(
        self,
        layer: MoleculesLayerType,
        props: Annotated[list[str], {"widget_type": CheckBoxes, "choices": cylmeasure.LatticeParameters.choices()}] = ("spacing",),
    ):  # fmt: skip
        """
        Calculate lattice structures and store the results as new feature columns.

        Parameters
        ----------
        {layer}
        props : list of str, optional
            Properties to calculate.
        """
        layer = assert_layer(layer, self.parent_viewer)
        spl = _assert_source_spline_exists(layer)
        mole = layer.molecules
        feat = mole.features

        def _calculate(p: str):
            return cylmeasure.LatticeParameters(p).calculate(mole, spl)

        layer.molecules = layer.molecules.with_features([_calculate(p) for p in props])
        self.reset_choices()  # choices regarding of features need update
        return undo_callback(layer.feature_setter(feat))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def convolve_feature(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("convolve_feature", dtype_kind="uifb")}],
        method: Literal["mean", "max", "min", "median"] = "mean",
        footprint: Annotated[Any, {"widget_type": KernelEdit}] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    ):  # fmt: skip
        """
        Run a convolution on the lattice.

        The convolution is similar to that in the context of image analysis, except for
        the cylindric boundary. During the convolution, the edges will not be considered,
        i.e., NaN value will be ignored and convolution will be the convolution of valid
        regions.

        Parameters
        ----------
        {layer}
        method : str
            Convolution method.
        {target}{footprint}
        """
        from cylindra import cylfilters

        layer = assert_layer(layer, self.parent_viewer)
        utils.assert_column_exists(layer.molecules.features, target)
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        nrise = _assert_source_spline_exists(layer).nrise()
        out = cylfilters.run_filter(
            layer.molecules.features, footprint, target, nrise, method
        )
        feature_name = f"{target}_{method}"
        layer.molecules = layer.molecules.with_features(out.alias(feature_name))
        self.reset_choices()
        match layer.colormap_info:
            case str(color):
                layer.face_color = color
            case info:
                layer.set_colormap(feature_name, info.clim, info.cmap)
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def count_neighbors(
        self,
        layer: MoleculesLayerType,
        footprint: Annotated[Any, {"widget_type": KernelEdit}] = [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        column_name: str = "neighbor_count",
    ):  # fmt: skip
        """
        Count the number of neighbors for each molecules.

        Parameters
        ----------
        {layer}{footprint}
        column_name : str
            Name of the new column that stores the number of counts.
        """
        from cylindra import cylfilters

        layer = assert_layer(layer, self.parent_viewer)
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        nrise = _assert_source_spline_exists(layer).nrise()
        out = cylfilters.count_neighbors(layer.molecules.features, footprint, nrise)
        layer.molecules = layer.molecules.with_features(out.alias(column_name))
        self.reset_choices()
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def distance_from_spline(
        self,
        layer: MoleculesLayerType,
        spline: Annotated[int, {"choices": _get_splines}],
        column_name: str = "distance",
        interval: nm = 1.0,
    ):
        """
        Add a new column that stores the shortest distance from the given spline.

        Parameters
        ----------
        {layer}{spline}
        interval: nm, default 1.0
            Sampling interval along the spline. Note that small value will increase the
            memory usage and computation time.
        """
        spl = self.tomogram.splines[spline]
        layer = assert_layer(layer, self.parent_viewer)
        if interval <= 0:
            raise ValueError("`precision` must be positive.")
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        npartitions = utils.ceilint(spl.length() / interval)
        sample_points = spl.map(np.linspace(0, 1, npartitions))
        dist = utils.distance_matrix(layer.molecules.pos, sample_points)
        dist_min = pl.Series(column_name, np.min(dist, axis=1))
        layer.molecules = layer.molecules.with_features(dist_min)
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def binarize_feature(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("binarize_feature", dtype_kind="uif")}],
        threshold: Annotated[float, {"widget_type": "FloatSlider"}] = 0.0,
        larger_true: bool = True,
        suffix: str = "_binarize",
    ):  # fmt: skip
        """
        Binarization of a layer feature by thresholding.

        Parameters
        ----------
        {layer}{target}
        threshold : float, optional
            Threshold value used for binarization.
        larger_true : bool, optional
            If true, values larger than `threshold` will be True.
        suffix : str, default "_binarize"
            Suffix of the new feature column name.
        """
        from cylindra import cylfilters

        layer = assert_layer(layer, self.parent_viewer)
        utils.assert_column_exists(layer.molecules.features, target)
        if suffix == "":
            raise ValueError("`suffix` cannot be empty.")
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        ser = cylfilters.binarize(layer.molecules.features, threshold, target)
        if not larger_true:
            ser = -ser
        feature_name = f"{target}{suffix}"
        layer.molecules = layer.molecules.with_features(
            ser.alias(feature_name).cast(pl.Boolean)
        )
        self.reset_choices()
        layer.set_colormap(feature_name, (0, 1), {0: "#A5A5A5", 1: "#FF0000"})
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @set_design(text=capitalize, location=_sw.MoleculesMenu.Features)
    def label_feature_clusters(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("label_feature_clusters", dtype_kind="b")}],
        suffix: str = "_label",
    ):  # fmt: skip
        """
        Label a binarized feature column based on the molecules structure.

        This method does the similar task as `scipy.ndimage.label`, where the isolated
        "islands" of True values will be labeled by position integers.

        Parameters
        ----------
        {layer}{target}
        suffix : str, default "_binarize"
            Suffix of the new feature column name.
        """
        from napari.utils.colormaps import label_colormap

        from cylindra import cylfilters

        layer = assert_layer(layer, self.parent_viewer)
        utils.assert_column_exists(layer.molecules.features, target)
        if suffix == "":
            raise ValueError("`suffix` cannot be empty.")
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        nrise = _assert_source_spline_exists(layer).nrise()
        out = cylfilters.label(layer.molecules.features, target, nrise)
        feature_name = f"{target}{suffix}"
        layer.molecules = layer.molecules.with_features(out.alias(feature_name))
        self.reset_choices()
        label_max = int(out.max())
        cmap = label_colormap(label_max, seed=0.9414)
        layer.set_colormap(feature_name, (0, label_max), cmap)
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @set_design(text="Analyze region properties", location=_sw.MoleculesMenu.Features)
    def regionprops_features(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("regionprops_features", dtype_kind="uif")}],
        label: Annotated[str, {"choices": _choice_getter("regionprops_features", dtype_kind="ui")}],
        properties: Annotated[list[str], {"choices": cylmeasure.RegionProfiler.CHOICES, "widget_type": CheckBoxes}] = ("area", "mean"),
    ):  # fmt: skip
        """
        Analyze region properties using another feature column as the labels.

        For instance, if the target data is [0, 1, 2, 3, 4] and the labels are [0, 1, 1, 2, 2],
        the the property "mean" will be [1.5, 3.5]. For some properties such as "length" and
        "width", the monomer connection will be considered.

        Parameters
        ----------
        {layer}{target}
        label: str
            The feature name that will be used as the labels.
        properties : list of str
            Properties to calculate.
        """
        from magicclass.ext.polars import DataFrameView

        layer = assert_layer(layer, self.parent_viewer)
        utils.assert_column_exists(
            layer.molecules.features, [target, label, Mole.nth, Mole.pf]
        )
        spl = _assert_source_spline_exists(layer)
        reg = cylmeasure.RegionProfiler.from_components(
            layer.molecules, spl, target, label
        )
        df = reg.calculate(properties)
        view = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(view, name="Region properties")
        dock.setFloating(True)
        return undo_callback(dock.close).with_redo(dock.show)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Non-GUI methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @nogui
    @do_not_record
    def add_molecules(
        self,
        molecules: Molecules,
        name: "str | None" = None,
        source: "BaseComponent | None" = None,
        metadata: "dict[str, Any]" = {},
        cmap=None,
        **kwargs,
    ) -> MoleculesLayer:
        """Add molecules as a points layer to the viewer."""
        return add_molecules(
            self.parent_viewer,
            molecules,
            name,
            source=source,
            metadata=metadata,
            cmap=cmap,
            **kwargs,
        )

    @nogui
    @do_not_record
    def get_loader(
        self,
        name: str,
        output_shape: "tuple[nm, nm, nm] | None" = None,
        order: int = 1,
    ) -> SubtomogramLoader:
        """
        Create a subtomogram loader using current tomogram and a molecules layer.

        Parameters
        ----------
        name : str, optional
            Name of the molecules layer.
        order : int, default 1
            Interpolation order of the subtomogram loader.
        """
        mole = self.mole_layers[name].molecules
        return self.tomogram.get_subtomogram_loader(mole, output_shape, order=order)

    def _init_widget_state(self, _=None):
        """Initialize widget state of spline control and local properties for new plot."""
        self.SplineControl._init_widget()
        self.LocalProperties._init_text()
        self.LocalProperties._init_plot()
        return None

    def _try_removing_layer(self, layer: Layer):
        try:
            self.parent_viewer.layers.remove(layer)
        except ValueError as e:
            _Logger.print(f"ValueError: {e}")
        return None

    def _try_removing_layers(self, layers: "Layer | list[Layer]"):
        if isinstance(layers, Layer):
            layers = [layers]
        for layer in layers:
            self._try_removing_layer(layer)

    def _add_layers_future(self, layers: "Layer | list[Layer]"):
        def future_func():
            nonlocal layers
            if isinstance(layers, Layer):
                layers = [layers]
            with self._pend_reset_choices():
                for layer in layers:
                    self.parent_viewer.add_layer(layer)
            self.reset_choices()

        return future_func

    def _undo_callback_for_layer(self, layer: "Layer | list[Layer]"):
        return (
            undo_callback(self._try_removing_layers)
            .with_args(layer)
            .with_redo(self._add_layers_future(layer))
        )

    @thread_worker.callback
    def _send_tomogram_to_viewer(
        self,
        tomo: CylTomogram,
        filt: "ImageFilter | None" = None,
        invert: bool = False,
    ):
        viewer = self.parent_viewer
        self._tomogram = tomo
        self.GeneralInfo._refer_tomogram(tomo)

        bin_size = max(x[0] for x in tomo.multiscaled)
        self._current_binsize = bin_size
        imgb = tomo.get_multiscale(bin_size)
        self._update_reference_image(imgb)

        # update viewer dimensions
        viewer.scale_bar.unit = imgb.scale_unit
        viewer.dims.axis_labels = ("z", "y", "x")
        change_viewer_focus(viewer, np.asarray(imgb.shape) / 2, imgb.scale.x)

        try:
            parts = tomo.source.parts
            if len(parts) > 2:
                _name = "…/" + Path(*parts[-2:]).as_posix()
            else:
                _name = tomo.source.as_posix()
        except Exception:
            _name = f"Tomogram<{hex(id(tomo))}>"
        _Logger.print_html(f"<h2>{_name}</h2>")

        self.macro.clear_undo_stack()
        self.Overview.layers.clear()
        self._init_widget_state()
        self._init_layers()
        self.reset_choices()

        # backward compatibility
        if isinstance(filt, bool):
            if filt:
                filt = ImageFilter.Lowpass
            else:
                filt = None
        if filt is not None and not isinstance(imgb, ip.LazyImgArray):
            self.filter_reference_image(method=filt)
        if invert:
            self.invert_image()
        self.GeneralInfo.project_desc.value = ""  # clear the project description
        self._need_save = False
        self._macro_image_load_offset = len(self.macro)

    def _update_reference_image(
        self,
        img: ip.ImgArray | ip.LazyImgArray,
        bin_size: int | None = None,
    ):
        viewer = self.parent_viewer
        if bin_size is None:
            bin_size = round(img.scale.x / self.tomogram.scale, 2)
        _is_lazy = isinstance(img, ip.LazyImgArray)
        self._reserved_layers.is_lazy = _is_lazy
        if _is_lazy:
            img = ip.zeros(img.shape, dtype=np.int8, like=img)
            img[0, [0, 0, 1, 1], [0, 1, 0, 1]] = 1
            img[1, [0, 0, 1, 1], [0, 1, 0, 1]] = 1
        tr = self.tomogram.multiscale_translation(bin_size)
        # update image layer
        if self._reserved_layers.image not in viewer.layers:
            self._reserved_layers.reset_image(img, tr)
            with self._pend_reset_choices():
                viewer.add_layer(self._reserved_layers.image)
        else:
            self._reserved_layers.update_image(img, tr)
        if self._reserved_layers.highlight in viewer.layers:
            viewer.layers.remove(self._reserved_layers.highlight)
        self._reserved_layers.image.bounding_box.visible = _is_lazy

        # update overview
        if _is_lazy:
            self.Overview.image = np.zeros((1, 1), dtype=np.float32)
        else:
            self.Overview.image = img.mean(axis="z")
        self.Overview.ylim = (0, img.shape[1])

    def _on_layer_removing(self, event: "Event"):
        # NOTE: To make recorded macro completely reproducible, removing molecules
        # from the viewer layer list must always be monitored.
        if self.parent_viewer is None:
            return  # may happen during cleanup
        layer: Layer = self.parent_viewer.layers[event.index]
        if (
            isinstance(layer, MoleculesLayer)
            and self.macro.active
            and layer.name != PREVIEW_LAYER_NAME  # ignore preview layer
        ):
            expr = mk.Mock(mk.symbol(self)).parent_viewer.layers[layer.name].expr
            undo = self._add_layers_future(layer)
            self.macro.append_with_undo(mk.Expr("del", [expr]), undo)
        return

    def _on_molecules_layer_renamed(self, event: "Event"):
        """When layer name is renamed, record `ui.parent_viewer["old"].name = "new"`"""
        layer: MoleculesLayer = event.source
        if layer._undo_renaming or not self.macro.active:
            return
        old_name = layer._old_name
        new_name = layer.name
        assert old_name is not None
        viewer_ = mk.Mock(mk.symbol(self)).parent_viewer
        expr = mk.Expr(mk.Head.assign, [viewer_.layers[old_name].name.expr, layer.name])
        return self.macro.append_with_undo(
            expr,
            undo=lambda: layer._rename(old_name),
            redo=lambda: layer._rename(new_name),
        )

    def _on_layer_inserted(self, event: "Event"):
        layer: Layer = event.value
        layer.events.name.connect(self.reset_choices)
        if isinstance(layer, MoleculesLayer):
            layer.events.name.connect(self._on_molecules_layer_renamed)

    def _disconnect_layerlist_events(self):
        viewer = self.parent_viewer
        viewer.layers.events.removing.disconnect(self._on_layer_removing)
        viewer.layers.events.inserted.disconnect(self._on_layer_inserted)

    def _init_layers(self):
        viewer = self.parent_viewer
        self._disconnect_layerlist_events()

        # remove all the molecules layers
        _layers_to_remove = list[str]()
        for layer in viewer.layers:
            if isinstance(layer, (MoleculesLayer, LandscapeSurface)):
                _layers_to_remove.append(layer.name)
            elif layer in (self._reserved_layers.prof, self._reserved_layers.work):
                _layers_to_remove.append(layer.name)

        with self._pend_reset_choices():
            for name in _layers_to_remove:
                layer: Layer = viewer.layers[name]
                viewer.layers.remove(layer)

            self._reserved_layers.init_layers()
            for layer in self._reserved_layers.to_be_removed:
                if layer in viewer.layers:
                    viewer.layers.remove(layer)
            viewer.add_layer(self._reserved_layers.prof)
            viewer.add_layer(self._reserved_layers.work)
        self.GlobalProperties._init_text()
        self.reset_choices()

        # Connect layer events.
        viewer.layers.events.removing.connect(self._on_layer_removing)
        viewer.layers.events.inserted.connect(self._on_layer_inserted)
        return None

    @contextmanager
    def _pend_reset_choices(self):
        """Temporarily disable the reset_choices method for better performance."""
        reset_choices = self.reset_choices
        self.reset_choices = lambda *_: None
        try:
            yield
        finally:
            self.reset_choices = reset_choices

    def _highlight_spline(self):
        i = self.SplineControl.num
        if i is None:
            return

        for layer in self.Overview.layers:
            if f"spline-{i}" in layer.name:
                layer.color = SplineColor.SELECTED
            else:
                layer.color = SplineColor.DEFAULT

        self._reserved_layers.highlight_spline(i)
        return None

    def _update_global_properties_in_widget(self):
        """Show global property values in widgets."""
        i = self.SplineControl.num
        if i is None:
            return
        self.GlobalProperties._set_text(self.splines[i])

    def _update_local_properties_in_widget(self, *, replot: bool = False):
        i = self.SplineControl.num
        tomo = self.tomogram
        if i is None or i >= len(tomo.splines):
            return
        j = self.SplineControl.pos
        spl = tomo.splines[i]
        if spl.props.has_loc([H.spacing, H.twist, H.npf, H.start]):
            self.LocalProperties._set_text(spl, j)
        else:
            self.LocalProperties._init_plot()
            self.LocalProperties._init_text()
        if replot:
            self.LocalProperties._plot_properties(spl)
        return None

    def _add_spline_to_images(self, spl: CylSpline, i: int):
        scale = self._reserved_layers.scale
        fit = self._reserved_layers.add_spline(i, spl)
        self.Overview.add_curve(
            fit[:, 2] / scale,
            fit[:, 1] / scale,
            color=SplineColor.DEFAULT,
            lw=2,
            name=f"spline-{i}",
            antialias=True,
        )
        self._set_orientation_marker(i)
        return None

    def _set_orientation_marker(self, idx: int):
        spl = self.tomogram.splines[idx]
        return self._reserved_layers.set_orientation(idx, spl.orientation)

    def _update_splines_in_images(self, _=None):
        """Refresh splines in overview canvas and napari canvas."""
        self.Overview.layers.clear()
        self._reserved_layers.prof.data = []
        scale = self._reserved_layers.scale
        for i, spl in enumerate(self.tomogram.splines):
            self._add_spline_to_images(spl, i)
            if spl._anchors is None:
                continue
            coords = spl.map()
            self.Overview.add_scatter(
                coords[:, 2] / scale,
                coords[:, 1] / scale,
                color=SplineColor.DEFAULT,
                symbol="x",
                lw=2,
                size=10,
                name=f"spline-{i}-anc",
            )
        self._highlight_spline()
        return None

    def _refer_spline_config(self, cfg: SplineConfig):
        """Update GUI states that are related to global variables."""
        fgui = get_function_gui(self.set_spline_props)
        fgui.npf.min, fgui.npf.max = cfg.npf_range.astuple()
        fgui.npf.value = int(cfg.npf_range.center)
        fgui.npf.value = None

        # update GUI default values
        fgui = get_function_gui(self.simulator.generate_molecules)
        fgui.spacing.value = cfg.spacing_range.center
        fgui.twist.value = cfg.twist_range.center
        fgui.npf.value = int(cfg.npf_range.center)

        for method in [self.map_monomers, self.map_monomers_with_extensions, self.map_along_pf, self.map_along_spline]:  # fmt: skip
            get_function_gui(method)["orientation"].value = cfg.clockwise


############################################################################################
#   Other helper functions
############################################################################################


def _filter_macro_for_reanalysis(macro_expr: mk.Expr, ui_sym: mk.Symbol):
    _manual_operations = {
        "open_image",
        "register_path",
        "load_splines",
        "load_molecules",
        "delete_spline",
        "add_multiscale",
        "set_multiscale",
        "spline_fitter.fit",
    }
    exprs = list[mk.Expr]()
    breaked_line: "mk.Expr | None" = None
    if len(macro_expr.args) == 0:
        raise ValueError("Macro is empty.")
    for line_id, line in enumerate(macro_expr.args):
        if line.head is not mk.Head.call:
            breaked_line = line
            break
        _fn, *_ = line.split_call()
        if _fn.head is not mk.Head.getattr:
            breaked_line = line
            break
        first, *attrs = _fn.split_getattr()
        if first != ui_sym:
            breaked_line = line
            break
        func_full_name = ".".join(map(str, attrs))
        if line_id == 0 and func_full_name != "open_image":
            raise ValueError("The first line of macro must be `ui.open_image`.")
        if func_full_name not in _manual_operations:
            breaked_line = line
            break
        exprs.append(line)
    if breaked_line is not None:
        exprs.append(mk.Expr(mk.Head.comment, [str(breaked_line) + " ... break here."]))

    return mk.Expr(mk.Head.block, exprs)


def _remove_config_kwargs(macro: mk.Macro) -> mk.Macro:
    macro_args_new = []
    for line in macro.args:
        _fn, _args, _kwargs = line.split_call()
        if "config" in _kwargs:
            _kwargs.pop("config")
        line_new = mk.Expr.unsplit_call(_fn, _args, _kwargs)
        macro_args_new.append(line_new)
    return mk.Macro(macro_args_new)


def _assert_source_spline_exists(layer: MoleculesLayer) -> "CylSpline":
    if (spl := layer.source_spline) is None:
        raise ValueError(f"Cannot find the source spline of layer {layer.name!r}.")
    return spl
