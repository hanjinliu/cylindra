from typing import Annotated, TYPE_CHECKING, Literal, Union, Any, Sequence
import warnings
from weakref import WeakSet

import impy as ip
import macrokit as mk
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from acryo import Molecules, SubtomogramLoader

from magicgui.widgets import Widget
from magicclass import (
    MagicTemplate,
    bind_key,
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
from magicclass.types import (
    Colormap as ColormapType,
    Optional,
    Path,
    ExprStr,
)
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.undo import undo_callback

from napari.layers import Layer

from cylindra import utils, _config, cylstructure
from cylindra.components import CylSpline, CylTomogram, SplineConfig
from cylindra.const import (
    PREVIEW_LAYER_NAME,
    PropertyNames as H,
    MoleculesHeader as Mole,
    Ori,
    nm,
    SplineColor,
    ImageFilter,
    FileFilter,
)
from cylindra._custom_layers import MoleculesLayer
from cylindra.project import CylindraProject, extract

from cylindra.widgets import _shared_doc, subwidgets, widget_utils
from cylindra.widgets.sta import SubtomogramAveraging

from cylindra.widgets.widget_utils import (
    add_molecules,
    change_viewer_focus,
    POLARS_NAMESPACE,
)
from cylindra.widgets._accessors import MoleculesLayerAccessor
from cylindra.widgets._widget_ext import (
    ProtofilamentEdit,
    OffsetEdit,
    CheckBoxes,
    KernelEdit,
    SingleRotationEdit,
)
from cylindra.widgets._annotated import (
    MoleculesLayerType,
    MoleculesLayersType,
    assert_layer,
    assert_list_of_layers,
)
from cylindra.widgets._main_utils import (
    SplineTracker,
    normalize_offsets,
    normalize_radius,
    rotvec_from_axis_and_degree,
)
from cylindra.widgets._reserved_layers import ReservedLayers
from cylindra.widgets import _progress_desc as _pdesc

if TYPE_CHECKING:
    from cylindra.widgets.batch import CylindraBatchWidget
    from napari.utils.events import Event
    from cylindra.components._base import BaseComponent

ICON_DIR = Path(__file__).parent / "icons"
DEFAULT_COLORMAP = {
    0.00: "#0B0000",  # black
    0.30: "#872D9D",  # purple
    0.68: "#FF0000",  # red
    1.00: "#FFFF00",  # yellow
}
TWO_WAY_COLORMAP = {
    0.00: "#000000",  # black
    0.25: "#2659FF",  # blue
    0.50: "#FFDBFE",  # white
    1.00: "#FF6C6C",  # red
}
REGIONPROPS_CHOICES = [
    "area",
    "length",
    "width",
    "sum",
    "mean",
    "median",
    "max",
    "min",
    "std",
]
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

# stylesheet
_WIDGETS_PATH = Path(__file__).parent
_STYLE = (
    _WIDGETS_PATH.joinpath("style.qss")
    .read_text()
    .replace("%QSS_PATH%", _WIDGETS_PATH.as_posix())
)
_NSPLINES = "len(self.splines) if splines is None else len(splines)"


def _choice_getter(method_name: str, dtype_kind: str = ""):
    def _get_choice(self: "CylindraMainWidget", w=None) -> list[str]:
        # don't use get_function_gui. It causes RecursionError.
        gui = self[method_name].mgui
        if gui is None or gui.layer.value is None:
            return []
        features = gui.layer.value.features
        if dtype_kind == "":
            return features.columns
        return [c for c in features.columns if features[c].dtype.kind in dtype_kind]

    _get_choice.__qualname__ = "CylindraMainWidget._get_choice"
    return _get_choice


############################################################################################
#   The Main Widget of cylindra
############################################################################################


@magicclass(widget_type="scrollable", stylesheet=_STYLE, name="cylindra")
@_shared_doc.update_cls
class CylindraMainWidget(MagicTemplate):
    # Main GUI class.

    # Weak references of active widgets. Useful for test session teardown.
    _active_widgets = WeakSet[Widget]()

    # Widget for manual spline fitting
    spline_fitter = field(subwidgets.SplineFitter, name="_Spline fitter")
    # Widget for manual spline clipping
    spline_clipper = field(subwidgets.SplineClipper, name="_Spline clipper")
    # Widget for sweeping along splines
    spline_slicer = field(subwidgets.SplineSlicer, name="_Spline slicer")
    # Widget for pre-filtering/pre-processing
    image_processor = field(subwidgets.ImageProcessor, name="_Image Processor")
    # Widget for tomogram simulator
    cylinder_simulator = field(subwidgets.CylinderSimulator, name="_Cylinder Simulator")
    # Widget for measuring FFT parameters from a 2D power spectra
    spectra_inspector = field(subwidgets.SpectraInspector, name="_SpectraInspector")
    # Widget for subtomogram analysis
    sta = field(SubtomogramAveraging, name="_Subtomogram averaging")

    mole_layers = MoleculesLayerAccessor.field()

    @property
    def batch(self) -> "CylindraBatchWidget":
        """Return the batch analyzer."""
        if self._batch is None:
            self.Analysis.open_project_batch_analyzer()
        return self._batch

    # Menu bar
    File = field(subwidgets.File)
    ImageMenu = subwidgets.Image
    Splines = subwidgets.Splines
    MoleculesMenu = subwidgets.MoleculesMenu
    Analysis = field(subwidgets.Analysis)
    Others = field(subwidgets.Others)

    # Toolbar
    toolbar = subwidgets.CylindraToolbar

    # Child widgets
    GeneralInfo = field(subwidgets.GeneralInfo)
    # Widget for controling splines
    SplineControl = subwidgets.SplineControl
    # Widget for summary of local properties
    LocalProperties = field(subwidgets.LocalPropertiesWidget, name="Local Properties")  # fmt: skip
    # Widget for summary of glocal properties
    GlobalProperties = field(subwidgets.GlobalPropertiesWidget, name="Global Properties")  # fmt: skip
    # Widget for 2D overview of splines
    overview = field(QtImageCanvas, name="Overview").with_options(tooltip="Overview of splines")  # fmt: skip

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
        self.overview.min_height = 300

        # load all the workflows
        for file in _config.WORKFLOWS_DIR.glob("*.py"):
            try:
                self.Others.Workflows.append_workflow(file)
            except Exception as e:
                _Logger.exception(f"Failed to load workflow {file.stem}: {e}")

        @self.macro.on_appended.append
        def _on_appended(expr: mk.Expr):
            self._need_save = not str(expr).startswith("ui.open_image(")

        @self.macro.on_popped.append
        def _on_popped(*_):
            self._need_save = len(self.macro) >= self._macro_offset and not str(
                self.macro[-1]
            ).startswith("ui.open_image(")

        self.default_config = SplineConfig.from_file(
            _config.get_config().default_spline_config_path
        )
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
            out = [1] + out
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

    def _get_all_spline_ids(self, splines: list[int] = None) -> list[int]:
        nspl = len(self.tomogram.splines)
        if splines is None:
            splines = list(range(nspl))
        else:
            for i in splines:
                if i >= nspl:
                    raise ValueError(f"Spline index {i} is out of range.")
        if len(splines) == 0:
            raise ValueError("No spline is selected.")
        return splines

    _Splines = Annotated[
        list[int],
        {
            "choices": _get_splines,
            "widget_type": CheckBoxes,
            "validator": _get_all_spline_ids,
        },
    ]

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "add_spline.svg")
    @bind_key("F1")
    def register_path(
        self,
        coords: Annotated[np.ndarray, {"validator": _get_spline_coordinates}] = None,
        config: Annotated[Union[dict[str, Any], SplineConfig], {"validator": _get_default_config}] = None,
        err_max: Annotated[nm, {"bind": 0.5}] = 0.5,
    ):  # fmt: skip
        """Register points as a spline path."""
        if coords.size == 0:
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

    _runner = field(subwidgets.Runner)
    _image_loader = subwidgets.ImageLoader
    _file_iterator = field(subwidgets.FileIterator)

    def _confirm_delete(self):
        i = self.SplineControl.num
        if i is None:
            # If user is writing the first spline, there's no spline registered.
            return False
        return self.tomogram.splines[i].has_props()

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "run_all.svg")
    @bind_key("F2")
    @do_not_record
    def open_runner(self):
        """Run cylindrical fitting algorithm with various settings."""
        return self._runner.show(run=False)

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "clear_last.svg")
    @confirm(text="Spline has properties. Are you sure to delete it?", condition=_confirm_delete)  # fmt: skip
    @do_not_record(recursive=False)
    def clear_current(self):
        """Clear current selection."""
        if self._reserved_layers.work.data.size > 0:
            self._reserved_layers.work.data = []
        else:
            self.delete_spline(self.SplineControl.num)
        return None

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "clear_all.svg")
    @confirm(text="Are you sure to clear all?\nYou cannot undo this.")
    @do_not_record
    def clear_all(self):
        """Clear all the splines and results."""
        self.macro.clear_undo_stack()
        self.overview.layers.clear()
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
        """Run workflow of a python file."""
        main = _config.get_main_function(filename)
        out = main(self, *args, **kwargs)
        return out

    @_image_loader.wraps
    @set_design(text="Open")
    @dask_thread_worker.with_progress(desc="Reading image")
    @confirm(text="You may have unsaved data. Open a new tomogram?", condition="self._need_save")  # fmt: skip
    def open_image(
        self,
        path: Annotated[Union[str, Path], {"bind": _image_loader.path}],
        scale: Annotated[nm, {"bind": _image_loader.scale.scale_value}] = None,
        tilt_range: Annotated[Any, {"bind": _image_loader.tilt_range.range}] = None,
        bin_size: Annotated[Sequence[int], {"bind": _image_loader.bin_size}] = [1],
        filter: Annotated[ImageFilter | None, {"bind": _image_loader.filter}] = ImageFilter.Lowpass,
        eager: Annotated[bool, {"bind": _image_loader.eager}] = False
    ):  # fmt: skip
        """
        Load an image file and process it before sending it to the viewer.

        Parameters
        ----------
        path : Path
            Path to the tomogram. Must be 3-D image.
        scale : float, default is 1.0
            Pixel size in nm/pixel unit.
        bin_size : int or list of int, default is [1]
            Initial bin size of image. Binned image will be used for visualization in the viewer.
            You can use both binned and non-binned image for analysis.
        {filter}
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
        return self._send_tomogram_to_viewer.with_args(tomo, filter)

    @open_image.started.connect
    def _open_image_on_start(self):
        return self._image_loader.close()

    @File.wraps
    @set_design(text="Load project")
    @thread_worker.with_progress(desc="Reading project", total=0)
    @confirm(text="You may have unsaved data. Open a new project?", condition="self._need_save")  # fmt: skip
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+P")
    def load_project(
        self,
        path: Path.Read[FileFilter.PROJECT],
        filter: Union[ImageFilter, None] = ImageFilter.Lowpass,
        paint: bool = False,
        read_image: Annotated[bool, {"label": "read image data"}] = True,
        update_config: bool = False,
    ):
        """
        Load a project json file.

        Parameters
        ----------
        path : path-like or CylindraProject
            Path to the project file, or the project directory that contains a project
            file, or a CylindraProject object.
        {filter}
        paint : bool, default is False
            Whether to paint cylinder properties if available.
        read_image : bool, default is True
            Whether to read image data from the project directory. If false, image data
            will be memory-mapped and will not be shown in the viewer. Unchecking this
            is useful to decrease loading time.
        update_config : bool, default is False
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
            f"filter={str(filter)!r}, {paint=}, {read_image=}, {update_config=})</code>"
        )
        if project_path is not None:
            _Logger.print(f"Project loaded: {project_path.as_posix()}")
            self._project_dir = project_path
        yield from project._to_gui(
            self,
            filter=filter,
            paint=paint,
            read_image=read_image,
            update_config=update_config,
        )

    @File.wraps
    @set_design(text="Save project")
    @do_not_record
    @bind_key("Ctrl+K, Ctrl+S")
    def save_project(
        self,
        save_path: Path.Save,
        molecules_ext: Literal[".csv", ".parquet"] = ".csv",
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
        save_path : Path
            Path of json file.
        molecules_ext : str, default is ".csv"
            Extension of the molecule file. Can be ".csv" or ".parquet".
        """
        save_path = Path(save_path)
        dir_posix = save_path.as_posix()
        CylindraProject.save_gui(self, save_path, molecules_ext)
        _Logger.print(f"Project saved: {dir_posix}")
        self._need_save = False
        self._project_dir = save_path
        return None

    @File.wraps
    @set_design(text="Overwrite project")
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

    @File.wraps
    @set_design(text="Load splines")
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

    @File.wraps
    @set_design(text="Load molecules")
    def load_molecules(self, paths: Path.Multiple[FileFilter.CSV]):
        """Load molecules from a csv file."""
        if isinstance(paths, (str, Path, bytes)):
            paths = [paths]
        for path in paths:
            mole = Molecules.from_file(path)
            name = Path(path).stem
            add_molecules(self.parent_viewer, mole, name)
        return None

    @File.wraps
    @do_not_record
    @set_design(text="Save spline")
    def save_spline(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        save_path: Path.Save[FileFilter.JSON],
    ):
        """Save splines as a json file."""
        spl = self.tomogram.splines[spline]
        spl.to_json(save_path)
        return None

    @File.wraps
    @do_not_record
    @set_design(text="Save molecules")
    def save_molecules(
        self, layer: MoleculesLayer, save_path: Path.Save[FileFilter.CSV]
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

    @ImageMenu.wraps
    @set_design(text="Filter reference image")
    @dask_thread_worker.with_progress(desc=_pdesc.filter_image_fmt)
    @do_not_record
    def filter_reference_image(
        self,
        method: ImageFilter = ImageFilter.Lowpass,
    ):  # fmt: skip
        """Apply filter to enhance contrast of the reference image."""
        method = ImageFilter(method)
        with utils.set_gpu():
            img = self._reserved_layers.image_data
            overlap = [min(s, 32) for s in img.shape]
            _tiled = img.tiled(chunks=(224, 224, 224), overlap=overlap)
            sigma = 1.6 / self._reserved_layers.scale
            if method is ImageFilter.Lowpass:
                img_filt = _tiled.lowpass_filter(cutoff=0.2)
            elif method is ImageFilter.Gaussian:
                img_filt = _tiled.gaussian_filter(sigma=sigma, fourier=True)
            elif method is ImageFilter.DoG:
                img_filt = _tiled.dog_filter(low_sigma=sigma, fourier=True)
            elif method is ImageFilter.LoG:
                img_filt = _tiled.log_filter(sigma=sigma)
            else:
                raise ValueError(f"No method matches {method!r}")

        contrast_limits = np.percentile(img_filt, [1, 99.9])

        @thread_worker.callback
        def _filter_reference_image_on_return():
            self._reserved_layers.image.data = img_filt
            self._reserved_layers.image.contrast_limits = contrast_limits
            proj = self._reserved_layers.image.data.proj("z")
            self.overview.image = proj
            self.overview.contrast_limits = contrast_limits

        return _filter_reference_image_on_return

    @ImageMenu.wraps
    @set_design(text="Add multi-scale")
    @dask_thread_worker.with_progress(desc=lambda bin_size: f"Adding multiscale (bin = {bin_size})")  # fmt: skip
    def add_multiscale(
        self,
        bin_size: Annotated[int, {"choices": list(range(2, 17))}] = 4,
    ):
        """
        Add a new multi-scale image of current tomogram.

        Parameters
        ----------
        bin_size : int, default is 4
            Bin size of the new image
        """
        tomo = self.tomogram
        tomo.get_multiscale(binsize=bin_size, add=True)
        return thread_worker.callback(self.set_multiscale).with_args(bin_size)

    @ImageMenu.wraps
    @set_design(text="Set multi-scale")
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
        self._reserved_layers.update_image(
            imgb, bin_size, tomo.multiscale_translation(bin_size)
        )
        current_z = self.parent_viewer.dims.current_step[0]
        self.parent_viewer.dims.set_current_step(axis=0, value=current_z * factor)

        # update overview
        self.overview.image = imgb.proj("z")
        self.overview.xlim = [x * factor for x in self.overview.xlim]
        self.overview.ylim = [y * factor for y in self.overview.ylim]
        self._current_binsize = bin_size
        self.reset_choices()
        return undo_callback(self.set_multiscale).with_args(_old_bin_size)

    @ImageMenu.wraps
    @set_design(text="Sample subtomograms")
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

    @Splines.wraps
    @set_design(text="Show splines as curves")
    def show_splines(self):
        """Show 3D spline paths of cylinder central axes as a layer."""
        paths = [r.partition(100) for r in self.tomogram.splines]
        layer = self.parent_viewer.add_shapes(
            paths,
            shape_type="path",
            name="Spline Curves",
            edge_color="lime",
            edge_width=1,
        )
        return self._undo_callback_for_layer(layer)

    @Splines.wraps
    @set_design(text="Show splines as meshes")
    def show_splines_as_meshes(self):
        """Show 3D spline cylinder as a surface layer."""
        nodes = []
        vertices = []
        n_nodes = 0
        for i, spl in enumerate(self.tomogram.splines):
            n, v = spl.cylinder_model().to_mesh(spl)
            nodes.append(n)
            vertices.append(v + i * n_nodes)
            n_nodes += n.shape[0]
        nodes = np.concatenate(nodes, axis=0)
        vertices = np.concatenate(vertices, axis=0)
        layer = self.parent_viewer.add_surface([nodes, vertices], shading="smooth")
        # NOTE: re-adding surface layer is not redoable, since viewer.add_layer seems
        # broken for the surface layer.
        return undo_callback(self._try_removing_layers, redo=False).with_args(layer)

    @Splines.Orientation.wraps
    @set_design(text="Invert spline")
    def invert_spline(self, spline: Annotated[int, {"bind": SplineControl.num}] = None):
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

    @Splines.Orientation.wraps
    @set_design(text="Align to polarity")
    def align_to_polarity(
        self, orientation: Literal["MinusToPlus", "PlusToMinus"] = "MinusToPlus"
    ):
        """
        Align all the splines in the direction parallel to the cylinder polarity.

        Parameters
        ----------
        orientation : Ori, default is Ori.MinusToPlus
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

    @Splines.Orientation.wraps
    @set_design(text="Auto-align to polarity")
    @thread_worker.with_progress(desc="Auto-detecting polarities...", total=_NSPLINES)
    def auto_align_to_polarity(
        self,
        splines: _Splines = None,
        align_to: Annotated[Optional[Literal["MinusToPlus", "PlusToMinus"]], {"text": "Do not align"}] = None,
        depth: Annotated[nm, {"min": 5.0, "max": 500.0, "step": 5.0}] = 40,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Automatically detect the polarities and align if necessary.

        This function uses Fourier vorticity to detect the polarities of the splines.
        The subtomogram at the center of the spline will be sampled in the cylindric
        coordinate and the power spectra in (radius, angle) space will be calculated.
        The peak position of the `angle = nPF` line scan will be used to detect the
        polarity of the spline.

        Parameters
        ----------
        align_to : Ori, optional
            To which direction splines will be aligned. If not given, splines will
            not be inverted even if the orientation is not aligned.
        {depth}{bin_size}
        """
        tomo = self.tomogram
        _old_orientations = [spl.orientation for spl in self.tomogram.splines]
        for i in splines:
            tomo.infer_polarity(i=i, binsize=bin_size, depth=depth, update=True)
            yield
        _new_orientations = [spl.orientation for spl in self.tomogram.splines]

        if align_to is not None:
            return thread_worker.callback(self.align_to_polarity).with_args(align_to)
        else:

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
        for spl, ori in zip(self.tomogram.splines, orientations):
            spl.orientation = ori
        self._update_splines_in_images()
        self._init_widget_state()
        self.reset_choices()
        for i in range(len(self.tomogram.splines)):
            self._set_orientation_marker(i)
        if resample:
            self.sample_subtomograms()
        return None

    @Splines.wraps
    @set_design(text="Clip spline")
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
        lengths : tuple of float, default is (0., 0.)
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

    @Splines.wraps
    @set_design(text="Delete spline")
    @confirm(
        text="Spline has properties. Are you sure to delete it?",
        condition=_confirm_delete,
    )
    def delete_spline(self, i: Annotated[int, {"bind": SplineControl.num}]):
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

    @Splines.wraps
    @set_design(text="Copy spline")
    def copy_spline(self, i: Annotated[int, {"bind": SplineControl.num}]):
        """Make a copy of the current spline"""
        spl = self.tomogram.splines[i]
        self.tomogram.splines.append(spl.copy())
        self.reset_choices()
        self.SplineControl.num = len(self.tomogram.splines) - 1
        return undo_callback(self.delete_spline).with_args(-1)

    @Splines.wraps
    @set_design(text="Copy spline (new config)")
    def copy_spline_new_config(
        self,
        i: Annotated[int, {"bind": SplineControl.num}],
        npf_range: Annotated[tuple[int, int], {"options": {"min": 2, "max": 100}}] = (11, 17),
        spacing_range: Annotated[tuple[nm, nm], {"options": {"step": 0.05}}] = (3.9, 4.3),
        dimer_twist_range: Annotated[tuple[float, float], {"options": {"min": -45.0, "max": 45.0, "step": 0.05}}] = (-1.0, 1.0),
        rise_range: Annotated[tuple[float, float], {"options": {"min": -45.0, "max": 45.0, "step": 0.1}}] = (0.0, 45.0),
        rise_sign: Literal[-1, 1] = -1,
        clockwise: Literal["PlusToMinus", "MinusToPlus"] = "MinusToPlus",
        thickness_inner: Annotated[nm, {"min": 0.0, "step": 0.1}] = 2.8,
        thickness_outer: Annotated[nm, {"min": 0.0, "step": 0.1}] = 2.8,
        fit_depth: Annotated[nm, {"min": 4.0, "step": 1}] = 49.0,
        fit_width: Annotated[nm, {"min": 4.0, "step": 1}] = 44.0,
    ):  # fmt: skip
        """Make a copy of the current spline with a new configuration."""
        config = locals()
        del config["i"], config["self"]
        spl = self.tomogram.splines[i]
        self.tomogram.splines.append(spl.with_config(config))
        self.reset_choices()
        self.SplineControl.num = len(self.tomogram.splines) - 1
        return undo_callback(self.delete_spline).with_args(-1)

    @Splines.wraps
    @set_design(text="Fit splines")
    @thread_worker.with_progress(desc="Spline Fitting", total=_NSPLINES)
    def fit_splines(
        self,
        splines: _Splines = None,
        max_interval: Annotated[nm, {"label": "Max interval (nm)"}] = 30,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1.0,
        err_max: Annotated[nm, {"label": "Max fit error (nm)", "step": 0.1}] = 1.0,
        degree_precision: float = 0.5,
        edge_sigma: Annotated[Optional[nm], {"text": "Do not mask image"}] = 2.0,
        max_shift: nm = 5.0,
    ):  # fmt: skip
        """
        Fit cylinder with spline curve, using manually selected points.

        Parameters
        ----------
        {splines}{max_interval}{bin_size}{err_max}
        degree_precision : float, default is 0.5
            Precision of xy-tilt degree in angular correlation.
        edge_sigma : bool, default is False
            Check if cylindric structures are densely packed. Initial spline position must
            be "almost" fitted in dense mode.
        max_shift : nm, default is 5.0
            Maximum shift to be applied to each point of splines.
        """
        tomo = self.tomogram
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

    @Splines.wraps
    @set_design(text="Add anchors")
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
        how : str, default is "pack"
            How to add anchors.
            - "pack": (x---x---x-) Pack anchors from the starting point of splines.
            - "equal": (x--x--x--x) Equally distribute anchors between the starting point
              and the end point of splines. Actual intervals will be smaller.
        """
        tomo = self.tomogram
        with SplineTracker(widget=self, indices=splines) as tracker:
            if how == "pack":
                tomo.make_anchors(splines, interval=interval)
            elif how == "equal":
                tomo.make_anchors(splines, max_interval=interval)
            else:
                raise ValueError(f"Unknown method: {how}")
        self._update_splines_in_images()
        return tracker.as_undo_callback()

    @Splines.wraps
    @set_design(text="Refine splines")
    @thread_worker.with_progress(desc="Refining splines", total=_NSPLINES)
    def refine_splines(
        self,
        splines: _Splines = None,
        max_interval: Annotated[nm, {"label": "Maximum interval (nm)"}] = 30,
        err_max: Annotated[nm, {"label": "Max fit error (nm)", "step": 0.1}] = 0.8,
        corr_allowed: Annotated[float, {"label": "Correlation allowed", "max": 1.0, "step": 0.1}] = 0.9,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Refine splines using the global cylindric structural parameters.

        Parameters
        ----------
        {splines}{max_interval}{err_max}
        corr_allowed : float, defaul is 0.9
            How many images will be used to make template for alignment. If 0.9, then top 90%
            will be used.
        {bin_size}
        """
        tomo = self.tomogram
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

    @Splines.wraps
    @set_design(text="Set spline parameters")
    def set_spline_props(
        self,
        spline: Annotated[int, {"bind": SplineControl.num}],
        npf: Annotated[Optional[int], {"label": "number of PF", "text": "Do not update"}] = None,
        orientation: Annotated[Optional[Literal["MinusToPlus", "PlusToMinus"]], {"text": "Do not update"}] = None,
    ):  # fmt: skip
        """
        Set spline global properties.

        This method will overwrite spline properties with the user input. You should
        not call this method unless there's a good reason to do so, e.g. the number
        of protofilaments is obviously wrong.

        Parameters
        ----------
        npf : int , optional
        orientation : str, optional
            If given, update the spline orientation.
        """
        spl = self.tomogram.splines[spline]
        old_spl = spl.copy()
        spl.update_props(npf=npf, orientation=orientation)
        self.sample_subtomograms()
        self._update_splines_in_images()

        @undo_callback
        def out():
            self.tomogram.splines[spline] = old_spl
            self.sample_subtomograms()
            self._update_splines_in_images()

        return out

    @Splines.wraps
    @set_design(text="Molecules to spline")
    def molecules_to_spline(
        self,
        layers: MoleculesLayersType = (),
        err_max: Annotated[nm, {"label": "Max fit error (nm)", "step": 0.1}] = 0.8,
        delete_old: Annotated[bool, {"label": "Delete old splines"}] = True,
        inherit_props: Annotated[bool, {"label": "Inherit properties from old splines"}] = True,
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
        delete_old : bool, default is True
            If True, delete the old spline if the molecules has one. For instance, if
            "Mole-0" has the spline "Spline-0" as the source, and a spline "Spline-1" is
            created from "Mole-0", then "Spline-0" will be deleted from the list.
        inherit_props : bool, default is True
            If True, copy the global properties from the old spline to the new one.
        missing_ok : bool, default is False
            If False, raise an error if the source spline is not found in the tomogram.
        update_sources : bool, default is True
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
            mole = layer.molecules
            if _s := layer.source_spline:
                _config = _s.config
            else:
                _config = self.default_config
            spl = utils.molecules_to_spline(mole, _config, err_max=err_max)
            try:
                idx = tomo.splines.index(layer.source_spline)
            except ValueError:
                tomo.splines.append(spl)
            else:
                if inherit_props:
                    spl.globalprops = tomo.splines[idx].globalprops.clone()
                old_spl = tomo.splines[idx]
                # Must be updated here, otherwise each.source_component may return
                # None since GC may delete the old spline.
                if update_sources:
                    for each in self.parent_viewer.layers:
                        if not isinstance(each, MoleculesLayer):
                            continue
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

    @Analysis.wraps
    @set_design(text="Measure radius")
    @thread_worker.with_progress(desc="Measuring Radius", total=_NSPLINES)
    def measure_radius(
        self,
        splines: _Splines = None,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        min_radius: Annotated[nm, {"min": 0.1, "step": 0.1}] = 1.0,
    ):  # fmt: skip
        """
        Measure cylinder radius for each spline curve.

        Parameters
        ----------
        {splines}{bin_size}{min_radius}
        """
        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                self.tomogram.measure_radius(i, binsize=bin_size, min_radius=min_radius)
                yield

        return tracker.as_undo_callback()

    @Analysis.wraps
    @set_design(text="Set radius")
    def set_radius(
        self,
        splines: _Splines = None,
        radius: ExprStr.In[POLARS_NAMESPACE] = 10.0,
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
        if isinstance(radius, pl.Expr):
            radius_expr: "pl.Expr | float" = radius
        elif isinstance(radius, str):
            radius_expr = ExprStr(radius, POLARS_NAMESPACE).eval()
            if not isinstance(radius_expr, pl.Expr):
                radius_expr = float(radius_expr)
        else:
            radius_expr = float(radius)
        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                spl = self.tomogram.splines[i]
                if isinstance(radius_expr, pl.Expr):
                    spl.radius = spl.props.glob.select(radius_expr).to_numpy()[0, 0]
                else:
                    spl.radius = radius_expr
        return tracker.as_undo_callback()

    @Analysis.wraps
    @set_design(text="Measure local radius")
    @thread_worker.with_progress(desc="Measuring local radii", total=_NSPLINES)
    def measure_local_radius(
        self,
        splines: _Splines = None,
        interval: _Interval = None,
        depth: Annotated[nm, {"min": 2.0, "step": 0.5}] = 50.0,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        min_radius: Annotated[nm, {"min": 0.1, "step": 0.1}] = 1.0,
        update_glob: Annotated[bool, {"text": "Also update the global radius"}] = True,
    ):  # fmt: skip
        """
        Measure radius for each local region along splines.

        Parameters
        ----------
        {splines}{interval}{depth}{bin_size}{min_radius}{update_glob}
        """
        tomo = self.tomogram

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
                    update_glob=update_glob,
                )
                if i == splines[-1]:
                    yield _on_yield
                else:
                    yield

        return tracker.as_undo_callback()

    @Analysis.wraps
    @set_design(text="Measure radius by molecules")
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
            _radius_df.append(df.with_columns(cylstructure.calc_radius(mole, spl)))

        if _duplicated:
            _layer_names = ", ".join(repr(l.name) for l in layers)
            raise ValueError(f"Layers {_layer_names} have duplicated spline sources.")

        indices = [self.tomogram.splines.index(spl) for spl in _splines]
        with SplineTracker(widget=self, indices=indices) as tracker:
            for i, spl, df in zip(indices, _splines, _radius_df):
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
                spl.props.update_loc([radii], depth)
                if update_glob:
                    spl.radius = df[Mole.radius].mean()
        self._update_local_properties_in_widget(replot=True)
        return tracker.as_undo_callback()

    @Analysis.wraps
    @set_design(text="Local FT analysis")
    @thread_worker.with_progress(desc="Local Fourier transform", total=_NSPLINES)
    def local_ft_analysis(
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
        radius : str, default is "global"
            If "local", use the local radius for the analysis. If "global", use the
            global radius.
        {update_glob}
        """
        tomo = self.tomogram

        @thread_worker.callback
        def _local_ft_analysis_on_yield(i: int):
            self._update_splines_in_images()
            if i == self.SplineControl.num:
                self.sample_subtomograms()

        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                if interval is not None:
                    if radius == "local":
                        raise ValueError(
                            "With `interval`, local radius values will be dropped. Please set "
                            "`radius='global'` or `interval=None`."
                        )
                    tomo.make_anchors(i=i, interval=interval)
                tomo.local_ft_params(
                    i=i,
                    depth=depth,
                    binsize=bin_size,
                    radius=radius,
                    update_glob=update_glob,
                )
                yield _local_ft_analysis_on_yield.with_args(i)
        return tracker.as_undo_callback()

    @Analysis.wraps
    @set_design(text="Global FT analysis")
    @thread_worker.with_progress(desc="Global Fourier transform", total=_NSPLINES)
    def global_ft_analysis(
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

        with SplineTracker(widget=self, indices=splines, sample=True) as tracker:
            for i in splines:
                spl = tomo.splines[i]
                if spl.radius is None:
                    tomo.measure_radius(i=i)
                tomo.global_ft_params(i=i, binsize=bin_size)
                yield

        # show all in a table
        df = (
            self.tomogram.splines.collect_globalprops()
            .drop(H.spline_id)
            .to_pandas()
            .transpose()
        )
        df.columns = [f"Spline-{i}" for i in range(len(df.columns))]

        @thread_worker.callback
        def _global_ft_analysis_on_return():
            self.sample_subtomograms()
            _Logger.print_table(df, precision=3)
            self._update_global_properties_in_widget()

            return tracker.as_undo_callback()

        return _global_ft_analysis_on_return

    def _get_reanalysis_macro(self, path: Path):
        """Get the macro expression for reanalysis in the given project path."""
        _ui_sym = mk.symbol(self)
        project = CylindraProject.from_file(path)
        with project.open_project() as dir:
            macro_path = dir / "script.py"
            macro_expr = extract(macro_path.read_text())
        return _filter_macro_for_reanalysis(macro_expr, _ui_sym)

    @Analysis.wraps
    @set_design(text="Re-analyze current tomogram")
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
        self.macro.clear_undo_stack()
        return None

    @Analysis.wraps
    @set_design(text="Re-analyze project")
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
        self.macro.clear_undo_stack()
        return None

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Monomer mapping methods
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map monomers")
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
        extensions : (int, int), default is (0, 0)
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

        for i in splines:
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

    def _offsets_validator(
        self, offsets: _OffsetType, args: dict[str, Any]
    ) -> tuple[nm, float]:
        idx = args["spline"]
        spl = self.tomogram.splines[idx]
        return normalize_offsets(offsets, spl)

    _NormalizedOffsetType = Annotated[
        Optional[tuple[nm, float]],
        {
            "text": "Infer offsets from spline global properties",
            "options": {"widget_type": OffsetEdit},
            "validator": _offsets_validator,
        },
    ]

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map monomers with extensions")
    def map_monomers_with_extensions(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        n_extend: Annotated[dict[int, tuple[int, int]], {"label": "prepend/append", "widget_type": ProtofilamentEdit}] = {},
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
        offsets: _NormalizedOffsetType = None,
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
            offsets=offsets,
            radius=normalize_radius(radius, spl),
        )
        layer = self.add_molecules(mole, f"{prefix}-{spline}", source=spl)
        return self._undo_callback_for_layer(layer)

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map centers")
    def map_centers(
        self,
        splines: _Splines = None,
        molecule_interval: Annotated[Optional[nm], {"text": "Set to dimer length"}] = None,
        orientation: Literal[None, "PlusToMinus", "MinusToPlus"] = None,
        prefix: str = "Center",
    ):  # fmt: skip
        """
        Map molecules along splines. Each molecule is rotated by skewing.

        Parameters
        ----------
        {splines}{molecule_interval}{orientation}{prefix}
        """
        tomo = self.tomogram
        if len(splines) == 0 and len(tomo.splines) > 0:
            splines = tuple(range(len(tomo.splines)))
        mols = tomo.map_centers(
            i=splines, interval=molecule_interval, orientation=orientation
        )
        _Logger.print_html("<code>map_centers</code>")
        _added_layers = []
        for i, mol in enumerate(mols):
            _name = f"{prefix}-{i}"
            layer = self.add_molecules(mol, _name, source=tomo.splines[splines[i]])
            _added_layers.append(layer)
            _Logger.print(f"{_name!r}: n = {len(mol)}")
        return self._undo_callback_for_layer(_added_layers)

    @MoleculesMenu.Mapping.wraps
    @set_design(text="Map alogn PF")
    def map_along_pf(
        self,
        spline: Annotated[int, {"choices": _get_splines}],
        molecule_interval: Annotated[Optional[nm], {"text": "Set to dimer length"}] = None,
        offsets: _NormalizedOffsetType = None,
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
        _Logger.print_html("<code>map_along_PF</code>")
        mol = tomo.map_pf_line(
            i=spline,
            interval=molecule_interval,
            offsets=normalize_offsets(offsets, tomo.splines[spline]),
            orientation=orientation,
        )
        _name = f"{prefix}-{spline}"
        layer = self.add_molecules(mol, _name, source=tomo.splines[spline])
        _Logger.print(f"{_name!r}: n = {len(mol)}")
        return self._undo_callback_for_layer(layer)

    @MoleculesMenu.wraps
    @set_design(text="Set source spline")
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

    @MoleculesMenu.Combine.wraps
    @set_design(text="Concatenate molecules")
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

    @MoleculesMenu.Combine.wraps
    @set_design(text="Merge molecule info")
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

    @MoleculesMenu.wraps
    @set_design(text="Split molecules by feature")
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
        n_unique = layer.molecules.features[by].n_unique()
        if n_unique > 48:
            raise ValueError(f"Too many groups ({n_unique}).")
        _added_layers = list[MoleculesLayer]()
        for _key, mole in layer.molecules.groupby(by):
            new = self.add_molecules(
                mole, name=f"{layer.name}_{_key}", source=layer.source_component
            )
            _added_layers.append(new)
        return self._undo_callback_for_layer(_added_layers)

    @MoleculesMenu.wraps
    @set_design(text="Translate molecules")
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
            coordinate or the internal coordinate is used depends on the ``internal``
            argument.
        internal : bool, default is True
            If true, the translation is applied to the internal coordinates, i.e. molecules
            with different rotations are translated differently.
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

    @MoleculesMenu.wraps
    @set_design(text="Rotate molecules")
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
            Rotation axes and degrees. For example, ``[("z", 20), ("y", -10)]`` means rotation
            by 20 degrees around the molecule Z axis and then by -10 degrees around the Y axis.
        {inherit_source}
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        new_layers = list[MoleculesLayer]()
        if (
            len(degrees) == 2
            and isinstance(degrees[0], str)
            and isinstance(degrees[1], float)
        ):
            degrees = [degrees]
        for layer in layers:
            mole = layer.molecules
            for axis, deg in degrees:
                mole = mole.rotate_by_rotvec_internal(
                    rotvec_from_axis_and_degree(axis, deg)
                )
            if inherit_source:
                source = layer.source_component
            else:
                source = None
            new = self.add_molecules(mole, name=f"{layer.name}-Rot", source=source)
            new_layers.append(new)
        return self._undo_callback_for_layer(new_layers)

    @MoleculesMenu.wraps
    @set_design(text="Rename molecule layers")
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

    @MoleculesMenu.wraps
    @set_design(text="Delete molecule layers")
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
        return self.mole_layers.delete(
            include=include, exclude=exclude, pattern=pattern
        )

    @MoleculesMenu.wraps
    @set_design(text="Filter molecules")
    def filter_molecules(
        self,
        layer: MoleculesLayerType,
        predicate: ExprStr.In[POLARS_NAMESPACE],
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
        if isinstance(predicate, pl.Expr):
            expr = predicate
        else:
            expr = ExprStr(predicate, POLARS_NAMESPACE).eval()
        out = mole.filter(expr)
        if inherit_source:
            source = layer.source_component
        else:
            source = None
        new = self.add_molecules(out, name=f"{layer.name}-Filt", source=source)

        return self._undo_callback_for_layer(new)

    @MoleculesMenu.View.wraps
    @set_design(text="Paint molecules by features")
    @bind_key("Ctrl+K, C")
    def paint_molecules(
        self,
        layer: MoleculesLayerType,
        color_by: Annotated[str, {"choices": _choice_getter("paint_molecules")}],
        cmap: ColormapType = DEFAULT_COLORMAP,
        limits: Annotated[tuple[float, float], {"options": {"min": -20, "max": 20, "step": 0.01}}] = (4.00, 4.24),
    ):  # fmt: skip
        """
        Paint molecules by a feature.

        Parameters
        ----------
        {layer}{color_by}{cmap}{limits}
        """
        layer = assert_layer(layer, self.parent_viewer)
        layer.set_colormap(color_by, limits, cmap)
        info = layer.colormap_info
        return undo_callback(layer.set_colormap).with_args(
            name=info.name, clim=info.clim, cmap_input=info.cmap
        )

    @MoleculesMenu.View.wraps
    @set_design(text="Plot molecule feature in 2D")
    def plot_molecule_feature(
        self,
        layer: MoleculesLayerType,
        backend: Literal["inline", "qt"] = "inline",
    ):
        """
        Plot current molecule feature coloring in 2D figure.

        For data visualization, plotting in 2D is better than in 3D. Current
        colormap in the 3D canvas is directly used for 2D plotting.

        Parameters
        ----------
        {layer}
        backend : "inline" or "qt", optional
            Plotting backend. "inline" means the plot is shown in the console.
        """
        from matplotlib.patches import Circle
        from matplotlib.axes import Axes

        layer = assert_layer(layer, self.parent_viewer)
        mole = layer.molecules
        nth = mole.features[Mole.nth].to_numpy()
        pf = mole.features[Mole.pf].to_numpy()
        npf = int(pf.max() + 1)
        if spl := layer.source_spline:
            spacing = spl.props.get_glob(H.spacing)
            rise = np.deg2rad(spl.props.get_glob(H.rise))
            tan = (
                np.tan(rise)
                / spacing
                * (2 * np.pi * spl.radius / npf)
                * spl.config.rise_sign
            )
        else:
            tan = utils.infer_start_from_molecules(mole) / npf
        y = nth + tan * pf

        face_color = layer.face_color
        if backend == "inline":
            plt.figure()
            ax: Axes = plt.gca()
        elif backend == "qt":
            from magicclass.widgets import Figure

            fig = Figure()
            ax = fig.ax
            fig.show()
            self._active_widgets.add(fig)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        for i in range(mole.count()):
            center = (pf[i], y[i])
            circ = Circle(center, 0.5, fc=face_color[i], ec="black", lw=0.1)
            ax.add_patch(circ)
        ax.set_title(layer.name)
        ax.set_xlim(pf.min() - 0.6, pf.max() + 0.6)
        if backend == "inline":
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        elif backend == "qt":
            ax.set_ylim(y.mean() - pf.mean() - 0.6, y.mean() + pf.mean() + 0.6)
        ax.set_aspect("equal")
        return undo_callback(lambda: _Logger.print("Undoing plotting does nothing"))

    @MoleculesMenu.MoleculeFeatures.wraps
    @confirm(
        text="Column already exists. Overwrite?",
        condition="column_name in layer.molecules.features.columns",
    )
    @set_design(text="Calculate molecule features")
    def calculate_molecule_features(
        self,
        layer: MoleculesLayerType,
        column_name: str,
        expression: ExprStr.In[POLARS_NAMESPACE],
    ):
        """
        Calculate a new feature from the existing features.

        This method is identical to running ``with_columns`` on the features dataframe
        as a ``polars.DataFrame``. For example,

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
        pl_expr = eval(str(expression), POLARS_NAMESPACE, {})
        if isinstance(pl_expr, pl.Expr):
            new_feat = feat.with_columns(pl_expr.alias(column_name))
        else:
            new_feat = feat.with_columns(pl.Series(column_name, pl_expr))
        layer.features = new_feat
        self.reset_choices()  # choices regarding to features need update
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Interpolate spline properties")
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
        suffix : str, default is "_spl"
            Suffix of the new feature column names.
        """
        from scipy.interpolate import interp1d

        layer = assert_layer(layer, self.parent_viewer)
        if interpolation == 0:
            kind = "nearest"
        elif interpolation == 1:
            kind = "linear"
        elif interpolation == 3:
            kind = "cubic"
        else:
            raise ValueError(f"`interpolation` must be 0, 1 or 3. Got {interpolation}.")

        spl = _assert_source_spline_exists(layer)
        feat = layer.molecules.features
        anc = spl.anchors
        interp = interp1d(anc, spl.props.loc.to_numpy(), kind=kind, axis=0)
        pos_nm = feat[Mole.position].to_numpy()
        values = interp(spl.y_to_position(pos_nm).clip(anc.min(), anc.max()))
        layer.molecules = layer.molecules.with_features(
            [
                pl.Series(f"{c}{suffix}", values[:, i])
                for i, c in enumerate(spl.props.loc.columns)
            ]
        )
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Calculate lattice structure")
    def calculate_lattice_structure(
        self,
        layer: MoleculesLayerType,
        props: Annotated[list[str], {"widget_type": CheckBoxes, "choices": cylstructure.LatticeParameters.choices()}] = ("interv",),
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
            return cylstructure.LatticeParameters(p).calculate(mole, spl)

        layer.molecules = layer.molecules.with_features([_calculate(p) for p in props])
        self.reset_choices()  # choices regarding of features need update
        return undo_callback(layer.feature_setter(feat))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Calculate curve index")
    def calculate_curve_index(self, layer: MoleculesLayerType):
        """
        Calculate the curve index for each molecule.

        The "curve index" is 1 if the molecule is inside the curve. Outside will
        be -1, and a straight line will be 0.

        Parameters
        ----------
        {layer}
        """
        layer = assert_layer(layer, self.parent_viewer)
        spl = _assert_source_spline_exists(layer)
        mole = layer.molecules
        feat = mole.features
        new_feat_col = cylstructure.calc_curve_index(mole, spl)
        layer.molecules = layer.molecules.with_features(new_feat_col)
        self.reset_choices()
        return undo_callback(layer.feature_setter(feat))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Convolve feature")
    def convolve_feature(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("convolve_feature", dtype_kind="uifb")}],
        method: Literal["mean", "max", "min", "median"],
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
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        nrise = _assert_source_spline_exists(layer).nrise()
        out = cylfilters.run_filter(
            layer.molecules.features, footprint, target, nrise, method
        )
        feature_name = f"{target}_{method}"
        layer.molecules = layer.molecules.with_features(out.alias(feature_name))
        self.reset_choices()
        if layer.colormap_info is not None:
            layer.set_colormap(
                feature_name, layer.colormap_info.clim, layer.colormap_info.cmap
            )
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Binarize feature by thresholding")
    def binarize_feature(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("binarize_feature", dtype_kind="uif")}],
        threshold: Annotated[float, {"widget_type": "FloatSlider"}] = 0.0,
        larger_true: bool = True,
    ):  # fmt: skip
        """
        Add a binarization of a layer feature.

        Parameters
        ----------
        {layer}{target}
        threshold : float, optional
            Threshold value used for binarization.
        larger_true : bool, optional
            If true, values larger than `threshold` will be True.
        """
        from cylindra import cylfilters

        layer = assert_layer(layer, self.parent_viewer)
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        ser = cylfilters.binarize(layer.molecules.features, threshold, target)
        if not larger_true:
            ser = -ser
        feature_name = f"{target}_binarize"
        layer.molecules = layer.molecules.with_features(
            ser.alias(feature_name).cast(pl.Boolean)
        )
        self.reset_choices()
        layer.set_colormap(feature_name, (0, 1), {0: "#A5A5A5", 1: "#FF0000"})
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Label feature clusters")
    def label_feature_clusters(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("label_feature_clusters", dtype_kind="b")}],
    ):  # fmt: skip
        """
        Label a binarized feature column based on the molecules structure.

        This method does the similar task as `scipy.ndimage.label`, where the isolated "islands"
        of True values will be labeled by position integers.

        Parameters
        ----------
        {layer}{target}
        """
        from cylindra import cylfilters
        from napari.utils.colormaps import label_colormap

        layer = assert_layer(layer, self.parent_viewer)
        feat, cmap_info = layer.molecules.features, layer.colormap_info
        nrise = _assert_source_spline_exists(layer).nrise()
        out = cylfilters.label(layer.molecules.features, target, nrise).cast(pl.UInt32)
        feature_name = f"{target}_label"
        layer.molecules = layer.molecules.with_features(out.alias(feature_name))
        self.reset_choices()
        label_max = int(out.max())
        cmap = label_colormap(label_max, seed=0.9414)
        layer.set_colormap(feature_name, (0, label_max), cmap)
        return undo_callback(layer.feature_setter(feat, cmap_info))

    @MoleculesMenu.MoleculeFeatures.wraps
    @set_design(text="Analyze region properties")
    def regionprops_features(
        self,
        layer: MoleculesLayerType,
        target: Annotated[str, {"choices": _choice_getter("regionprops_features", dtype_kind="uif")}],
        label: Annotated[str, {"choices": _choice_getter("regionprops_features", dtype_kind="ui")}],
        properties: Annotated[list[str], {"choices": REGIONPROPS_CHOICES, "widget_type": CheckBoxes}] = ("area", "mean"),
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
        from cylindra._cylindra_ext import RegionProfiler
        from magicclass.ext.polars import DataFrameView

        layer = assert_layer(layer, self.parent_viewer)
        feat = layer.molecules.features
        nth = feat[Mole.nth].cast(pl.Int32).to_numpy()
        pf = feat[Mole.pf].cast(pl.Int32).to_numpy()
        values = feat[target].cast(pl.Float32).to_numpy()
        labels = feat[label].cast(pl.UInt32).to_numpy()
        nrise = _assert_source_spline_exists(layer).nrise()
        npf = _assert_source_spline_exists(layer).props.get_glob(H.npf)

        reg = RegionProfiler.from_features(nth, pf, values, labels, npf, nrise)
        df = pl.DataFrame(reg.calculate(properties))
        view = DataFrameView(value=df)
        dock = self.parent_viewer.window.add_dock_widget(view, name="Region properties")
        dock.setFloating(True)
        return undo_callback(dock.close).with_redo(dock.show)

    @toolbar.wraps
    @set_design(icon=ICON_DIR / "pick_next.svg")
    @bind_key("F3")
    @do_not_record
    def pick_next(self):
        """Automatically pick cylinder center using previous two points."""
        picker = self.toolbar.Adjust._get_picker()
        points = self._reserved_layers.work.data
        if len(points) < 2:
            raise IndexError("Auto picking needs at least two points.")
        imgb = max(self.tomogram.multiscaled, key=lambda x: x[0])[1]
        scale = imgb.scale.x
        next_point = picker.iter_pick(imgb, points[-1], points[-2]).next()
        self._reserved_layers.work.add(next_point)
        change_viewer_focus(self.parent_viewer, next_point / scale, scale)
        return None

    @ImageMenu.wraps
    @thread_worker.with_progress(desc="Paint cylinders ...")
    @set_design(text="Paint cylinders")
    def paint_cylinders(
        self,
        color_by: Annotated[str, {"choices": [H.spacing, H.dimer_twist, H.rise, H.npf]}] = H.spacing,
        cmap: ColormapType = DEFAULT_COLORMAP,
        limits: Optional[tuple[float, float]] = (3.95, 4.28),
    ):  # fmt: skip
        """
        Paint cylinder fragments by its local properties.

        Parameters
        ----------
        {color_by}{cmap}{limits}
        """
        tomo = self.tomogram
        all_df = tomo.splines.collect_localprops()
        if color_by not in all_df.columns:
            raise ValueError(f"Column {color_by} does not exist.")

        paint_device = widget_utils.PaintDevice(
            self._reserved_layers.image.data.shape,
            self._reserved_layers.image.scale[-1],
        )
        lbl = yield from paint_device.paint_cylinders(self.tomogram, color_by)

        # Labels layer properties
        _id = "ID"
        _str = "structure"
        columns = [_id, H.rise, H.spacing, H.dimer_twist, _str]
        df = (
            all_df.select([H.spline_id, H.pos_id, H.rise, H.spacing, H.dimer_twist, H.npf, H.start])
            .with_columns(
                pl.format("{}-{}", pl.col(H.spline_id), pl.col(H.pos_id)).alias(_id),
                pl.format("{}_{}", pl.col(H.npf), pl.col(H.start)).alias(_str),
                pl.col(H.rise),
                pl.col(H.spacing),
                pl.col(H.dimer_twist),
            )
        )  # fmt: skip
        back = pl.DataFrame([pl.Series(_id, [None], dtype=pl.Utf8)])
        props = pl.concat([back, df[columns]], how="diagonal")
        if limits is None:
            limits = float(all_df[color_by].min()), float(all_df[color_by].max())

        @thread_worker.callback
        def _on_return():
            # Add labels layer
            self._reserved_layers.add_paint(lbl, props)
            if self._reserved_layers.paint not in self.parent_viewer.layers:
                self.parent_viewer.add_layer(self._reserved_layers.paint)
            self._reserved_layers.paint.set_colormap(color_by, limits, cmap)
            # TODO: undo paint
            return undo_callback(
                lambda: _Logger.print("undoing paint_cylinders do nothing.")
            )

        return _on_return

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
        visible: bool = True,
    ) -> MoleculesLayer:
        """Add molecules as a points layer to the viewer."""
        return add_molecules(
            self.parent_viewer,
            molecules,
            name,
            source=source,
            metadata=metadata,
            visible=visible,
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
        order : int, default is 1
            Interpolation order of the subtomogram loader.
        """
        mole = self.mole_layers[name].molecules
        return self.tomogram.get_subtomogram_loader(mole, output_shape, order=order)

    def _init_widget_state(self, _=None):
        """Initialize widget state of spline control and local properties for new plot."""
        self.SplineControl.pos = 0
        self.SplineControl["pos"].max = 0
        self.SplineControl.footer.highlight_subvolume = False
        self.LocalProperties._init_text()

        for i in range(3):
            del self.SplineControl.canvas[i].image
            self.SplineControl.canvas[i].layers.clear()
            self.SplineControl.canvas[i].text_overlay.text = ""
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
            for layer in layers:
                self.parent_viewer.add_layer(layer)

        return future_func

    def _undo_callback_for_layer(self, layer: "Layer | list[Layer]"):
        return (
            undo_callback(self._try_removing_layers)
            .with_args(layer)
            .with_redo(self._add_layers_future(layer))
        )

    @thread_worker.callback
    def _send_tomogram_to_viewer(
        self, tomo: CylTomogram, filt: "ImageFilter | None" = None
    ):
        viewer = self.parent_viewer
        self._tomogram = tomo
        bin_size = max(x[0] for x in tomo.multiscaled)
        self._current_binsize = bin_size
        imgb = tomo.get_multiscale(bin_size)
        _is_lazy = isinstance(imgb, ip.LazyImgArray)
        if _is_lazy:
            imgb = ip.zeros(imgb.shape, dtype=np.int8, like=imgb)
            imgb[0, [0, 0, 1, 1], [0, 1, 0, 1]] = 1
            imgb[1, [0, 0, 1, 1], [0, 1, 0, 1]] = 1
        tr = tomo.multiscale_translation(bin_size)
        # update image layer
        if self._reserved_layers.image not in viewer.layers:
            self._reserved_layers.reset_image(imgb, bin_size, tr)
            viewer.add_layer(self._reserved_layers.image)  # TODO: <-- tomogram sampled
        else:
            self._reserved_layers.update_image(imgb, bin_size, tr)
        if self._reserved_layers.highlight in viewer.layers:
            viewer.layers.remove(self._reserved_layers.highlight)

        self.GeneralInfo._refer_tomogram(tomo)

        # update viewer dimensions
        viewer.scale_bar.unit = imgb.scale_unit
        viewer.dims.axis_labels = ("z", "y", "x")
        change_viewer_focus(viewer, np.asarray(imgb.shape) / 2, imgb.scale.x)

        # update labels layer
        self._reserved_layers.init_paint()

        # update overview
        proj = imgb.proj("z")
        self.overview.image = proj
        self.overview.ylim = (0, proj.shape[0])

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
        self.overview.layers.clear()
        self._init_widget_state()
        self._init_layers()
        self.reset_choices()

        # backward compatibility
        if isinstance(filt, bool):
            if filt:
                filt = ImageFilter.Lowpass
            else:
                filt = None
        if filt is not None and not _is_lazy:
            self.filter_reference_image(method=filt)
        self.GeneralInfo.project_desc.value = ""  # clear the project description
        self._need_save = False
        self._macro_image_load_offset = len(self.macro)

    def _on_layer_removing(self, event: "Event"):
        # NOTE: To make recorded macro completely reproducible, removing molecules
        # from the viewer layer list must always be monitored.
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
            if isinstance(layer, MoleculesLayer):
                _layers_to_remove.append(layer.name)
            elif layer in (self._reserved_layers.prof, self._reserved_layers.work):
                _layers_to_remove.append(layer.name)

        for name in _layers_to_remove:
            layer: Layer = viewer.layers[name]
            viewer.layers.remove(layer)

        self._reserved_layers.init_prof_and_work()
        viewer.add_layer(self._reserved_layers.prof)
        viewer.add_layer(self._reserved_layers.work)
        self._reserved_layers.init_paint()
        self.GlobalProperties._init_text()

        # Connect layer events.
        viewer.layers.events.removing.connect(self._on_layer_removing)
        viewer.layers.events.inserted.connect(self._on_layer_inserted)
        return None

    @SplineControl.num.connect
    def _highlight_spline(self):
        i = self.SplineControl.num
        if i is None or self._reserved_layers.prof is None:
            return

        for layer in self.overview.layers:
            if f"spline-{i}" in layer.name:
                layer.color = SplineColor.SELECTED
            else:
                layer.color = SplineColor.DEFAULT

        self._reserved_layers.highlight_spline(i)
        return None

    @SplineControl.num.connect
    def _update_global_properties_in_widget(self, _=None):
        """Show global property values in widgets."""
        i = self.SplineControl.num
        if i is None:
            return
        self.GlobalProperties._set_text(self.splines[i])

    @SplineControl.num.connect
    @SplineControl.pos.connect
    def _update_local_properties_in_widget(self, _=None, *, replot: bool = False):
        i = self.SplineControl.num
        tomo = self.tomogram
        if i is None or i >= len(tomo.splines):
            return
        j = self.SplineControl.pos
        spl = tomo.splines[i]
        if spl.props.has_loc([H.spacing, H.dimer_twist, H.npf, H.start]):
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
        self.overview.add_curve(
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
        self.overview.layers.clear()
        self._reserved_layers.prof.data = []
        scale = self._reserved_layers.scale
        for i, spl in enumerate(self.tomogram.splines):
            self._add_spline_to_images(spl, i)
            if spl._anchors is None:
                continue
            coords = spl.map()
            self.overview.add_scatter(
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

        fgui = get_function_gui(self.cylinder_simulator.update_model)
        fgui.spacing.min, fgui.spacing.max = cfg.spacing_range.astuple()
        fgui.spacing.value = cfg.spacing_range.center
        fgui.dimer_twist.min, fgui.dimer_twist.max = cfg.dimer_twist_range.astuple()
        fgui.dimer_twist.value = cfg.dimer_twist_range.center
        fgui.npf.min, fgui.npf.max = cfg.npf_range.astuple()
        fgui.npf.value = int(cfg.npf_range.center)

        self.cylinder_simulator.spline.props.update_glob(
            spacing=fgui.spacing.value,
            dimer_twist=fgui.dimer_twist.value,
            npf=fgui.npf.value,
        )

        fgui = get_function_gui(self.paint_cylinders)
        fgui.limits.value = cfg.spacing_range.astuple()

        for method in [self.map_monomers, self.map_monomers_with_extensions, self.map_along_pf, self.map_centers]:  # fmt: skip
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


def _assert_source_spline_exists(layer: MoleculesLayer) -> "CylSpline":
    if (spl := layer.source_spline) is None:
        raise ValueError(f"Cannot find the source spline of layer {layer.name!r}.")
    return spl
