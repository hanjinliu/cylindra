import re
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterable, Literal

import impy as ip
import napari
import numpy as np
import polars as pl
from acryo import Molecules, SubtomogramLoader, alignment, pipe
from acryo._utils import SubvolumeOutOfBoundError
from magicclass import (
    MagicTemplate,
    abstractapi,
    do_not_record,
    field,
    impl_preview,
    magicclass,
    magicmenu,
    magictoolbar,
    nogui,
    set_design,
    vfield,
)
from magicclass.ext.dask import dask_thread_worker as dask_worker
from magicclass.logging import getLogger
from magicclass.types import Optional, Path
from magicclass.undo import undo_callback
from magicclass.utils import thread_worker
from magicclass.widgets import HistoryFileEdit, Separator
from scipy.spatial.transform import Rotation

from cylindra import _config, _shared_doc, cylmeasure, utils, widget_utils
from cylindra._napari import LandscapeSurface
from cylindra.components.landscape import Landscape
from cylindra.components.seam_search import (
    BooleanSeamSearcher,
    CorrelationSeamSearcher,
    ManualSeamSearcher,
    SeamSearchResult,
)
from cylindra.const import (
    ALN_SUFFIX,
    ANNEALING_RESULT,
    INTERPOLATION_CHOICES,
    LANDSCAPE_PREFIX,
    SEAM_SEARCH_RESULT,
    FileFilter,
    Ori,
    nm,
)
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import PropertyNames as H
from cylindra.core import ACTIVE_WIDGETS
from cylindra.types import MoleculesLayer
from cylindra.widget_utils import PolarsExprStr, capitalize, timer
from cylindra.widgets import _annealing
from cylindra.widgets import _progress_desc as _pdesc
from cylindra.widgets._annotated import (
    FSCFreq,
    MoleculesLayersType,
    MoleculesLayerType,
    assert_layer,
    assert_list_of_layers,
)
from cylindra.widgets._process_template import TemplateImage
from cylindra.widgets._widget_ext import MultiFileEdit, RandomSeedEdit, RotationsEdit
from cylindra.widgets.subwidgets._child_widget import ChildWidget

if TYPE_CHECKING:
    from dask.array.core import Array
    from napari.layers import Image
    from numpy.typing import NDArray

    from cylindra.components import CylSpline
    from cylindra.components.landscape import AnnealingResult


def _get_template_shape(
    self: "SubtomogramAveraging", size: nm | None, others: dict
) -> nm:
    if size is not None:
        return size
    if "template_path" in others:
        # for `calculate_fsc` and `classify_pca`
        main = self._get_main()
        loader = main.tomogram.get_subtomogram_loader(Molecules.empty())
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(
                others.get("template_path"), allow_none=True
            ),
            mask=self.params._get_mask(params=others.get("mask_params")),
        )
        _size = max(template.shape) * loader.scale
    else:
        _size = max(self._get_shape_in_nm(size))
    return _size


def _validate_landscape_layer(self: "SubtomogramAveraging", layer) -> str:
    if isinstance(layer, LandscapeSurface):
        return layer.name
    elif isinstance(layer, str):
        if layer not in self.parent_viewer.layers:
            raise ValueError(f"{layer!r} does not exist in the viewer.")
        return layer
    else:
        raise TypeError(f"{layer!r} is not a valid landscape.")


def _get_landscape_layers(self: "SubtomogramAveraging", *_) -> list[LandscapeSurface]:
    viewer = self.parent_viewer
    if viewer is None:
        return []
    return [l for l in viewer.layers if isinstance(l, LandscapeSurface)]


_PathOrNone = str | Path | None
_PathOrPathsOrNone = str | Path | list[str | Path] | None

# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_Rotations = Annotated[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    {"widget_type": RotationsEdit},
]
_MaxShifts = Annotated[
    tuple[nm, nm, nm],
    {
        "options": {"max": 10.0, "step": 0.1},
        "label": "max shifts (nm)",
    },
]
_SubVolumeSize = Annotated[
    Optional[nm],
    {
        "text": "use template or mask shape",
        "options": {"value": 12.0, "max": 100.0},
        "label": "size (nm)",
        "validator": _get_template_shape,
    },
]
_DistRangeLon = Annotated[
    tuple[nm, nm],
    {
        "options": {"min": 0.1, "max": 1000.0, "step": 0.05},
        "label": "longitudinal range (nm)",
    },
]
_DistRangeLat = Annotated[
    tuple[nm, nm],
    {
        "options": {"min": 0.1, "max": 1000.0, "step": 0.05},
        "label": "lateral range (nm)",
    },
]
_AngleMaxLon = Annotated[
    float, {"max": 90.0, "step": 0.5, "label": "maximum angle (deg)"}
]
_LandscapeLayer = Annotated[
    LandscapeSurface,
    {
        "choices": _get_landscape_layers,
        "validator": _validate_landscape_layer,
    },
]
_RandomSeeds = Annotated[list[int], {"widget_type": RandomSeedEdit}]

# choices
METHOD_CHOICES = (
    ("Phase Cross Correlation", "pcc"),
    ("Zero-mean Normalized Cross Correlation", "zncc"),
)
_Logger = getLogger("cylindra")


def _get_alignment(method: str):
    match method:
        case "zncc":
            return alignment.ZNCCAlignment
        case "pcc":
            return alignment.PCCAlignment
        case _:  # pragma: no cover
            raise ValueError(f"Method {method!r} is unknown.")


class TemplateChoice(Enum):
    last_average = "Use last averaged image"
    from_file = "From file"
    from_files = "From files"


class MaskChoice(Enum):
    no_mask = "no mask"
    blur_template = "blur template"
    from_file = "from file"


def _choice_getter(method_name: str, dtype_kind: str = ""):
    def _get_choice(self: "SubtomogramAveraging", w=None) -> list[str]:
        # don't use get_function_gui. It causes RecursionError.
        gui = self[method_name].mgui
        if gui is None or gui.layer.value is None:
            return []
        features = gui.layer.value.features
        if dtype_kind == "":
            return features.columns
        return [c for c in features.columns if features[c].dtype.kind in dtype_kind]

    _get_choice.__qualname__ = "SubtomogramAveraging._get_choice"
    return _get_choice


@magicclass(
    layout="horizontal",
    widget_type="groupbox",
    name="Parameters",
    visible=False,
    record=False,
)
class MaskParameters(MagicTemplate):
    """
    Parameters for soft mask creation.

    Soft mask creation has three steps.
    (1) Create binary mask by applying thresholding to the template image.
    (2) Morphological dilation of the binary mask.
    (3) Gaussian filtering the mask.

    Attributes
    ----------
    dilate_radius : nm
        Radius of dilation (nm) applied to binarized template.
    sigma : nm
        Standard deviation (nm) of Gaussian blur applied to the edge of binary image.
    """

    dilate_radius = vfield(0.3, record=False).with_options(min=-20, max=20, step=0.1)
    sigma = vfield(0.8, record=False).with_options(max=20, step=0.1)


@magicclass(layout="horizontal", widget_type="frame", visible=False, record=False)
class mask_path(MagicTemplate):
    """Path to the mask image."""

    mask_path = vfield(Path.Read[FileFilter.IMAGE])


@magicmenu
class Averaging(MagicTemplate):
    """Average subtomograms."""

    average_all = abstractapi()
    average_subset = abstractapi()
    average_groups = abstractapi()
    average_filtered = abstractapi()
    split_and_average = abstractapi()


@magicmenu
class STAnalysis(MagicTemplate):
    """Analysis of subtomograms."""

    calculate_correlation = abstractapi()
    calculate_fsc = abstractapi()
    classify_pca = abstractapi()
    sep0 = field(Separator)

    @magicmenu(name="Seam search")
    class SeamSearch(MagicTemplate):
        seam_search = abstractapi()
        seam_search_by_feature = abstractapi()
        seam_search_manually = abstractapi()
        save_seam_search_result = abstractapi()

    sep1 = field(Separator)
    save_last_average = abstractapi()


@magicmenu
class Alignment(MagicTemplate):
    """Alignment of subtomograms."""

    align_averaged = abstractapi()
    align_all = abstractapi()
    align_all_template_free = abstractapi()
    sep0 = field(Separator)
    align_all_viterbi = abstractapi()
    align_all_annealing = abstractapi()
    save_annealing_scores = abstractapi()
    sep1 = field(Separator)
    TemplateImage = TemplateImage
    extend_filaments = abstractapi()


@magicmenu
class LandscapeMenu(MagicTemplate):
    """Construct and analyze correlation landscape."""

    construct_landscape = abstractapi()
    run_align_on_landscape = abstractapi()
    run_viterbi_on_landscape = abstractapi()
    run_annealing_on_landscape = abstractapi()


@magicclass(record=False, properties={"margins": (0, 0, 0, 0)})
class StaParameters(MagicTemplate):
    """
    Parameters for subtomogram averaging/alignment.

    Attributes
    ----------
    template_path : Path
        Path to the template (reference) image file, or layer name of reconstruction.
    mask_path : str
        Select how to create a mask.
    """

    template_choice = vfield(TemplateChoice.from_file, label="Template")
    avg_info = field("No image", label="info").with_options(
        visible=False, enabled=False
    )
    template_path = field(
        Path.Read[FileFilter.IMAGE], widget_type=HistoryFileEdit, label="path"
    )
    template_paths = field(
        list[Path], widget_type=MultiFileEdit, label="paths"
    ).with_options(filter=FileFilter.IMAGE, visible=False)

    mask_choice = vfield(MaskChoice.no_mask, label="Mask")
    params = field(MaskParameters, name="Parameters")
    mask_path = field(mask_path)

    _last_average: ip.ImgArray | None = None  # the global average result
    _viewer: "napari.Viewer | None" = None

    def __post_init__(self):
        self._template: ip.ImgArray | None = None

        # load history
        line = self.template_path
        for fp in _config.get_template_path_hist():
            if fp.exists():
                line.append_history(str(fp))

        self.min_width = 350

    @template_choice.connect
    def _on_template_switch(self):
        v = self.template_choice
        self.avg_info.visible = v is TemplateChoice.last_average
        self.template_path.visible = v is TemplateChoice.from_file
        self.template_paths.visible = v is TemplateChoice.from_files

    @mask_choice.connect
    def _on_mask_switch(self):
        v = self.mask_choice
        self.params.visible = v is MaskChoice.blur_template
        self.mask_path.visible = v is MaskChoice.from_file

    def _set_last_average(self, img: ip.ImgArray):
        assert img.ndim == 3
        StaParameters._last_average = img
        self.avg_info.value = f"Image of shape {tuple(img.shape)}"

    def _save_history(self):
        try:
            line = self.template_path
            _hist = [str(p) for p in line.get_history()[-20:]]
            _config.set_template_path_hist(_hist)
        except Exception:
            pass

    def _norm_template_param(
        self,
        path: _PathOrPathsOrNone = None,
        allow_none: bool = False,
        allow_multiple: bool = False,
    ) -> pipe.ImageProvider:
        if path is None:
            if StaParameters._last_average is None:
                if allow_none:
                    return None
                raise ValueError("No average image available.")
            return pipe.from_array(
                StaParameters._last_average, StaParameters._last_average.scale.x
            )
        elif isinstance(path, (str, Path)):
            path = Path(path)
            self._save_history()
            if path.is_dir():
                if allow_none:
                    return None
                raise ValueError(f"Template image must be a file, got {path}.")
            return pipe.from_file(path)
        else:
            if not allow_multiple:
                raise ValueError(f"Cannot provide multiple template images: {path}.")
            return pipe.from_files(path)

    def _get_template_input(
        self, allow_multiple: bool = False
    ) -> None | Path | list[Path]:
        match self.template_choice:
            case TemplateChoice.last_average:
                return None
            case TemplateChoice.from_file:
                return Path(self.template_path.value)
            case TemplateChoice.from_files:
                if not allow_multiple:
                    raise ValueError("Cannot provide multiple template images.")
                out = self.template_paths.value
                if len(out) == 0:
                    raise ValueError("No template image provided.")
                elif len(out) == 1:
                    return Path(out[0])
                return [Path(f) for f in out]
            case v:  # pragma: no cover
                raise ValueError(f"Unknown template choice: {v!r}")

    def _get_mask_params(self, params=None) -> str | tuple[nm, nm] | None:
        match self.mask_choice:
            case MaskChoice.no_mask:
                params = None
            case MaskChoice.blur_template:
                params = (self.params.dilate_radius, self.params.sigma)
            case MaskChoice.from_file:
                params = self.mask_path.mask_path
            case v:  # pragma: no cover
                raise ValueError(f"Unknown mask choice: {v!r}")
        return params

    _sentinel = object()

    def _get_mask(self, params: "str | tuple[nm, nm] | None" = _sentinel):
        if params is self._sentinel:
            params = self._get_mask_params()
        if params is None:
            return None
        elif isinstance(params, tuple):
            radius, sigma = params
            return pipe.soft_otsu(radius=radius, sigma=sigma)
        else:
            return pipe.from_file(params)

    def _show_reconstruction(
        self, image: ip.ImgArray, name: str, store: bool = True
    ) -> "Image":
        from skimage.filters.thresholding import threshold_yen

        if StaParameters._viewer is not None:
            try:
                # This line will raise RuntimeError if viewer window had been closed by user.
                StaParameters._viewer.window.activate()
            except (RuntimeError, AttributeError):
                StaParameters._viewer = None
        if StaParameters._viewer is None:
            from cylindra.widgets.subwidgets import Volume

            if qt_window := getattr(self.parent_viewer.window, "_qt_window", None):
                show = qt_window.isVisible()
            else:
                show = False
            StaParameters._viewer = viewer = napari.Viewer(
                title=name,
                axis_labels=("z", "y", "x"),
                ndisplay=3,
                show=show,
            )
            volume_menu = Volume(viewer)
            volume_menu.native.setParent(
                viewer.window.main_menu, volume_menu.native.windowFlags()
            )
            viewer.window.main_menu.addMenu(volume_menu.native)
            viewer.layers.events.connect(volume_menu.reset_choices)
            viewer.window.resize(10, 10)
            viewer.window.activate()
        StaParameters._viewer.scale_bar.visible = True
        StaParameters._viewer.scale_bar.unit = "nm"
        if store:
            self._set_last_average(image)

        return StaParameters._viewer.add_image(
            image,
            scale=image.scale,
            name=name,
            rendering="iso",
            iso_threshold=threshold_yen(image.value),
            blending="opaque",
        )


@magicclass(widget_type="scrollable", use_native_menubar=False)
@_shared_doc.update_cls
class SubtomogramAveraging(ChildWidget):
    """Widget for subtomogram averaging."""

    AveragingMenu = field(Averaging, name="Averaging")
    STAnalysisMenu = field(STAnalysis, name="Analysis")
    AlignmentMenu = field(Alignment, name="Alignment")
    LandscapeMenu = field(LandscapeMenu, name="Landscape")
    params = field(StaParameters)

    @property
    def sub_viewer(self) -> "napari.Viewer | None":
        """The napari viewer for subtomogram averaging."""
        return StaParameters._viewer

    def _template_param(self, *_):  # for bind
        return self.params._get_template_input(allow_multiple=False)

    def _template_params(self, *_):  # for bind
        return self.params._get_template_input(allow_multiple=True)

    def _get_mask_params(self, *_):  # for bind
        return self.params._get_mask_params()

    def _get_template_image(self) -> ip.ImgArray:
        scale = self._get_dummy_loader().scale
        template = self.params._norm_template_param(
            self._template_params(),
            allow_none=False,
            allow_multiple=True,
        ).provide(scale)
        if isinstance(template, list):
            template = ip.asarray(np.stack(template, axis=0), axes="zyx")
        else:
            template = ip.asarray(template, axes="zyx")
        return template.set_scale(zyx=scale, unit="nm")

    def _get_mask_image(self, template_params) -> ip.ImgArray:
        loader = self._get_dummy_loader()
        _, mask = loader.normalize_input(
            self.params._norm_template_param(
                template_params, allow_none=True, allow_multiple=True
            ),
            self.params._get_mask(),
        )
        if mask is None:
            raise ValueError("Mask is None.")
        return ip.asarray(mask, axes="zyx").set_scale(zyx=loader.scale, unit="nm")

    @magictoolbar
    class STATools(MagicTemplate):
        show_template = abstractapi()
        show_template_original = abstractapi()
        show_mask = abstractapi()

    @set_design(icon="ic:baseline-view-in-ar", location=STATools)
    @do_not_record
    def show_template(self):
        """Load and show template image in the scale of the tomogram."""
        template = self._get_template_image()
        self._show_rec(template, name="Template image", store=False)

    @set_design(icon="material-symbols:view-in-ar", location=STATools)
    @do_not_record
    def show_template_original(self):
        """Load and show template image in the original scale."""
        _input = self._template_params()
        if _input is None:
            raise ValueError("No template path provided.")
        elif isinstance(_input, Path):
            self._show_rec(ip.imread(_input), name="Template image", store=False)
        else:
            for i, fp in enumerate(_input):
                img = ip.imread(fp)
                self._show_rec(img, name=f"Template image [{i}]", store=False)

    @set_design(icon="fluent:shape-organic-20-filled", location=STATools)
    @do_not_record
    def show_mask(self):
        """Load and show mask image in the scale of the tomogram."""
        mask = self._get_mask_image(self._template_params())
        self._show_rec(mask, name="Mask image", store=False)

    @property
    def last_average(self) -> "ip.ImgArray | None":
        """Last averaged image if exists."""
        return StaParameters._last_average

    def _get_shape_in_nm(self, default: int = None) -> tuple[nm, nm, nm]:
        if default is None:
            tmp = self._get_template_image()
            return tuple(np.array(tmp.sizesof("zyx")) * tmp.scale.x)
        else:
            return (default,) * 3

    @thread_worker.callback
    def _show_rec(self, img: ip.ImgArray, name: str, store: bool = True):
        return self.params._show_reconstruction(img, name, store)

    def _get_loader(
        self,
        binsize: int,
        molecules: Molecules,
        shape: tuple[nm, nm, nm] = None,
        order: int = 1,
    ) -> SubtomogramLoader:
        """
        Returns proper subtomogram loader, template image and mask image that matche the
        bin size.
        """
        return self._get_main().tomogram.get_subtomogram_loader(
            molecules,
            binsize=binsize,
            order=order,
            output_shape=shape,
        )

    def _get_dummy_loader(self):
        return self._get_loader(binsize=1, molecules=Molecules.empty())

    def _get_available_binsize(self, _=None) -> list[int]:
        parent = self._get_main()
        out = [x[0] for x in parent.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return out

    @set_design(text="Average all molecules", location=Averaging)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Subtomogram averaging of {!r}"))
    def average_all(
        self,
        layers: MoleculesLayersType,
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Subtomogram averaging using all the molecules in the selected layer(s).

        If multiple layers are selected, subtomograms around all the molecules will
        be averaged.

        Parameters
        ----------
        {layers}{size}{interpolation}{bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            _concat_molecules(layers), shape, binsize=bin_size, order=interpolation
        )
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{_avg_name(layers)}")

    @set_design(text="Average subset of molecules", location=Averaging)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Subtomogram averaging (subset) of {!r}"))  # fmt: skip
    def average_subset(
        self,
        layers: MoleculesLayersType,
        size: _SubVolumeSize = None,
        method: Literal["steps", "first", "last", "random"] = "steps",
        number: int = 64,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Subtomogram averaging using a subset of subvolumes.

        If multiple layers are selected, subtomograms around all the molecules will
        be concatenated before choosing a subset.

        Parameters
        ----------
        {layers}{size}
        method : str, optional
            How to choose subtomogram subset.
            (1) steps: Each 'steps' subtomograms from the tip of spline.
            (2) first: First subtomograms.
            (3) last: Last subtomograms.
            (4) random: choose randomly.
        number : int, default
            Number of subtomograms to use.
        {bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        molecules = _concat_molecules(layers)
        nmole = len(molecules)
        shape = self._get_shape_in_nm(size)
        sl = _get_slice_for_average_subset(method, nmole, number)
        mole = molecules.subset(sl)
        loader = parent.tomogram.get_subtomogram_loader(
            mole, shape, binsize=bin_size, order=1
        )
        img = ip.asarray(loader.average(), axes="zyx").set_scale(zyx=loader.scale)
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG(n={number})]{_avg_name(layers)}")

    @set_design(text="Average group-wise", location=Averaging)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Grouped subtomogram averaging of {!r}"))  # fmt: skip
    def average_groups(
        self,
        layers: MoleculesLayersType,
        size: _SubVolumeSize = None,
        by: PolarsExprStr = "col('pf-id')",
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Group-wise subtomogram averaging using molecules grouped by the given expression.

        This method first group molecules by its features, and then average each group.
        This method is useful for such as get average of each protofilament and segmented
        subtomogram averaging.

        Parameters
        ----------
        {layers}{size}
        by : str or polars expression
            Expression to group molecules.
        {interpolation}{bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            _concat_molecules(layers), shape, binsize=bin_size, order=interpolation
        )
        expr = widget_utils.norm_expr(by)
        avg_dict = loader.groupby(expr).average()
        avgs = np.stack([avg_dict[k] for k in sorted(avg_dict.keys())], axis=0)
        img = ip.asarray(avgs, axes="pzyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{_avg_name(layers)}", store=False)

    @set_design(text="Average filtered", location=Averaging)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Filtered subtomogram averaging of {!r}"))  # fmt: skip
    def average_filtered(
        self,
        layers: MoleculesLayersType,
        size: _SubVolumeSize = None,
        predicate: PolarsExprStr = "col('pf-id') == 0",
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Subtomogram averaging using molecules filtered by the given expression.

        This method first concatenate molecules in the selected layers, and then filter them
        by the predicate.

        Parameters
        ----------
        {layers}{size}
        predicate : str or polars expression
            Filter expression to select molecules.
        {interpolation}{bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            _concat_molecules(layers), shape, binsize=bin_size, order=interpolation
        )
        avg = loader.filter(widget_utils.norm_expr(predicate)).average()
        img = ip.asarray(avg, axes="zyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{_avg_name(layers)}")

    @set_design(text="Split and average molecules", location=Averaging)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Split-and-averaging of {!r}"))  # fmt: skip
    def split_and_average(
        self,
        layers: MoleculesLayersType,
        n_pairs: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Split molecules into two groups and average separately.

        Parameters
        ----------
        {layers}
        n_pairs : int, default 1
            How many pairs of average will be calculated.
        {size}{interpolation}{bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        molecules = _concat_molecules(layers)
        shape = self._get_shape_in_nm(size)
        loader = parent.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation
        )
        axes = "ipzyx" if n_pairs > 1 else "pzyx"
        img = ip.asarray(loader.average_split(n_set=n_pairs), axes=axes)
        img.set_scale(zyx=loader.scale)
        t0.toc()
        return self._show_rec.with_args(img, f"[Split]{_avg_name(layers)}", store=False)

    @set_design(text="Align average to template", location=Alignment)
    @dask_worker.with_progress(descs=_pdesc.align_averaged_fmt)
    def align_averaged(
        self,
        layers: MoleculesLayersType,
        template_path: Annotated[_PathOrNone, {"bind": _template_param}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        max_shifts: Optional[_MaxShifts] = None,
        rotations: _Rotations = ((0.0, 0.0), (15.0, 1.0), (3.0, 1.0)),
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
    ):  # fmt: skip
        """
        Align the averaged image at current monomers to the template image.

        This function creates a new layer with transformed monomers, which should
        align well with template image.

        Parameters
        ----------
        {layers}{template_path}{mask_params}{max_shifts}{rotations}{bin_size}{method}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()

        new_layers = list[MoleculesLayer]()

        @thread_worker.callback
        def _on_yield(mole_trans: Molecules, layer: MoleculesLayer):
            points = parent.add_molecules(
                mole_trans,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
                source=layer.source_component,
            )
            new_layers.append(points)
            layer.visible = False
            _Logger.print_html(f"{layer.name!r} &#8594; {points.name!r}")

        mole = layers[0].molecules
        loader = self._get_loader(bin_size, mole, order=1)
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(
                template_path, allow_multiple=False
            ),
            mask=self.params._get_mask(params=mask_params),
        )
        temp_norm = utils.normalize_image(template)

        _scale = parent.tomogram.scale * bin_size

        if max_shifts is None:
            max_shifts = _default_align_averaged_shifts(mole)

        model = _get_alignment(method)(
            template,
            mask,
            rotations=rotations,
            tilt=None,  # NOTE: because input is an average
        )
        _spl_globs = list[
            tuple[weakref.ReferenceType["CylSpline"], pl.DataFrame, pl.DataFrame]
        ]()
        for layer in layers:
            mole = layer.molecules
            loader = self._get_loader(bin_size, mole, order=1)
            _img_trans, result = model.fit(
                loader.average(template.shape),
                max_shifts=[_s / _scale for _s in max_shifts],
            )

            rotator = Rotation.from_quat(result.quat)
            svec = result.shift * _scale
            _mole_trans = mole.linear_transform(result.shift * _scale, rotator)

            # write offsets to spline globalprops if available
            if spl := layer.source_spline:
                _mole_trans = _update_mole_pos(_mole_trans, mole, spl)
                if spl.radius is None:
                    _radius: nm = cylmeasure.calc_radius(mole, spl).mean()
                else:
                    _radius = spl.radius
                _glob_old = spl.props.glob.clone()
                _glob_new = _update_offset(spl, rotator.apply(svec), _radius)
                spl.props.glob = _glob_new
                _spl_globs.append((weakref.ref(spl), _glob_old, _glob_new))

            yield _on_yield.with_args(_mole_trans, layer)

            # create images for visualization in the logger. Image is magenta, template is green
            img_norm = utils.normalize_image(_img_trans)
            merge = np.stack([img_norm, temp_norm, img_norm], axis=-1)
            with _Logger.set_plt():
                widget_utils.plot_projections(merge)

            # logging
            rvec = rotator.as_rotvec()
            _fmt = "  {:.2f}  ".format
            _Logger.print_table(
                [
                    ["", "X", "Y", "Z"],
                    ["Shift (nm)", _fmt(svec[2]), _fmt(svec[1]), _fmt(svec[0])],
                    ["Rot vector", _fmt(rvec[2]), _fmt(rvec[1]), _fmt(rvec[0])],
                ],
                header=False,
                index=False,
            )

        t0.toc()

        @thread_worker.callback
        def _align_averaged_on_return():
            @undo_callback
            def _out():
                parent._try_removing_layers(new_layers)
                for spl_ref, old, _ in _spl_globs:
                    if spl := spl_ref():
                        spl.props.glob = old

            @_out.with_redo
            def _out():
                parent._add_layers_future(new_layers)()
                for spl_ref, _, new in _spl_globs:
                    if spl := spl_ref():
                        spl.props.glob = new

            return _out

        return _align_averaged_on_return

    sep0 = field(Separator)

    @set_design(text="Align all molecules", location=Alignment)
    @dask_worker.with_progress(descs=_pdesc.align_all_fmt)
    def align_all(
        self,
        layers: MoleculesLayersType,
        template_path: Annotated[_PathOrPathsOrNone, {"bind": _template_params}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Align the input template image to all the molecules.

        Parameters
        ----------
        {layers}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{method}{bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()

        combiner = MoleculesCombiner()

        loader = self._get_loader(
            binsize=bin_size,
            molecules=combiner.concat(layer.molecules for layer in layers),
            order=interpolation,
        )
        aligned_loader = loader.align(
            template=self.params._norm_template_param(
                template_path, allow_multiple=True
            ),
            mask=self.params._get_mask(params=mask_params),
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            alignment_model=_get_alignment(method),
            tilt=main.tomogram.tilt_model,
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    @set_design(text="Align all (template-free)", location=Alignment)
    @dask_worker.with_progress(descs=_pdesc.align_template_free_fmt)
    def align_all_template_free(
        self,
        layers: MoleculesLayersType,
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: _SubVolumeSize = 12.0,
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Run template-free alignment for the given layers (EXPERIMENTAL).

        Parameters
        ----------
        {layers}{mask_params}{size}{max_shifts}{rotations}{cutoff}{interpolation}
        {method}{bin_size}
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()
        combiner = MoleculesCombiner()
        molecules = combiner.concat(layer.molecules for layer in layers)
        mask = self.params._get_mask(params=mask_params)
        if size is None:
            raise NotImplementedError("'size' must be given.")
        else:
            shape = tuple(
                main.tomogram.nm2pixel(self._get_shape_in_nm(size), binsize=bin_size)
            )

        aligned_loader = (
            self._get_loader(binsize=bin_size, molecules=molecules, order=interpolation)
            .reshape(shape=shape)
            .align_no_template(
                mask=mask,
                max_shifts=max_shifts,
                rotations=rotations,
                cutoff=cutoff,
                alignment_model=_get_alignment(method),
                tilt=main.tomogram.tilt_model,
            )
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    sep1 = field(Separator)

    @set_design(text="Viterbi Alignment", location=Alignment)
    @dask_worker.with_progress(descs=_pdesc.align_viterbi_fmt)
    def align_all_viterbi(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[_PathOrPathsOrNone, {"bind": _template_params}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        range_long: _DistRangeLon = (4.0, 4.28),
        angle_max: _AngleMaxLon = 5.0,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
    ):  # fmt: skip
        """
        Subtomogram alignment using 1D Viterbi alignment.

        1D Viterbi alignment is an alignment algorithm that considers the distance and
        the skew angle between every longitudinally adjacent monomers. The classical
        Viterbi algorithm is used to find the global optimal solution of the alignment.
        Note that Viterbi alignment is data size dependent, i.e. the alignment result
        of a molecule may vary depending on the total number of molecules in the dataset.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{range_long}{angle_max}{bin_size}{upsample_factor}
        """
        t0 = timer()
        layer = assert_layer(layer, self.parent_viewer)
        landscape = self._construct_landscape(
            molecules=layer.molecules,
            template_path=template_path,
            mask_params=mask_params,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            order=interpolation,
            upsample_factor=upsample_factor,
            bin_size=bin_size,
        )

        yield
        mole = landscape.run_viterbi_along_spline(
            spl=layer.source_spline,
            range_long=range_long,
            angle_max=angle_max,
        )
        t0.toc()
        return self._align_all_on_return.with_args([mole], [layer])

    @set_design(text="Simulated annealing", location=Alignment)
    @dask_worker.with_progress(descs=_pdesc.align_annealing_fmt)
    def align_all_annealing(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[_PathOrPathsOrNone, {"bind": _template_params}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        range_long: _DistRangeLon = (4.0, 4.28),
        range_lat: _DistRangeLat = (5.1, 5.3),
        angle_max: _AngleMaxLon = 5.0,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        random_seeds: _RandomSeeds = (0, 1, 2, 3, 4),
    ):  # fmt: skip
        """
        2D-constrained subtomogram alignment using simulated annealing.

        This alignment method considers the distance between every adjacent monomers.
        Two-dimensionally connected optimization can be approximated by the simulated
        annealing algorithm.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{range_long}{range_lat}{angle_max}{bin_size}
        {temperature_time_const}{upsample_factor}{random_seeds}
        """
        t0 = timer()
        layer = assert_layer(layer, self.parent_viewer)
        if layer.source_spline is None:
            raise ValueError("RMA requires a spline.")
        main = self._get_main()
        landscape = self._construct_landscape(
            molecules=layer.molecules,
            template_path=template_path,
            mask_params=mask_params,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            order=interpolation,
            bin_size=bin_size,
            upsample_factor=upsample_factor,
        )
        yield
        mole, results = landscape.run_annealing_along_spline(
            layer.source_spline,
            range_long=range_long,
            range_lat=range_lat,
            angle_max=angle_max,
            temperature_time_const=temperature_time_const,
            random_seeds=random_seeds,
        )
        t0.toc()

        @thread_worker.callback
        def _on_return():
            points = main.add_molecules(
                mole,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
                source=layer.source_component,
                metadata={ANNEALING_RESULT: results[0]},
            )
            layer.visible = False
            with _Logger.set_plt():
                _annealing.plot_annealing_result(results)

            return self._undo_for_new_layer([layer.name], [points])

        return _on_return

    @set_design(text=capitalize, location=Alignment)
    @dask_worker.with_progress(desc="Extending filaments")
    def extend_filaments(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[_PathOrPathsOrNone, {"bind": _template_params}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.2,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        range_long: _DistRangeLon = (4.0, 4.28),
        angle_max: _AngleMaxLon = 10.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 3,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        min_score: Annotated[float, {"min": -1.0, "max": 1.0}] = 0.05,
    ):
        """
        Extend filaments by iterative Viterbi alignment (EXPERIMENTAL).

        ```
        x-o-o   <- fix the first molecule, align the second and third molecules
          x-o-o <- extend by one molecule, and repeat the process
        ```

        Parameters
        ----------
        {layer}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{range_long}{angle_max}{upsample_factor}{bin_size}
        min_score : float, default 0.05
            Minimum score to accept the alignment result.
        """
        import matplotlib.pyplot as plt

        from cylindra._dask import compute, delayed

        t0 = timer()
        layer = assert_layer(layer, self.parent_viewer)

        func = delayed(self._extend_pf)
        if Mole.pf in layer.molecules.features:
            molecules = list(layer.molecules.group_by(Mole.pf))
        else:
            molecules = [(None, layer.molecules)]
        tasks = [
            func(
                each, template_path=template_path, mask_params=mask_params,
                max_shifts=max_shifts, rotations=rotations, cutoff=cutoff,
                order=interpolation, range_long=range_long, angle_max=angle_max,
                upsample_factor=upsample_factor, min_score=min_score, bin_size=bin_size,
                npf=pf,
            )
            for pf, each in molecules
        ]  # fmt: skip
        results = compute(*tasks)
        mole_out = Molecules.concat([result[0] for result in results])
        t0.toc()

        with _Logger.set_plt():
            plt.figure()
            for result in results:
                plt.plot(result[1], lw=1)
            plt.show()
        return self._align_all_on_return.with_args([mole_out], [layer])

    @set_design(text=capitalize, location=LandscapeMenu)
    @dask_worker.with_progress(descs=_pdesc.construct_landscape_fmt)
    def construct_landscape(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[_PathOrPathsOrNone, {"bind": _template_params}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
    ):
        """
        Construct a cross-correlation landscape for subtomogram alignment.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{bin_size}{upsample_factor}
        """
        layer = assert_layer(layer, self.parent_viewer)
        lnd = self._construct_landscape(
            molecules=layer.molecules,
            template_path=template_path,
            mask_params=mask_params,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            order=interpolation,
            bin_size=bin_size,
            upsample_factor=upsample_factor,
        )
        surf = LandscapeSurface(lnd, name=f"{LANDSCAPE_PREFIX}{layer.name}")
        surf.source_component = layer.source_component

        @thread_worker.callback
        def _on_return():
            self.parent_viewer.add_layer(surf)
            self._get_main()._reserved_layers.to_be_removed.add(surf)
            layer.visible = False

        return _on_return

    @set_design(text="Run alignment on landscape", location=LandscapeMenu)
    @dask_worker.with_progress(desc="Peak detection on landscape")
    def run_align_on_landscape(self, landscape_layer: _LandscapeLayer):
        """Find the optimal displacement for each molecule on the landscape."""
        landscape_layer = _assert_landscape_layer(landscape_layer, self.parent_viewer)
        landscape = landscape_layer.landscape
        spl = landscape_layer.source_spline
        mole_opt, _ = landscape.run_min_energy(spl)
        return self._align_on_landscape_on_return.with_args(
            mole_opt, landscape_layer.name, spl
        )

    @set_design(text="Run Viterbi alignment on landscape", location=LandscapeMenu)
    @dask_worker.with_progress(desc="Running Viterbi alignment")
    def run_viterbi_on_landscape(
        self,
        landscape_layer: _LandscapeLayer,
        range_long: _DistRangeLon = (4.0, 4.28),
        angle_max: _AngleMaxLon = 5.0,
    ):
        """
        Run Viterbi alignment on the landscape.

        Parameters
        ----------
        {landscape_layer}{range_long}{angle_max}
        """
        t0 = timer()
        landscape_layer = _assert_landscape_layer(landscape_layer, self.parent_viewer)
        spl = landscape_layer.source_spline
        mole = landscape_layer.landscape.run_viterbi_along_spline(
            spl=spl,
            range_long=range_long,
            angle_max=angle_max,
        )
        t0.toc()
        return self._align_on_landscape_on_return.with_args(
            mole, landscape_layer.name, spl
        )

    @set_design(text="Run annealing on landscape", location=LandscapeMenu)
    @dask_worker.with_progress(desc="Running simulated annealing")
    def run_annealing_on_landscape(
        self,
        landscape_layer: _LandscapeLayer,
        range_long: _DistRangeLon = (4.0, 4.28),
        range_lat: _DistRangeLat = (5.1, 5.3),
        angle_max: _AngleMaxLon = 5.0,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        random_seeds: _RandomSeeds = (0, 1, 2, 3, 4),
    ):
        """
        Run simulated annealing on the landscape.

        Parameters
        ----------
        {landscape_layer}{range_long}{range_lat}{angle_max}{temperature_time_const}
        {random_seeds}
        """
        t0 = timer()
        landscape_layer = _assert_landscape_layer(landscape_layer, self.parent_viewer)
        spl = landscape_layer.source_spline
        if spl is None:
            raise ValueError("RMA requires a spline.")
        mole, results = landscape_layer.landscape.run_annealing_along_spline(
            spl=spl,
            range_long=range_long,
            range_lat=range_lat,
            angle_max=angle_max,
            temperature_time_const=temperature_time_const,
            random_seeds=random_seeds,
        )
        t0.toc()

        @thread_worker.callback
        def _plot_result():
            with _Logger.set_plt():
                _annealing.plot_annealing_result(results)

        yield _plot_result
        return self._align_on_landscape_on_return.with_args(
            mole,
            landscape_layer.name,
            source=spl,
            metadata={ANNEALING_RESULT: results[0]},
        )

    def _get_layers_with_annealing_result(self, *_) -> list[MoleculesLayer]:
        if self.parent_viewer is None:
            return []
        return [
            (layer.name, layer)
            for layer in self.parent_viewer.layers
            if ANNEALING_RESULT in layer.metadata
        ]

    @set_design(text=capitalize, location=Alignment)
    @do_not_record
    def save_annealing_scores(
        self,
        layer: Annotated[MoleculesLayer, {"choices": _get_layers_with_annealing_result}],
        path: Path.Save[FileFilter.CSV],
    ):  # fmt: skip
        """Save RMA scores to a CSV file."""
        layer = assert_layer(layer, self.parent_viewer)
        try:
            result: AnnealingResult = layer.metadata[ANNEALING_RESULT]
        except KeyError:
            raise ValueError(
                f"Layer {layer!r} does not have annealing result."
            ) from None
        x = result.batch_size * np.arange(result.energies.size)
        df = pl.DataFrame({"iteration": x, "score": -result.energies})
        return df.write_csv(path, include_header=False)

    @set_design(text=capitalize, location=STAnalysis)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Calculating correlations of {!r}"))  # fmt: skip
    def calculate_correlation(
        self,
        layers: MoleculesLayersType,
        template_path: Annotated[_PathOrPathsOrNone, {"bind": _template_params}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        metric: Literal["zncc", "ncc"] = "zncc",
        column_prefix: str = "score",
    ):
        """
        Calculate correlation between template images and the subtomograms.

        This method will load every subtomograms, calculate the correlation between
        the template images and each subtomogram, and save the correlation values
        as new columns in the molecules features.

        Parameters
        ----------
        {layers}{template_path}{mask_params}{interpolation}{bin_size}
        metric : str, default "zncc"
            Metric to calculate correlation.
        column_prefix : str, default "score"
            Prefix of the column names of the calculated correlations.
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()
        scale = main.tomogram.scale
        tmps = []
        _shapes = set()
        if isinstance(template_path, (Path, str)):
            template_path = [template_path]
        for path in template_path:
            template_image = pipe.from_file(path).provide(scale)
            tmps.append(template_image)
            _shapes.add(template_image.shape)
        if len(_shapes) != 1:
            raise ValueError(f"Inconsistent shapes: {_shapes}")
        output_shape = _shapes.pop()
        mask = self.params._get_mask(mask_params)
        match mask:
            case None:
                msk = 1
            case pipe.ImageConverter:
                msk = mask.convert(np.stack(tmps, axis=0).sum(axis=0), scale)
            case pipe.ImageProvider:
                msk = mask.provide(scale)
            case _:  # pragma: no cover
                raise RuntimeError("Unreachable")
        corr_fn = ip.ncc if metric == "ncc" else ip.zncc
        funcs = []
        for tmp in tmps:
            funcs.append(_define_correlation_function(tmp, msk, corr_fn))

        for layer in layers:
            mole = layer.molecules
            out = main.tomogram.get_subtomogram_loader(
                mole,
                order=interpolation,
                output_shape=output_shape,
                binsize=bin_size,
            ).apply(
                funcs,
                schema=[f"{column_prefix}_{i}" for i in range(len(template_path))],
            )
            layer.molecules = layer.molecules.with_features(out.cast(pl.Float32))
        return None

    @set_design(text="Calculate FSC", location=STAnalysis)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Calculating FSC of {!r}"))
    def calculate_fsc(
        self,
        layers: MoleculesLayersType,
        template_path: Annotated[_PathOrNone, {"bind": _template_param}] = None,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        n_pairs: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: FSCFreq = None,
    ):
        """
        Calculate Fourier Shell Correlation using the selected monomer layer.

        Parameters
        ----------
        {layers}
        template_path : template input type
            Used only when soft-Otsu mask parameters are given.
        {mask_params}{size}
        seed : int, optional
            Random seed used for subtomogram sampling.
        {interpolation}
        n_pairs : int, default 1
            How many sets of image pairs will be generated to average FSC.
        show_average : bool, default True
            If true, subtomogram average will be shown after FSC calculation.
        dfreq : float, default 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be
            calculated at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        t0 = timer()
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()
        mole = _concat_molecules(layers)

        loader = main.tomogram.get_subtomogram_loader(mole, order=interpolation)
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path, allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        fsc, avg = loader.reshape(
            template=template if size is None else None,
            mask=mask,
            shape=None if size is None else (main.tomogram.nm2pixel(size),) * 3,
        ).fsc_with_average(mask=mask, seed=seed, n_set=n_pairs, dfreq=dfreq)

        if show_average:
            img_avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale)
        else:
            img_avg = None

        result = widget_utils.FscResult.from_dataframe(fsc, loader.scale)
        criteria = [0.5, 0.143]
        _name = _avg_name(layers)
        t0.toc()

        @thread_worker.callback
        def _calculate_fsc_on_return():
            _Logger.print_html(f"<b>Fourier Shell Correlation of {_name!r}</b>")
            with _Logger.set_plt():
                result.plot(criteria)
            for _c in criteria:
                _r = result.get_resolution(_c)
                _Logger.print_html(f"Resolution at FSC={_c:.3f} ... <b>{_r:.3f} nm</b>")

            if img_avg is not None:
                _rec_layer: "Image" = self._show_rec(
                    img_avg,
                    name=f"[AVG]{_name}",
                )
                _rec_layer.metadata["fsc"] = result

        return _calculate_fsc_on_return

    @set_design(text="PCA/K-means classification", location=STAnalysis)
    @dask_worker.with_progress(descs=_pdesc.classify_pca_fmt)
    def classify_pca(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[_PathOrNone, {"bind": _template_param}] = None,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        size: _SubVolumeSize = None,
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
        n_components: Annotated[int, {"min": 2, "max": 20}] = 2,
        n_clusters: Annotated[int, {"min": 2, "max": 100}] = 2,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
    ):  # fmt: skip
        """
        Classify molecules in a layer using PCA and K-means clustering.

        Parameters
        ----------
        {layer}
        template_path : template input type
            Used only when soft-Otsu mask parameters are given.
        {mask_params}{size}{cutoff}{interpolation}{bin_size}
        n_components : int, default 2
            The number of PCA dimensions.
        n_clusters : int, default
            The number of clusters.
        seed : int, default 0
            Random seed.
        """
        from cylindra.widgets.subwidgets import PcaViewer

        t0 = timer()
        layer = assert_layer(layer, self.parent_viewer)
        tomo = self._get_main().tomogram
        loader = self._get_loader(
            binsize=bin_size, molecules=layer.molecules, order=interpolation
        )
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path, allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        shape = None
        if size is not None and mask is None:
            shape = (tomo.nm2pixel(size, binsize=bin_size),) * 3
        out, pca = loader.reshape(
            template=template if mask is None and shape is None else None,
            mask=mask,
            shape=shape,
        ).classify(
            mask=mask,
            seed=seed,
            cutoff=cutoff,
            n_components=n_components,
            n_clusters=n_clusters,
            label_name="cluster",
        )

        avgs_dict = out.groupby("cluster").average()
        avgs = ip.asarray(
            np.stack(list(avgs_dict.values()), axis=0), axes=["cluster", "z", "y", "x"]
        ).set_scale(zyx=loader.scale, unit="nm")
        t0.toc()

        @thread_worker.callback
        def _on_return():
            layer.molecules = out.molecules  # update features
            pca_viewer = PcaViewer(pca)
            pca_viewer.native.setParent(self.native, pca_viewer.native.windowFlags())
            pca_viewer.show()
            self._show_rec(avgs, name=f"[PCA]{layer.name}", store=False)
            ACTIVE_WIDGETS.add(pca_viewer)

        return _on_return

    @set_design(text="Seam search by correlation", location=STAnalysis.SeamSearch)
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Seam search of {!r}"))
    def seam_search(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[_PathOrNone, {"bind": _template_param}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        anti_template_path: Annotated[Optional[Path.Read[FileFilter.IMAGE]], {"text": "Do not use anti-template", "label": "anti-template path"}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        npf: Annotated[Optional[int], {"text": "use global properties"}] = None,
        show_average: Annotated[str, {"label": "show averages as", "choices": [None, "Raw", "Filtered"]}] = "Filtered",
        cutoff: _CutoffFreq = 0.25,
    ):  # fmt: skip
        """
        Search for the best seam position.

        Try all patterns of seam positions and compare cross correlation values. If
        molecule assembly has 13 protofilaments, this method will try 26 patterns.

        Parameters
        ----------
        {layer}{template_path}{mask_params}
        anti_template_path : Path, optional
            The anti-template used for seam search. For instance, if the template is
            beta-tubulin, the anti-template is alpha-tubulin.
        {interpolation}
        npf : int, optional
            Number of protofilaments. By default the global properties stored in the
            corresponding spline will be used.
        show_average : bool, default True
            If true, all the subtomogram averages will be shown.
        {cutoff}
        """
        t0 = timer()
        layer = assert_layer(layer, self.parent_viewer)
        loader, npf = self._seam_search_input(layer, npf, interpolation)
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        if anti_template_path is not None:
            anti_template = ip.asarray(
                pipe.from_file(anti_template_path).provide(loader.scale),
                axes="zyx",
            )
        else:
            anti_template = None

        seam_searcher = CorrelationSeamSearcher(npf)
        result = seam_searcher.search(
            loader=loader,
            template=ip.asarray(template, axes="zyx"),
            anti_template=anti_template,
            mask=mask,
            cutoff=cutoff,
        )

        t0.toc()

        @thread_worker.callback
        def _seam_search_on_return():
            new_feat = result.as_series(loader.molecules.count())
            layer.features = layer.molecules.features.with_columns(new_feat)
            layer.metadata[SEAM_SEARCH_RESULT] = result
            if show_average is not None:
                if show_average == "Filtered":
                    sigma = 0.25 / loader.scale
                    result.averages.gaussian_filter(sigma=sigma, update=True)
                self._show_rec(result.averages, layer.name, store=False)

            # plot all the correlation
            _Logger.print_html("<code>seam_search</code>")
            with _Logger.set_plt():
                _Logger.print(f"layer = {layer.name!r}")
                _Logger.print(f"template = {Path(template_path).as_posix()!r}")
                if anti_template_path is not None:
                    _Logger.print(
                        f"anti_template = {Path(anti_template_path).as_posix()!r}"
                    )
                widget_utils.plot_seam_search_result(result.scores, npf)

        return _seam_search_on_return

    @set_design(text=capitalize, location=STAnalysis.SeamSearch)
    def seam_search_by_feature(
        self,
        layer: MoleculesLayerType,
        by: Annotated[str, {"choices": _choice_getter("seam_search_by_feature")}],
    ):
        """
        Search for seams by a feature.

        Parameters
        ----------
        {layer}
        by : str
            Name of the feature that will be used for seam search.
        """
        layer = assert_layer(layer, self.parent_viewer)
        feat = layer.molecules.features
        if by not in feat.columns:
            raise ValueError(f"Column {by} does not exist.")
        npf = utils.roundint(layer.molecules.features[Mole.pf].max() + 1)
        seam_searcher = BooleanSeamSearcher(npf)
        result = seam_searcher.search(feat[by])
        new_feat = result.as_series(feat.shape[0])
        layer.features = layer.molecules.features.with_columns(new_feat)
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    @set_design(text=capitalize, location=STAnalysis.SeamSearch)
    def seam_search_manually(
        self,
        layer: MoleculesLayerType,
        location: int = 0,
    ):
        """
        Search for seams manually.

        Seam location is represented by a number in the range [0, 2 * npf - 1].

        Parameters
        ----------
        {layer}
        location : int
            Seam location.
        """
        layer = assert_layer(layer, self.parent_viewer)
        feat = layer.molecules.features
        npf = utils.roundint(layer.molecules.features[Mole.pf].max() + 1)
        seam_searcher = ManualSeamSearcher(npf)
        result = seam_searcher.search(location)
        new_feat = result.as_series(feat.shape[0])
        layer.features = layer.molecules.features.with_columns(new_feat)
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    def _get_seam_searched_layers(self, *_) -> list[MoleculesLayer]:
        if self.parent_viewer is None:
            return []
        return [
            (layer.name, layer)
            for layer in self.parent_viewer.layers
            if SEAM_SEARCH_RESULT in layer.metadata
        ]

    @set_design(text="Save seam search result", location=STAnalysis.SeamSearch)
    @do_not_record
    def save_seam_search_result(
        self,
        layer: Annotated[MoleculesLayer | str, {"choices": _get_seam_searched_layers}],
        path: Path.Save[FileFilter.CSV],
    ):
        """
        Save seam search result.

        Parameters
        ----------
        layer : str or MoleculesLayer
            Layer that contains seam search result.
        path : Path
            Path to save the result.
        """
        layer = assert_layer(layer, self.parent_viewer)
        result = layer.metadata.get(SEAM_SEARCH_RESULT, None)
        if not isinstance(result, SeamSearchResult):
            raise TypeError("The layer does not have seam search result.")
        return result.to_dataframe().write_csv(path)

    def _seam_search_input(
        self, layer: MoleculesLayer, npf: int, order: int
    ) -> tuple[SubtomogramLoader, int]:
        parent = self._get_main()
        mole = layer.molecules
        loader = parent.tomogram.get_subtomogram_loader(mole, order=order)
        if npf is None:
            npf = mole.features[Mole.pf].unique().len()
        return loader, npf

    @set_design(text="Save last average", location=STAnalysis)
    def save_last_average(self, path: Path.Save[FileFilter.IMAGE]):
        """Save the lastly generated average image."""
        path = Path(path)
        img = self.last_average
        if img is None:
            raise ValueError(
                "No average image is available. You have to average subtomograms first."
            )
        return img.imsave(path)

    @average_all.started.connect
    @align_averaged.started.connect
    @align_all.started.connect
    @calculate_fsc.started.connect
    def _show_subtomogram_averaging(self):
        return self.show()

    def _extend_pf(
        self,
        mole: Molecules,
        range_long: tuple[float, float],
        angle_max: float,
        min_score: float,
        npf: int | None = None,
        **kwargs,
    ):
        score_hist_0 = list[float]()
        score_hist_1 = list[float]()
        next_mole, cur_fixed = _prep_next_molecules(mole)
        aligned_molecules = list[Molecules]()
        while True:
            try:
                landscape = self._construct_landscape(molecules=next_mole, **kwargs)
            except SubvolumeOutOfBoundError:
                if score_hist_0:
                    score_hist_0.pop()
                break
            if len(score_hist_0) == 0:
                score_hist_0.append(-landscape.energies[(0, *landscape.offset)])
            aligned_mole = landscape.run_viterbi_fixed_start(
                cur_fixed,
                range_long=range_long,
                angle_max=angle_max,
            )
            next_mole, cur_fixed = _prep_next_molecules(aligned_mole)
            aligned_molecules.append(aligned_mole[:1])
            score_hist_1.append(aligned_mole.features["score"][0])
            if score_hist_1[-1] + score_hist_0[-1] < min_score * 2:
                break
            score_hist_0.append(aligned_mole.features["score"][1])

        if len(aligned_molecules) > 1:
            fin = Molecules.concat(aligned_molecules[:-1])
        else:
            fin = Molecules.empty(["score"])
        scores = (np.array(score_hist_0) + np.array(score_hist_1)) / 2
        if npf is not None:
            fin = fin.with_features(pl.repeat(npf, pl.len()).alias(Mole.pf))
        return fin, scores

    @thread_worker.callback
    def _align_all_on_return(
        self, molecules: list[Molecules], old_layers: list[MoleculesLayer]
    ):
        """The return callback function for alignment methods."""
        main = self._get_main()
        new_layers = list[MoleculesLayer]()
        for mole, layer in zip(molecules, old_layers, strict=True):
            points = main.add_molecules(
                mole,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
                source=layer.source_component,
            )
            new_layers.append(points)
            layer.visible = False
            _Logger.print_html(f"{layer.name!r} &#8594; {points.name!r}")
        return self._undo_for_new_layer([l.name for l in old_layers], new_layers)

    @thread_worker.callback
    def _align_on_landscape_on_return(
        self,
        mole: Molecules,
        name: str,
        source=None,
        metadata: dict[str, Any] = {},
    ):
        main = self._get_main()
        if name.startswith(LANDSCAPE_PREFIX):
            nchars = len(LANDSCAPE_PREFIX)
            mole_name = name[nchars:].strip()
        else:
            mole_name = name
        points = main.add_molecules(
            mole,
            name=_coerce_aligned_name(mole_name, self.parent_viewer),
            source=source,
            metadata=metadata,
        )
        _Logger.print_html(f"{name!r} &#8594; {points.name!r}")
        return mole

    def _undo_for_new_layer(
        self,
        old_names: list[str],
        new_layers: list[MoleculesLayer],
    ):
        @undo_callback
        def out():
            main = self._get_main()
            main._try_removing_layers(new_layers)
            for name in old_names:
                if name not in main.parent_viewer.layers:
                    continue
                main.parent_viewer.layers[name].visible = True

        @out.with_redo
        def out():
            main = self._get_main()
            for points in new_layers:
                main.parent_viewer.add_layer(points)

        return out

    @nogui
    @do_not_record
    def get_template(
        self, template_path: str | Path, scale: float | None = None
    ) -> ip.ImgArray:
        """A non-GUI method to get the template"""
        if scale is None:
            scale = self._get_main().tomogram.scale
        img = self.params._norm_template_param(template_path).provide(scale)
        return ip.asarray(img, axes="zyx").set_scale(zyx=scale)

    @nogui
    @do_not_record
    def get_mask(
        self,
        mask_params: Any,
        scale: float | None = None,
        template_path: str | Path | None = None,
    ) -> ip.ImgArray:
        """A non-GUI method to get the mask."""
        if scale is None:
            scale = self._get_main().tomogram.scale
        if isinstance(mask_params, tuple):
            if template_path is None:
                raise ValueError("Template path is required when using soft-Otsu mask.")
            template = self.params._norm_template_param(template_path).provide(scale)
            radius, sigma = mask_params
            mask = pipe.soft_otsu(radius=radius, sigma=sigma).convert(template, scale)
        elif isinstance(mask_params, (str, Path)):
            mask = pipe.from_file(mask_params).provide(scale)
        else:
            raise TypeError(
                f"Cannot create mask image using parameter: {mask_params!r}"
            )
        return ip.asarray(mask, axes="zyx").set_scale(zyx=scale)

    @nogui
    @do_not_record
    def get_subtomograms(
        self,
        layers: str | MoleculesLayer | list[str | MoleculesLayer],
        shape: tuple[nm, nm, nm],
        bin_size: int = 1,
        order: int = 3,
    ) -> "Array":
        """
        A non-GUI method to get all the subtomograms as a dask array.

        Parameters
        ----------
        layers : str, MoleculesLayer or list of them
            All the layers that will be used to construct the subtomogram array.
        shape : (nm, nm, nm)
            Shape of output subtomograms.
        bin_size : int, default
            Bin size of the subtomograms.
        order : int, default 3
            Interpolation order.

        Returns
        -------
        Array
            4D Dask array.
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        tomo = parent.tomogram
        loader = tomo.get_subtomogram_loader(
            _concat_molecules(layers), shape, binsize=bin_size, order=order
        )
        return loader.construct_dask()

    def _get_simple_annealing_model(self, layer: MoleculesLayer):
        # TODO: This method should finally be moved to some utils module since
        # this analysis is independent of annealing. Currently annealing and
        # graph construction cannot be separated.
        parent = self._get_main()
        scale = parent.tomogram.scale
        return _annealing.get_annealing_model(
            layer.molecules,
            layer.source_spline,
            (0, 0, 0),
            scale,
        )

    def _construct_landscape(
        self,
        molecules: Molecules,
        template_path: Any,
        mask_params=None,
        max_shifts: tuple[nm, nm, nm] = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: float = 0.5,
        order: int = 3,
        upsample_factor: int = 5,
        bin_size: int = 1,
    ):  # fmt: skip
        parent = self._get_main()
        loader = parent.tomogram.get_subtomogram_loader(
            molecules, binsize=bin_size, order=order
        )
        return Landscape.from_loader(
            loader=loader,
            template=template_path,
            mask=self.params._get_mask(params=mask_params),
            max_shifts=max_shifts,
            upsample_factor=upsample_factor,
            alignment_model=alignment.ZNCCAlignment.with_params(
                rotations=rotations,
                cutoff=cutoff,
                tilt=parent.tomogram.tilt_model,
            ),
        ).normed()


def _prep_next_molecules(m: Molecules) -> tuple[Molecules, np.ndarray]:
    fixed, p0, rot = m.pos[-2], m.pos[-1], m.rotator[-1:]
    rot2 = Rotation.concatenate([rot, rot])
    return Molecules(np.stack([p0, p0 * 2 - fixed]), rot2), fixed


def _coerce_aligned_name(name: str, viewer: "napari.Viewer"):
    num = 1
    if re.match(rf".*-{ALN_SUFFIX}(\d)+$", name):
        try:
            name, suf = name.rsplit(f"-{ALN_SUFFIX}", 1)
            num = int(suf) + 1
        except Exception:
            num = 1

    existing_names = {layer.name for layer in viewer.layers}
    while name + f"-{ALN_SUFFIX}{num}" in existing_names:
        num += 1
    return name + f"-{ALN_SUFFIX}{num}"


def _concat_molecules(layers: MoleculesLayersType) -> Molecules:
    return Molecules.concat([l.molecules for l in layers])


def _avg_name(layers: MoleculesLayersType) -> str:
    if len(layers) == 1:
        name = layers[0].name
    else:
        name = f"{len(layers)}-layers"
    return name


def _get_slice_for_average_subset(method: str, nmole: int, number: int):
    if nmole < number:
        raise ValueError(f"There are only {nmole} subtomograms.")
    match method:
        case "steps":
            step = nmole // number
            sl = slice(0, step * number, step)
        case "first":
            sl = slice(0, number)
        case "last":
            sl = slice(-number, -1)
        case "random":
            sl_all = np.arange(nmole, dtype=np.uint32)
            np.random.shuffle(sl_all)
            sl = sl_all[:number]
        case _:  # pragma: no cover
            raise ValueError(f"method {method!r} not supported.")
    return sl


def _assert_landscape_layer(layer, viewer: napari.Viewer) -> LandscapeSurface:
    if isinstance(layer, str):
        layer = viewer.layers[layer]
    if isinstance(layer, LandscapeSurface):
        return layer
    else:
        raise TypeError(f"Layer {layer.name!r} is not a landscape layer.")


def _default_align_averaged_shifts(mole: Molecules) -> "NDArray[np.floating]":
    npf = mole.features[Mole.pf].max() + 1
    dy = np.sqrt(np.sum((mole.pos[0] - mole.pos[1]) ** 2))  # axial shift
    dx = np.sqrt(np.sum((mole.pos[0] - mole.pos[npf]) ** 2))  # lateral shift
    return (dy * 0.6, dy * 0.6, dx * 0.6)


def _define_correlation_function(
    temp: "NDArray[np.float32]",
    mask: "NDArray[np.float32] | float",
    func: Callable[[ip.ImgArray, ip.ImgArray], float],
) -> "Callable[[NDArray[np.float32]], float]":
    temp = ip.asarray(temp, axes="zyx")

    def _fn(img: "NDArray[np.float32]") -> float:
        corr = float(func(ip.asarray(img * mask, axes="zyx"), temp * mask))
        return corr

    return _fn


def _update_mole_pos(new: Molecules, old: Molecules, spl: "CylSpline") -> Molecules:
    """
    Update the "position-nm" feature of molecules.

    Feature "position-nm" is the coordinate of molecules along the source spline.
    After alignment, this feature should be updated accordingly. This fucntion
    will do this.
    """
    if Mole.position not in old.features.columns:
        return new
    _u = spl.y_to_position(old.features[Mole.position])
    vec = spl.map(_u, der=1)  # the tangent vector of the spline
    vec_norm = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    dy = np.sum((new.pos - old.pos) * vec_norm, axis=1)
    return new.with_features(pl.col(Mole.position) + dy)


def _update_offset(spl: "CylSpline", dr: tuple[nm, nm, nm], radius: nm):
    _dz, _dy, _dx = dr
    _offset_y = _dy
    _offset_r = np.sqrt((_dz + radius) ** 2 + _dx**2) - radius
    _offset_a = np.arctan2(_dx, _dz + radius)
    if spl.orientation is Ori.PlusToMinus:
        _offset_y = -_offset_y
        _offset_a = -_offset_a
    if spl.props.has_glob(H.offset_axial):
        _offset_y += spl.props.get_glob(H.offset_axial)
    if spl.props.has_glob(H.offset_angular):
        _offset_a += spl.props.get_glob(H.offset_angular)
    if spl.props.has_glob(H.offset_radial):
        _offset_r += spl.props.get_glob(H.offset_radial)
    return spl.props.glob.with_columns(
        pl.Series(H.offset_axial, [_offset_y], dtype=pl.Float32),
        pl.Series(H.offset_angular, [_offset_a], dtype=pl.Float32),
        pl.Series(H.offset_radial, [_offset_r], dtype=pl.Float32),
    )


class MoleculesCombiner:
    """Class to split/combine molecules for batch analysis."""

    def __init__(self, identifier: str = ".molecule_object_id"):
        self._identifier = identifier

    def concat(self, molecules: "Molecules | Iterable[Molecules]") -> Molecules:
        if isinstance(molecules, Molecules):
            return molecules
        inputs = list[Molecules]()
        for i, mole in enumerate(molecules):
            inputs.append(
                mole.with_features(
                    pl.Series(self._identifier, np.full(len(mole), i, dtype=np.uint32))
                )
            )
        return Molecules.concat(inputs)

    def split(
        self, molecules: Molecules, layers: list[MoleculesLayer]
    ) -> list[Molecules]:
        if self._identifier not in molecules.features.columns:
            return molecules
        out = list[Molecules]()
        for i, (_, mole) in enumerate(molecules.groupby(self._identifier)):
            mole0 = mole.drop_features(self._identifier)
            layer = layers[i]
            if spl := layer.source_spline:
                mole0 = _update_mole_pos(mole0, layer.molecules, spl)
            out.append(mole0)
        return out


impl_preview(SubtomogramAveraging.align_all_annealing, text="Preview molecule network")(
    _annealing.preview_single
)
impl_preview(
    SubtomogramAveraging.run_annealing_on_landscape, text="Preview molecule network"
)(_annealing.preview_landscape_function)
