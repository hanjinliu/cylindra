from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Sequence,
    TYPE_CHECKING,
    Annotated,
)
import re
from scipy.spatial.transform import Rotation
from magicclass import (
    do_not_record,
    magicclass,
    magicmenu,
    field,
    magictoolbar,
    nogui,
    vfield,
    MagicTemplate,
    set_design,
    impl_preview,
    abstractapi,
)
from magicclass.widgets import HistoryFileEdit, Separator
from magicclass.types import Optional, Path, ExprStr
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.undo import undo_callback
from magicclass.ext.dask import dask_thread_worker as dask_worker

from acryo import Molecules, SubtomogramLoader, alignment, pipe

import numpy as np
import impy as ip
import polars as pl
import napari

from cylindra import utils, _config, cylstructure
from cylindra.types import MoleculesLayer
from cylindra.const import (
    ALN_SUFFIX,
    SEAM_SEARCH_RESULT,
    ANNEALING_RESULT,
    INTERPOLATION_CHOICES,
    MoleculesHeader as Mole,
    nm,
    PropertyNames as H,
    Ori,
    FileFilter,
)
from cylindra.widgets._widget_ext import (
    RotationsEdit,
    RandomSeedEdit,
    MultiFileEdit,
)
from cylindra.components.landscape import Landscape
from cylindra.components.seam_search import (
    CorrelationSeamSearcher,
    BooleanSeamSearcher,
    ManualSeamSearcher,
    SeamSearchResult,
)

from ._annotated import (
    MoleculesLayerType,
    MoleculesLayersType,
    FSCFreq,
    assert_layer,
    assert_list_of_layers,
)
from .widget_utils import capitalize, timer, POLARS_NAMESPACE
from .subwidgets._child_widget import ChildWidget
from . import widget_utils, _shared_doc, _progress_desc as _pdesc, _annealing

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from napari.layers import Image
    from cylindra.components import CylSpline
    from cylindra.components.landscape import AnnealingResult


def _get_template_shape(self: "SubtomogramAveraging", size: nm) -> list[str]:
    if size is None:
        tmp = self.template
        size = max(tmp.shape) * tmp.scale.x
    return size


# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_Rotations = Annotated[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    {"widget_type": RotationsEdit},
]
_MaxShifts = Annotated[
    tuple[nm, nm, nm],
    {"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"},
]
_SubVolumeSize = Annotated[
    Optional[nm],
    {
        "text": "Use template shape",
        "options": {"value": 12.0, "max": 100.0},
        "label": "size (nm)",
        "validator": _get_template_shape,
    },
]
_ImagePaths = Annotated[
    list[Path], {"widget_type": MultiFileEdit, "filter": FileFilter.IMAGE}
]
_DistRangeLon = Annotated[
    tuple[nm, nm],
    {
        "options": {"min": 0.1, "max": 1000.0, "step": 0.05},
        "label": "Longitudinal range (nm)",
    },
]
_DistRangeLat = Annotated[
    tuple[nm, nm],
    {
        "options": {"min": 0.1, "max": 1000.0, "step": 0.05},
        "label": "Lateral range (nm)",
    },
]
_AngleMaxLon = Annotated[
    float, {"max": 90.0, "step": 0.5, "label": "Maximum angle (deg)"}
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


MASK_CHOICES = ("No mask", "Blur template", "From file")


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

    _get_choice.__qualname__ = "CylindraMainWidget.sta._get_choice"
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
    align_all_multi_template = abstractapi()

    @magicmenu
    class Constrained(MagicTemplate):
        align_all_viterbi = abstractapi()
        align_all_viterbi_multi_template = abstractapi()
        align_all_annealing = abstractapi()
        align_all_annealing_multi_template = abstractapi()
        save_annealing_scores = abstractapi()


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

    template_path = field(
        Optional[
            Annotated[Path.Read[FileFilter.IMAGE], {"widget_type": HistoryFileEdit}]
        ],
        label="Template",
    ).with_options(text="Use last averaged image", value=Path(""))
    mask_choice = vfield(label="Mask", record=False).with_choices(MASK_CHOICES)
    params = field(MaskParameters, name="Parameters", label="")
    mask_path = field(mask_path)

    _last_average: ip.ImgArray | None = None  # the global average result
    _viewer: "napari.Viewer | None" = None

    def __post_init__(self):
        self._template: ip.ImgArray = None
        self.mask_choice = MASK_CHOICES[0]

        # load history
        line: HistoryFileEdit = self.template_path.inner_widget
        for fp in _config.get_template_path_hist():
            line.append_history(str(fp))

    @mask_choice.connect
    def _on_mask_switch(self):
        v = self.mask_choice
        self.params.visible = v == MASK_CHOICES[1]
        self.mask_path.visible = v == MASK_CHOICES[2]

    def _save_history(self):
        try:
            line: HistoryFileEdit = self.template_path.inner_widget
            _hist = [str(p) for p in line.get_history()[-20:]]
            _config.set_template_path_hist(_hist)
        except Exception:
            pass

    def _get_template(self, path: Path | None = None, allow_none: bool = False):
        if path is None:
            path = self.template_path.value
        self._save_history()

        if path is None:
            if StaParameters._last_average is None:
                if allow_none:
                    return None
                raise ValueError(
                    "No average image found. You can uncheck 'Use last averaged image' and select "
                    "a template image from a file."
                )
            provider = pipe.from_array(
                StaParameters._last_average, StaParameters._last_average.scale.x
            )
        else:
            path = Path(path)
            if path.is_dir():
                if allow_none:
                    return None
                raise TypeError(f"Template image must be a file, got {path}.")
            provider = pipe.from_file(path)
        return provider

    def _get_mask_params(self, params=None) -> str | tuple[nm, nm] | None:
        v = self.mask_choice
        if v == MASK_CHOICES[0]:
            params = None
        elif v == MASK_CHOICES[1]:
            params = (self.params.dilate_radius, self.params.sigma)
        else:
            params = self.mask_path.mask_path
        return params

    def _set_mask_params(self, params):
        if params is None:
            self.mask_choice = MASK_CHOICES[0]
        elif isinstance(params, (tuple, list, np.ndarray)):
            self.mask_choice = MASK_CHOICES[1]
            self.params.dilate_radius, self.params.sigma = params
        else:
            self.mask_choice = MASK_CHOICES[2]
            self.mask_path.mask_path = params

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

            StaParameters._viewer = viewer = napari.Viewer(
                title=name, axis_labels=("z", "y", "x"), ndisplay=3
            )
            volume_menu = Volume()
            viewer.window.add_dock_widget(volume_menu)
            viewer.window.resize(10, 10)
            viewer.window.activate()
        StaParameters._viewer.scale_bar.visible = True
        StaParameters._viewer.scale_bar.unit = "nm"
        if store:
            StaParameters._last_average = image
        return StaParameters._viewer.add_image(
            image,
            scale=image.scale,
            name=name,
            rendering="iso",
            iso_threshold=threshold_yen(image.value),
            blending="opaque",
        )


@magicclass
@_shared_doc.update_cls
class SubtomogramAveraging(ChildWidget):
    """Widget for subtomogram averaging."""

    AveragingMenu = field(Averaging, name="Averaging")
    STAnalysisMenu = field(STAnalysis, name="Analysis")
    AlignmentMenu = field(Alignment, name="Alignment")
    params = field(StaParameters)

    @property
    def sub_viewer(self):
        """The napari viewer for subtomogram averaging."""
        return StaParameters._viewer

    def _get_template_path(self, *_):
        return self.params.template_path.value

    def _get_mask_params(self, *_):
        return self.params._get_mask_params()

    @magictoolbar
    class STATools(MagicTemplate):
        show_template = abstractapi()
        show_template_original = abstractapi()
        show_mask = abstractapi()

    @set_design(icon="ic:baseline-view-in-ar", location=STATools)
    @do_not_record
    def show_template(self):
        """Load and show template image in the scale of the tomogram."""
        template = self.template
        if template is None:
            raise ValueError("No template to show.")
        self._show_rec(template, name="Template image", store=False)

    @set_design(icon="material-symbols:view-in-ar", location=STATools)
    @do_not_record
    def show_template_original(self):
        """Load and show template image in the original scale."""
        path = self.params.template_path.value
        if path is None:
            return self.show_template()
        if path.is_dir():
            raise TypeError(f"Template image must be a file, got {path}.")
        template = ip.imread(path)
        self._show_rec(template, name="Template image", store=False)

    @set_design(icon="fluent:shape-organic-20-filled", location=STATools)
    @do_not_record
    def show_mask(self):
        """Load and show mask image in the scale of the tomogram."""
        mask = self.mask
        if mask is None:
            raise ValueError("No mask to show.")
        self._show_rec(mask, name="Mask image", store=False)

    @property
    def template(self) -> "ip.ImgArray | None":
        """Template image."""
        loader = self._get_loader(binsize=1, molecules=Molecules.empty())
        template, _ = loader.normalize_input(self.params._get_template())
        scale = loader.scale
        return ip.asarray(template, axes="zyx").set_scale(zyx=scale, unit="nm")

    @property
    def mask(self) -> "ip.ImgArray | None":
        """Mask image."""
        loader = self._get_loader(binsize=1, molecules=Molecules.empty())
        _, mask = loader.normalize_input(
            self.params._get_template(allow_none=True), self.params._get_mask()
        )
        if mask is None:
            return None
        return ip.asarray(mask, axes="zyx").set_scale(zyx=loader.scale, unit="nm")

    @property
    def last_average(self) -> "ip.ImgArray | None":
        """Last averaged image if exists."""
        return StaParameters._last_average

    def _get_shape_in_nm(self, default: int = None) -> tuple[nm, nm, nm]:
        if default is None:
            tmp = self.template
            return tuple(np.array(tmp.shape) * tmp.scale.x)
        else:
            return (default,) * 3

    def _set_mask_params(self, params):
        return self.params._set_mask_params(params)

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
        Subtomogram averaging using all the molecules in the selected layer.

        Parameters
        ----------
        {layers}{size}{interpolation}{bin_size}
        """
        t0 = timer("average_all")
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

        Parameters
        ----------
        {layers}{size}
        method : str, optional
            How to choose subtomogram subset.
            (1) steps: Each 'steps' subtomograms from the tip of spline.
            (2) first: First subtomograms.
            (3) last: Last subtomograms.
            (4) random: choose randomly.
        number : int, default is 64
            Number of subtomograms to use.
        {bin_size}
        """
        t0 = timer("average_subset")
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
        by: ExprStr.In[POLARS_NAMESPACE] = "pl.col('pf-id')",
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
        t0 = timer("average_groups")
        layers = assert_list_of_layers(layers, self.parent_viewer)
        if isinstance(by, pl.Expr):
            expr = by
        else:
            expr = ExprStr(by, POLARS_NAMESPACE).eval()
        parent = self._get_main()
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            _concat_molecules(layers), shape, binsize=bin_size, order=interpolation
        )
        avgs = np.stack(list(loader.groupby(expr).average().values()), axis=0)
        img = ip.asarray(avgs, axes="pzyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{_avg_name(layers)}", store=False)

    @set_design(text="Split molecules and average", location=Averaging)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Split-and-averaging of {!r}"))  # fmt: skip
    def split_and_average(
        self,
        layers: MoleculesLayersType,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Split molecules into two groups and average separately.

        Parameters
        ----------
        {layers}
        n_set : int, default is 1
            How many pairs of average will be calculated.
        {size}{interpolation}{bin_size}
        """
        t0 = timer("split_and_average")
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        molecules = _concat_molecules(layers)
        shape = self._get_shape_in_nm(size)
        loader = parent.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation
        )
        axes = "ipzyx" if n_set > 1 else "pzyx"
        img = ip.asarray(loader.average_split(n_set=n_set), axes=axes)
        img.set_scale(zyx=loader.scale)
        t0.toc()
        return self._show_rec.with_args(img, f"[Split]{_avg_name(layers)}")

    @set_design(text="Align average to template", location=Alignment)
    @dask_worker.with_progress(descs=_pdesc.align_averaged_fmt)
    def align_averaged(
        self,
        layers: MoleculesLayersType,
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
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
        t0 = timer("align_averaged")
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
            template=self.params._get_template(path=template_path),
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
            # TODO: Undo cannot catch this change. Need to fix.
            if spl := layer.source_spline:
                _mole_trans = _update_mole_pos(_mole_trans, mole, spl)
                if spl.radius is None:
                    _radius: nm = cylstructure.calc_radius(mole, spl).mean()
                else:
                    _radius = spl.radius
                _dz, _dy, _dx = rotator.apply(svec)
                _offset_y = _dy
                _offset_r = np.sqrt((_dz + _radius) ** 2 + _dx**2) - _radius
                _offset_a = np.arctan2(_dx, _dz + _radius)
                if spl.orientation is Ori.PlusToMinus:
                    _offset_y = -_offset_y
                    _offset_a = -_offset_a
                if spl.props.has_glob(H.offset_axial):
                    _offset_y += spl.props.get_glob(H.offset_axial)
                if spl.props.has_glob(H.offset_angular):
                    _offset_a += spl.props.get_glob(H.offset_angular)
                if spl.props.has_glob(H.offset_radial):
                    _offset_r += spl.props.get_glob(H.offset_radial)
                spl.props.glob = spl.props.glob.with_columns(
                    pl.Series(H.offset_axial, [_offset_y], dtype=pl.Float32),
                    pl.Series(H.offset_angular, [_offset_a], dtype=pl.Float32),
                    pl.Series(H.offset_radial, [_offset_r], dtype=pl.Float32),
                )

            yield _on_yield.with_args(_mole_trans, layer)

            # create images for visualization in the logger. Image is magenta, template is green
            img_norm = utils.normalize_image(_img_trans)
            merge = np.stack([img_norm, temp_norm, img_norm], axis=-1)
            with _Logger.set_plt():
                widget_utils.plot_projections(merge)

            # logging
            rvec = rotator.as_rotvec()
            _Logger.print_table(
                [
                    ["", "X", "Y", "Z"],
                    ["Shift (nm)", f"{svec[2]:2f}", f"{svec[1]:2f}", f"{svec[0]:2f}"],
                    ["Rot vector", f"{rvec[2]:2f}", f"{rvec[1]:2f}", f"{rvec[0]:2f}"],
                ],
                header=False,
                index=False,
            )

        t0.toc()

        @thread_worker.callback
        def _align_averaged_on_return():
            return (
                undo_callback(parent._try_removing_layers)
                .with_args(new_layers)
                .with_redo(parent._add_layers_future(new_layers))
            )

        return _align_averaged_on_return

    sep0 = field(Separator)

    @set_design(text="Align all molecules", location=Alignment)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Alignment of {}"))
    def align_all(
        self,
        layers: MoleculesLayersType,
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
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
        {layers}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}{interpolation}{method}{bin_size}
        """
        t0 = timer("align_all")
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()

        combiner = MoleculesCombiner()

        aligned_loader = self._get_loader(
            binsize=bin_size,
            molecules=combiner.concat(layer.molecules for layer in layers),
            order=interpolation,
        ).align(
            template=self.params._get_template(path=template_path),
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
        Calculate the subtomogram average and use it as the template for the alignment.

        Parameters
        ----------
        {layers}{mask_params}{size}{max_shifts}{rotations}{cutoff}{interpolation}{method}{bin_size}
        """
        t0 = timer("align_all_template_free")
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        combiner = MoleculesCombiner()
        molecules = combiner.concat(layer.molecules for layer in layers)
        mask = self.params._get_mask(params=mask_params)
        if size is None:
            shape = None
            raise NotImplementedError("'size' must be given.")
        else:
            shape = tuple(
                parent.tomogram.nm2pixel(self._get_shape_in_nm(size), binsize=bin_size)
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
                tilt=parent.tomogram.tilt_model,
            )
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    @set_design(text="Align all (multi-template)", location=Alignment)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Multi-template alignment of {}"))  # fmt: skip
    def align_all_multi_template(
        self,
        layers: MoleculesLayersType,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):  # fmt: skip
        """
        Align all the input template images to all the molecules.

        Parameters
        ----------
        {layers}{template_paths}{mask_params}{max_shifts}{rotations}{cutoff}{interpolation}{method}{bin_size}
        """
        t0 = timer("align_all_multi_template")
        layers = assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_main()
        combiner = MoleculesCombiner()
        molecules = combiner.concat(layer.molecules for layer in layers)
        templates = pipe.from_files(template_paths)

        aligned_loader = self._get_loader(
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
        ).align_multi_templates(
            templates=templates,
            mask=self.params._get_mask(params=mask_params),
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            alignment_model=_get_alignment(method),
            tilt=parent.tomogram.tilt_model,
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    sep1 = field(Separator)

    @set_design(text="Viterbi Alignment", location=Alignment.Constrained)
    @dask_worker.with_progress(descs=_pdesc.align_viterbi_fmt)
    def align_all_viterbi(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range: _DistRangeLon = (4.0, 4.28),
        angle_max: Optional[float] = 5.0,
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
        {layer}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}{interpolation}
        distance_range : tuple of float
            Range of allowed distance between monomers.
        {angle_max}{upsample_factor}
        """
        kwargs = locals()
        kwargs.pop("self")
        t0 = timer("align_all_viterbi")
        out = yield from self._align_all_viterbi(**kwargs)
        t0.toc()
        return out

    @set_design(
        text="Viterbi Alignment (multi-template)", location=Alignment.Constrained
    )
    @dask_worker.with_progress(descs=_pdesc.align_viterbi_fmt)
    def align_all_viterbi_multi_template(
        self,
        layer: MoleculesLayerType,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range: _DistRangeLon = (4.0, 4.28),
        angle_max: Optional[float] = 5.0,
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
        {layer}{template_paths}{mask_params}{max_shifts}{rotations}{cutoff}{interpolation}
        distance_range : (float, float)
            Range of allowed distance between monomers.
        {angle_max}{upsample_factor}
        """
        kwargs = locals()
        kwargs.setdefault("template_path", kwargs.pop("template_paths"))
        kwargs.pop("self")
        t0 = timer("align_all_viterbi_multi_template")
        out = yield from self._align_all_viterbi(**kwargs)
        t0.toc()
        return out

    @set_design(text="Simulated annealing", location=Alignment.Constrained)
    @dask_worker.with_progress(descs=_pdesc.align_annealing_fmt)
    def align_all_annealing(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range_long: _DistRangeLon = (4.0, 4.28),
        distance_range_lat: _DistRangeLat = (5.1, 5.3),
        angle_max: _AngleMaxLon = 5.0,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        random_seeds: _RandomSeeds = (0, 1, 2, 3, 4),
        return_all: Annotated[bool, {"label": "Return all the annealing results"}] = False,
    ):  # fmt: skip
        """
        2D-constrained subtomogram alignment using simulated annealing.

        This alignment method considers the distance between every adjacent monomers.
        Two-dimensionally connected optimization can be approximated by the simulated
        annealing algorithm.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}{interpolation}
        distance_range_long : tuple of float
            Range of allowed distance between longitudianlly consecutive monomers.
        distance_range_lat : tuple of float
            Range of allowed distance between laterally consecutive monomers.
        {angle_max}{temperature_time_const}{upsample_factor}{random_seeds}
        return_all : bool, default is False
            If True, return all the annealing results for each random seed.
        """
        kwargs = locals()
        kwargs.pop("self")
        t0 = timer("align_all_annealing")
        out = yield from self._align_all_annealing(**kwargs)
        t0.toc()
        return out

    @set_design(
        text="Simulated annealing (multi-template)", location=Alignment.Constrained
    )
    @dask_worker.with_progress(descs=_pdesc.align_annealing_fmt)
    def align_all_annealing_multi_template(
        self,
        layer: MoleculesLayerType,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range_long: _DistRangeLon = (4.0, 4.28),
        distance_range_lat: _DistRangeLat = (5.1, 5.3),
        angle_max: _AngleMaxLon = 5.0,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        random_seeds: _RandomSeeds = (0, 1, 2, 3, 4),
        return_all: Annotated[bool, {"label": "Return all the annealing results"}] = False,
    ):  # fmt: skip
        """
        2D-constrained subtomogram alignment using simulated annealing.

        This alignment method considers the distance between every adjacent monomers.
        Two-dimensionally connected optimization can be approximated by the simulated
        annealing algorithm.

        Parameters
        ----------
        {layer}{template_paths}{mask_params}{max_shifts}{rotations}{cutoff}{interpolation}
        distance_range_long : tuple of float
            Range of allowed distance between longitudianlly consecutive monomers.
        distance_range_lat : tuple of float
            Range of allowed distance between laterally consecutive monomers.
        {angle_max}{temperature_time_const}{upsample_factor}{random_seeds}
        """
        kwargs = locals()
        kwargs.setdefault("template_path", kwargs.pop("template_paths"))
        kwargs.pop("self")
        t0 = timer("align_all_annealing_multi_template")
        out = yield from self._align_all_annealing(**kwargs)
        t0.toc()
        return out

    @nogui
    @do_not_record
    def construct_landscape(
        self,
        layer: MoleculesLayer,
        template_path: Any,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        max_shifts: tuple[nm, nm, nm] = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: float = 0.5,
        interpolation: int = 3,
        upsample_factor: int = 5,
    ):  # fmt: skip
        """
        Construct a landscape for subtomogram alignment.

        Parameters
        ----------
        layer : MoleculesLayer
            Layer to construct the landscape.
        template_path : template input type
            Template(s) used for landscape construction.
        mask_params : make input type
            Parameters used to create a mask.
        max_shifts : (float, float, float), default is (0.8, 0.8, 0.8)
            Maximum shift in nm.
        rotations : _Rotations, optional
            Rotation ranges of the template in degrees.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter.
        interpolation : int, default is 3
            Interpolation order.
        upsample_factor : int, default is 5
            Upsampling factor of the landscape.

        Returns
        -------
        Landscape
            The landscape instance.
        """
        parent = self._get_main()
        layer = assert_layer(layer, self.parent_viewer)
        return Landscape.from_loader(
            loader=parent.tomogram.get_subtomogram_loader(
                layer.molecules, order=interpolation
            ),
            template=template_path,
            mask=self.params._get_mask(params=mask_params),
            max_shifts=max_shifts,
            upsample_factor=upsample_factor,
            alignment_model=alignment.ZNCCAlignment.with_params(
                rotations=rotations,
                cutoff=cutoff,
                tilt=parent.tomogram.tilt_model,
            ),
        )

    def _align_all_viterbi(
        self,
        layer: MoleculesLayer,
        template_path: "Path | list[Path]",
        mask_params=None,
        max_shifts: tuple[nm, nm, nm] = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: float = 0.5,
        interpolation: int = 3,
        distance_range: tuple[nm, nm] = (4.0, 4.28),
        angle_max: "float | None" = 5.0,
        upsample_factor: int = 5,
    ):
        from cylindra._dask import delayed, compute

        parent = self._get_main()
        layer = assert_layer(layer, self.parent_viewer)
        landscape = self.construct_landscape(
            layer=layer,
            template_path=template_path,
            mask_params=mask_params,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            interpolation=interpolation,
            upsample_factor=upsample_factor,
        )

        yield
        max_shifts_px = tuple(s / parent.tomogram.scale for s in max_shifts)
        mole = layer.molecules
        npfs: Sequence[int] = mole.features[Mole.pf].unique(maintain_order=True)

        slices = [(mole.features[Mole.pf] == i).to_numpy() for i in npfs]
        viterbi_tasks = [
            delayed(landscape[sl].run_viterbi)(distance_range, angle_max)
            for sl in slices
        ]
        vit_out = compute(*viterbi_tasks)

        inds = np.empty((mole.count(), 3), dtype=np.int32)
        for i, result in enumerate(vit_out):
            inds[slices[i], :] = _check_viterbi_shift(result.indices, max_shifts_px, i)
        molecules_opt = landscape.transform_molecules(mole, inds)
        if spl := layer.source_spline:
            molecules_opt = _update_mole_pos(molecules_opt, mole, spl)
        return self._align_all_on_return.with_args([molecules_opt], [layer])

    def _align_all_annealing(
        self,
        layer: MoleculesLayer,
        template_path: "Path | list[Path]",
        mask_params=None,
        max_shifts: tuple[nm, nm, nm] = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: float = 0.5,
        interpolation: int = 3,
        distance_range_long: tuple[nm, nm] = (4.0, 4.28),
        distance_range_lat: tuple[nm, nm] = (5.1, 5.3),
        angle_max: float = 5.0,
        temperature_time_const: float = 1.0,
        upsample_factor: int = 5,
        random_seeds: Iterable[int] = range(5),
        return_all: bool = False,
    ):
        parent = self._get_main()
        layer = assert_layer(layer, self.parent_viewer)
        landscape = self.construct_landscape(
            layer=layer,
            template_path=template_path,
            mask_params=mask_params,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            interpolation=interpolation,
            upsample_factor=upsample_factor,
        )
        yield
        results = landscape.run_annealing(
            layer.source_spline,
            distance_range_long,
            distance_range_lat,
            angle_max,
            temperature_time_const=temperature_time_const,
            random_seeds=random_seeds,
        )
        if all(result.state == "failed" for result in results):
            raise RuntimeError(
                "Failed to optimize for all trials. You may check the distance range."
            )
        elif not any(result.state == "converged" for result in results):
            _Logger.print("Optimization did not converge for any trial.")

        _Logger.print_table(
            {
                "Iteration": [r.niter for r in results],
                "Score": [-float(r.energies[-1]) for r in results],
                "State": [r.state for r in results],
            }
        )
        results = sorted(results, key=lambda r: r.energies[-1])

        @thread_worker.callback
        def _on_return():
            if return_all:
                point_layers = []
                for i, result in enumerate(results):
                    mole_opt = landscape.transform_molecules(
                        layer.molecules, result.indices
                    )
                    if spl := layer.source_spline:
                        mole_opt = _update_mole_pos(mole_opt, layer.molecules, spl)
                    points = parent.add_molecules(
                        mole_opt,
                        name=f"{_coerce_aligned_name(layer.name, self.parent_viewer)} [{i}]",
                        source=layer.source_component,
                        metadata={ANNEALING_RESULT: result},
                    )
                    point_layers.append(points)
            else:
                mole_opt = landscape.transform_molecules(
                    layer.molecules, results[0].indices
                )
                if spl := layer.source_spline:
                    mole_opt = _update_mole_pos(mole_opt, layer.molecules, spl)
                points = parent.add_molecules(
                    mole_opt,
                    name=_coerce_aligned_name(layer.name, self.parent_viewer),
                    source=layer.source_component,
                    metadata={ANNEALING_RESULT: results[0]},
                )
                point_layers = [points]
            layer.visible = False
            with _Logger.set_plt():
                _annealing.plot_annealing_result(results)

            @undo_callback
            def out():
                parent._try_removing_layers(point_layers)
                layer.visible = True

            @out.with_redo
            def out():
                for points in point_layers:
                    parent.parent_viewer.add_layer(points)

            return out

        return _on_return

    def _get_layers_with_annealing_result(self, *_) -> list[MoleculesLayer]:
        if self.parent_viewer is None:
            return []
        return [
            (layer.name, layer)
            for layer in self.parent_viewer.layers
            if ANNEALING_RESULT in layer.metadata
        ]

    @set_design(text=capitalize, location=Alignment.Constrained)
    @do_not_record
    def save_annealing_scores(
        self,
        layer: Annotated[MoleculesLayer, {"choices": _get_layers_with_annealing_result}],
        path: Path.Save[FileFilter.CSV],
    ):  # fmt: skip
        layer = assert_layer(layer, self.parent_viewer)
        try:
            result: AnnealingResult = layer.metadata[ANNEALING_RESULT]
        except KeyError:
            raise ValueError(
                f"Layer {layer!r} does not have annealing result."
            ) from None
        x = result.batch_size * np.arange(result.energies.size)
        df = pl.DataFrame({"iteration": x, "score": -result.energies})
        return df.write_csv(path, has_header=False)

    @set_design(text=capitalize, location=STAnalysis)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Calculating correlations of {!r}"))  # fmt: skip
    def calculate_correlation(
        self,
        layers: MoleculesLayersType,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
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
        {layers}{template_paths}{mask_params}{interpolation}
        metric : str, default is "zncc"
            Metric to calculate correlation.
        column_prefix : str, default is "score"
            Prefix of the column names of the calculated correlations.
        """
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()
        scale = main.tomogram.scale
        tmps = []
        _shapes = set()
        if isinstance(template_paths, (Path, str)):
            template_paths = [template_paths]
        for path in template_paths:
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
            ).apply(
                funcs,
                schema=[f"{column_prefix}_{i}" for i in range(len(template_paths))],
            )
            layer.molecules = layer.molecules.with_features(out.cast(pl.Float32))
        return None

    @set_design(text="Calculate FSC", location=STAnalysis)
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Calculating FSC of {!r}"))
    def calculate_fsc(
        self,
        layers: MoleculesLayersType,
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: FSCFreq = None,
    ):
        """
        Calculate Fourier Shell Correlation using the selected monomer layer.

        Parameters
        ----------
        {layers}{mask_params}{size}
        seed : int, optional
            Random seed used for subtomogram sampling.
        {interpolation}
        n_set : int, default is 1
            How many sets of image pairs will be generated to average FSC.
        show_average : bool, default is True
            If true, subtomogram average will be shown after FSC calculation.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        t0 = timer("calculate_fsc")
        layers = assert_list_of_layers(layers, self.parent_viewer)
        main = self._get_main()
        mole = _concat_molecules(layers)

        loader = main.tomogram.get_subtomogram_loader(mole, order=interpolation)
        template, mask = loader.normalize_input(
            template=self.params._get_template(allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        fsc, avg = loader.reshape(
            template=template if size is None else None,
            mask=mask,
            shape=None if size is None else (main.tomogram.nm2pixel(size),) * 3,
        ).fsc_with_average(mask=mask, seed=seed, n_set=n_set, dfreq=dfreq)

        if show_average:
            img_avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale)
        else:
            img_avg = None

        freq = fsc["freq"].to_numpy()
        fsc_all = fsc.select(pl.col("^FSC.*$")).to_numpy()
        fsc_mean = np.mean(fsc_all, axis=1)
        fsc_std = np.std(fsc_all, axis=1)
        crit_0143, crit_0500 = 0.143, 0.500
        res0143 = widget_utils.calc_resolution(freq, fsc_mean, crit_0143, loader.scale)
        res0500 = widget_utils.calc_resolution(freq, fsc_mean, crit_0500, loader.scale)
        t0.toc()

        @thread_worker.callback
        def _calculate_fsc_on_return():
            _name = _avg_name(layers)
            _Logger.print_html(f"<b>Fourier Shell Correlation of {_name!r}</b>")
            with _Logger.set_plt():
                widget_utils.plot_fsc(
                    freq,
                    fsc_mean,
                    fsc_std,
                    [crit_0143, crit_0500],
                    main.tomogram.scale,
                )

            _Logger.print_html(f"Resolution at FSC=0.5 ... <b>{res0500:.3f} nm</b>")
            _Logger.print_html(f"Resolution at FSC=0.143 ... <b>{res0143:.3f} nm</b>")

            if img_avg is not None:
                _rec_layer: "Image" = self._show_rec(
                    img_avg,
                    name=f"[AVG]{_name}",
                )
                _rec_layer.metadata["fsc"] = widget_utils.FscResult(
                    freq, fsc_mean, fsc_std, res0143, res0500
                )

        return _calculate_fsc_on_return

    @set_design(text="PCA/K-means classification", location=STAnalysis)
    @dask_worker.with_progress(descs=_pdesc.classify_pca_fmt)
    def classify_pca(
        self,
        layer: MoleculesLayerType,
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: Annotated[Optional[nm], {"text": "Use mask shape", "options": {"value": 12.0, "max": 100.0}, "label": "size (nm)"}] = None,
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
        {layer}{mask_params}{size}{cutoff}{interpolation}{bin_size}
        n_components : int, default is 2
            The number of PCA dimensions.
        n_clusters : int, default is 2
            The number of clusters.
        seed : int, default is 0
            Random seed.
        """
        from cylindra.widgets.subwidgets import PcaViewer

        t0 = timer("classify_pca")
        layer = assert_layer(layer, self.parent_viewer)
        parent = self._get_main()

        loader = self._get_loader(
            binsize=bin_size, molecules=layer.molecules, order=interpolation
        )
        _, mask = loader.normalize_input(
            template=self.params._get_template(allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        out, pca = loader.reshape(
            mask=mask,
            shape=None
            if size is None
            else (parent.tomogram.nm2pixel(size, binsize=bin_size),) * 3,
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
        layer.molecules = out.molecules  # update features
        t0.toc()

        @thread_worker.callback
        def _on_return():
            pca_viewer = PcaViewer(pca)
            pca_viewer.native.setParent(self.native, pca_viewer.native.windowFlags())
            pca_viewer.show()
            self._show_rec(avgs, name=f"[PCA]{layer.name}", store=False)
            parent._active_widgets.add(pca_viewer)

        return _on_return

    @set_design(text=capitalize, location=STAnalysis.SeamSearch)
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Seam search of {!r}"))
    def seam_search(
        self,
        layer: MoleculesLayerType,
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        anti_template_path: Annotated[Optional[Path.Read[FileFilter.IMAGE]], {"text": "Do not use anti-template"}] = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        npf: Annotated[Optional[int], {"text": "Use global properties"}] = None,
        show_average: Annotated[str, {"label": "Show averages as", "choices": [None, "Raw", "Filtered"]}] = "Filtered",
        cutoff: _CutoffFreq = 0.25,
    ):  # fmt: skip
        """
        Search for the best seam position.

        Try all patterns of seam positions and compare cross correlation values. If molecule
        assembly has 13 protofilaments, this method will try 26 patterns.

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
        show_average : bool, default is True
            If true, all the subtomogram averages will be shown.
        {cutoff}
        """
        t0 = timer("seam_search")
        layer = assert_layer(layer, self.parent_viewer)
        loader, npf = self._seam_search_input(layer, npf, interpolation)
        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
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

        new_feat = result.as_series(loader.molecules.count())
        layer.features = layer.molecules.features.with_columns(new_feat)
        layer.metadata[SEAM_SEARCH_RESULT] = result

        t0.toc()

        @thread_worker.callback
        def _seam_search_on_return():
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
        feat = layer.features
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

        Parameters
        ----------
        {layer}
        location : int
            Seam location.
        """
        layer = assert_layer(layer, self.parent_viewer)
        feat = layer.features
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

    @thread_worker.callback
    def _align_all_on_return(
        self, molecules: list[Molecules], old_layers: list[MoleculesLayer]
    ):
        """The return callback function for alignment methods."""
        parent = self._get_main()
        new_layers = []
        for mole, layer in zip(molecules, old_layers, strict=True):
            points = parent.add_molecules(
                mole,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
                source=layer.source_component,
            )
            new_layers.append(points)
            layer.visible = False
            _Logger.print_html(f"{layer.name!r} &#8594; {points.name!r}")

        @undo_callback
        def out():
            parent._try_removing_layers(new_layers)
            for layer in old_layers:
                layer.visible = True

        @out.with_redo
        def out():
            for points in new_layers:
                parent.parent_viewer.add_layer(points)

        return out

    def _get_simple_annealing_model(self, layer: MoleculesLayer):
        # TODO: This method should finally be moved to some utils module since
        # this analysis is independent of annealing. Currently annealing and
        # graph construction cannot be separated.
        parent = self._get_main()
        scale = parent.tomogram.scale
        return _annealing.get_annealing_model(layer, (0, 0, 0), scale)


def _coerce_aligned_name(name: str, viewer: "napari.Viewer"):
    num = 1
    if re.match(rf".*-{ALN_SUFFIX}(\d)+", name):
        try:
            *pre, suf = name.split(f"-{ALN_SUFFIX}")
            num = int(suf) + 1
            name = "".join(pre)
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
        return func(ip.asarray(img * mask, axes="zyx"), temp * mask)

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


def _check_viterbi_shift(shift: "NDArray[np.int32]", offset: "NDArray[np.int32]", i):
    invalid = shift[:, 0] < 0
    if invalid.any():
        invalid_indices = np.where(invalid)[0]
        _Logger.print(
            f"Viterbi alignment could not determine the optimal positions for PF={i!r}"
        )
        for idx in invalid_indices:
            shift[idx, :] = offset

    return shift


impl_preview(SubtomogramAveraging.align_all_annealing, text="Preview molecule network")(
    _annealing.preview_single
)

impl_preview(
    SubtomogramAveraging.align_all_annealing_multi_template,
    text="Preview molecule network",
)(_annealing.preview_multiple)
