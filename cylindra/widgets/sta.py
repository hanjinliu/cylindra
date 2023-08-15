from typing import (
    Any,
    Iterable,
    Literal,
    Union,
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
    nogui,
    vfield,
    MagicTemplate,
    set_design,
    impl_preview,
    abstractapi,
)
from magicclass.widgets import HistoryFileEdit, Separator
from magicclass.types import Optional, Path, ExprStr, Bound
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.undo import undo_callback
from magicclass.ext.dask import dask_thread_worker as dask_worker

from acryo import Molecules, SubtomogramLoader, alignment, pipe
from acryo.tilt import single_axis

import numpy as np
import impy as ip
import polars as pl
import napari

from cylindra import utils, _config, cylstructure
from cylindra.types import MoleculesLayer, get_monomer_layers
from cylindra.const import (
    ALN_SUFFIX,
    MoleculesHeader as Mole,
    nm,
    PropertyNames as H,
    Ori,
)
from cylindra.widgets._widget_ext import (
    CheckBoxes,
    RotationEdit,
    RandomSeedEdit,
    MultiFileEdit,
)
from cylindra.components.landscape import Landscape, ViterbiResult
from cylindra.components.seam_search import (
    CorrelationSeamSearcher,
    FiducialSeamSearcher,
    BooleanSeamSearcher,
)

from .widget_utils import FileFilter, timer, POLARS_NAMESPACE
from . import widget_utils, _shared_doc, _progress_desc as _pdesc, _annealing

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from dask import array as da
    from napari.layers import Image
    from cylindra.components import CylSpline
    from cylindra.widgets.main import CylindraMainWidget

# annotated types
_MoleculeLayers = Annotated[
    list[MoleculesLayer],
    {"choices": get_monomer_layers, "widget_type": CheckBoxes, "value": ()},
]
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_Rotations = Annotated[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    {"widget_type": RotationEdit},
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
_FSCFreq = Annotated[
    Optional[float],
    {
        "label": "Frequency precision",
        "text": "Choose proper value",
        "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02},
    },
]

# choices
INTERPOLATION_CHOICES = (("nearest", 0), ("linear", 1), ("cubic", 3))
METHOD_CHOICES = (
    ("Phase Cross Correlation", "pcc"),
    ("Zero-mean Normalized Cross Correlation", "zncc"),
)
AVG_CHOICES = [None, "Raw", "Filtered"]
_Logger = getLogger("cylindra")


def _get_alignment(method: str):
    if method == "zncc":
        return alignment.ZNCCAlignment
    elif method == "pcc":
        return alignment.PCCAlignment
    else:
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

    dilate_radius = vfield(0.3, record=False).with_options(max=20, step=0.1)
    sigma = vfield(0.8, record=False).with_options(max=20, step=0.1)


@magicclass(layout="horizontal", widget_type="frame", visible=False, record=False)
class mask_path(MagicTemplate):
    """Path to the mask image."""

    mask_path = vfield(Path.Read[FileFilter.IMAGE])


@magicmenu
class SubtomogramAnalysis(MagicTemplate):
    """Analysis of subtomograms."""

    average_all = abstractapi()
    average_subset = abstractapi()
    average_groups = abstractapi()
    split_and_average = abstractapi()
    sep0 = field(Separator)
    calculate_fsc = abstractapi()
    classify_pca = abstractapi()
    sep1 = field(Separator)
    seam_search = abstractapi()
    seam_search_by_fiducials = abstractapi()
    seam_search_by_feature = abstractapi()
    sep2 = field(Separator)
    save_last_average = abstractapi()


@magicmenu
class Refinement(MagicTemplate):
    """Refinement and alignment of subtomograms."""

    align_averaged = abstractapi()
    align_all = abstractapi()
    align_all_template_free = abstractapi()
    align_all_multi_template = abstractapi()
    align_all_viterbi = abstractapi()
    align_all_viterbi_multi_template = abstractapi()
    align_all_annealing = abstractapi()
    align_all_annealing_multi_template = abstractapi()


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

    _last_average: ip.ImgArray = None  # the global average result

    def __post_init__(self):
        self._template: ip.ImgArray = None
        self._viewer: "napari.Viewer | None" = None
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

    def _get_template(self, path: Union[Path, None] = None, allow_none: bool = False):
        if path is None:
            path = self.template_path.value
        self._save_history()

        if path is None:
            if self._last_average is None:
                if allow_none:
                    return None
                raise ValueError(
                    "No average image found. You can uncheck 'Use last averaged image' and select "
                    "a template image from a file."
                )
            provider = pipe.from_array(self._last_average, self._last_average.scale.x)
        else:
            path = Path(path)
            if path.is_dir():
                if allow_none:
                    return None
                raise TypeError(f"Template image must be a file, got {path}.")
            provider = pipe.from_file(path)
        return provider

    def _get_mask_params(self, params=None) -> Union[str, tuple[nm, nm], None]:
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

        if self._viewer is not None:
            try:
                # This line will raise RuntimeError if viewer window had been closed by user.
                self._viewer.window.activate()
            except RuntimeError:
                self._viewer = None
        if self._viewer is None:
            from cylindra.widgets.subwidgets import Volume

            self._viewer = napari.Viewer(
                title=name, axis_labels=("z", "y", "x"), ndisplay=3
            )
            volume_menu = Volume()
            self._viewer.window.add_dock_widget(volume_menu)
            self._viewer.window.resize(10, 10)
            self._viewer.window.activate()
        self._viewer.scale_bar.visible = True
        self._viewer.scale_bar.unit = "nm"
        if store:
            self._last_average = image
        thr = threshold_yen(image.value)
        layer = self._viewer.add_image(
            image,
            scale=image.scale,
            name=name,
            rendering="iso",
            iso_threshold=thr,
            blending="opaque",
        )
        return layer


@magicclass
@_shared_doc.update_cls
class SubtomogramAveraging(MagicTemplate):
    """Widget for subtomogram averaging."""

    Subtomogram_analysis = field(SubtomogramAnalysis)
    Refinement = field(Refinement)
    params = StaParameters

    @property
    def sub_viewer(self):
        """The napari viewer for subtomogram averaging."""
        return self.params._viewer

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class Buttons(MagicTemplate):
        show_template = abstractapi()
        show_mask = abstractapi()

    @Buttons.wraps
    @set_design(text="Show template")
    @do_not_record
    def show_template(self):
        """Load and show template image in the scale of the tomogram."""
        self._show_rec(self.template, name="Template image", store=False)

    @Buttons.wraps
    @set_design(text="Show mask")
    @do_not_record
    def show_mask(self):
        """Load and show mask image in the scale of the tomogram."""
        self._show_rec(self.mask, name="Mask image", store=False)

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
        return self.params._last_average

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
        return self._get_parent().tomogram.get_subtomogram_loader(
            molecules,
            binsize=binsize,
            order=order,
            output_shape=shape,
        )

    def _get_parent(self) -> "CylindraMainWidget":
        from .main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)

    def _get_available_binsize(self, _=None) -> list[int]:
        parent = self._get_parent()
        out = [x[0] for x in parent.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return out

    @Subtomogram_analysis.wraps
    @set_design(text="Average all molecules")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Subtomogram averaging of {!r}"))
    def average_all(
        self,
        layer: MoleculesLayer,
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Subtomogram averaging using all the molecules in the selected layer.

        Parameters
        ----------
        {layer}{size}{interpolation}{bin_size}
        """
        t0 = timer("average_all")
        layer = _assert_layer(layer, self.parent_viewer)
        parent = self._get_parent()
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            layer.molecules, shape, binsize=bin_size, order=interpolation
        )
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{layer.name}")

    @Subtomogram_analysis.wraps
    @set_design(text="Average subset of molecules")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Subtomogram averaging (subset) of {!r}"))  # fmt: skip
    def average_subset(
        self,
        layer: MoleculesLayer,
        size: _SubVolumeSize = None,
        method: Literal["steps", "first", "last", "random"] = "steps",
        number: int = 64,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Subtomogram averaging using a subset of subvolumes.

        Parameters
        ----------
        {layer}{size}
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
        layer = _assert_layer(layer, self.parent_viewer)
        parent = self._get_parent()
        molecules = layer.molecules
        nmole = len(molecules)
        shape = self._get_shape_in_nm(size)
        sl = _get_slice_for_average_subset(method, nmole, number)
        mole = molecules.subset(sl)
        loader = parent.tomogram.get_subtomogram_loader(
            mole, shape, binsize=bin_size, order=1
        )
        img = ip.asarray(loader.average(), axes="zyx").set_scale(zyx=loader.scale)
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG(n={number})]{layer.name}")

    @Subtomogram_analysis.wraps
    @set_design(text="Average group-wise")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Grouped subtomogram averaging of {!r}"))  # fmt: skip
    def average_groups(
        self,
        layer: MoleculesLayer,
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
        {layer}{size}
        by : str or polars expression
            Expression to group molecules.
        {interpolation}{bin_size}
        """
        t0 = timer("average_groups")
        layer = _assert_layer(layer, self.parent_viewer)
        if isinstance(by, pl.Expr):
            expr = by
        else:
            expr = ExprStr(by, POLARS_NAMESPACE).eval()
        parent = self._get_parent()
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            layer.molecules, shape, binsize=bin_size, order=interpolation
        )
        avgs = np.stack(list(loader.groupby(expr).average().values()), axis=0)
        img = ip.asarray(avgs, axes="pzyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{layer.name}", store=False)

    @Subtomogram_analysis.wraps
    @set_design(text="Split molecules and average")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Split-and-averaging of {!r}"))  # fmt: skip
    def split_and_average(
        self,
        layer: MoleculesLayer,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: Annotated[int, {"choices": _get_available_binsize}] = 1,
    ):
        """
        Split molecules into two groups and average separately.

        Parameters
        ----------
        {layer}
        n_set : int, default is 1
            How many pairs of average will be calculated.
        {size}{interpolation}{bin_size}
        """
        t0 = timer("split_and_average")
        layer = _assert_layer(layer, self.parent_viewer)
        parent = self._get_parent()
        molecules = layer.molecules
        shape = self._get_shape_in_nm(size)
        loader = parent.tomogram.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation
        )
        axes = "ipzyx" if n_set > 1 else "pzyx"
        img = ip.asarray(loader.average_split(n_set=n_set), axes=axes)
        img.set_scale(zyx=loader.scale)
        t0.toc()
        return self._show_rec.with_args(img, f"[Split]{layer.name}")

    @Refinement.wraps
    @set_design(text="Align average to template")
    @dask_worker.with_progress(descs=_pdesc.align_averaged_fmt)
    def align_averaged(
        self,
        layers: _MoleculeLayers,
        template_path: Annotated[Union[str, Path], {"bind": params.template_path}],
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
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
        layers = _assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_parent()

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
            _mole_trans = mole.linear_transform(
                result.shift * _scale, rotator
            ).with_features(pl.col(Mole.position) + svec[1])

            # create images for visualization in the logger
            img_norm = utils.normalize_image(_img_trans)
            merge = np.stack([img_norm, temp_norm, img_norm], axis=-1)

            # write offsets to spline globalprops if available
            # TODO: Undo cannot catch this change. Need to fix.
            if spl := layer.source_spline:
                if spl.radius is None:
                    _radius: nm = cylstructure.calc_radius(mole, spl).mean()
                else:
                    _radius = spl.radius
                _dz, _offset_y, _dx = rotator.apply(svec)
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
                spl.globalprops = spl.globalprops.with_columns(
                    pl.Series(H.offset_axial, [_offset_y], dtype=pl.Float32),
                    pl.Series(H.offset_angular, [_offset_a], dtype=pl.Float32),
                    pl.Series(H.offset_radial, [_offset_r], dtype=pl.Float32),
                )

            yield _on_yield.with_args(_mole_trans, layer)

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

    @Refinement.wraps
    @set_design(text="Align all molecules")
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Alignment of {}"))
    def align_all(
        self,
        layers: _MoleculeLayers,
        template_path: Annotated[Union[str, Path], {"bind": params.template_path}],
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
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
        layers = _assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_parent()

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
            tilt=single_axis(parent.tomogram.tilt_range, axis="y"),
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    @Refinement.wraps
    @set_design(text="Align all (template-free)")
    @dask_worker.with_progress(descs=_pdesc.align_template_free_fmt)
    def align_all_template_free(
        self,
        layers: _MoleculeLayers,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
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
        layers = _assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_parent()
        combiner = MoleculesCombiner()
        molecules = combiner.concat(layer.molecules for layer in layers)
        mask = self.params._get_mask(params=mask_params)
        if size is None:
            shape = None
            raise NotImplementedError("'size' must be given.")
        else:
            shape = tuple(parent.tomogram.nm2pixel(self._get_shape_in_nm(size)))

        aligned_loader = (
            self._get_loader(binsize=bin_size, molecules=molecules, order=interpolation)
            .reshape(shape=shape)
            .align_no_template(
                mask=mask,
                max_shifts=max_shifts,
                rotations=rotations,
                cutoff=cutoff,
                alignment_model=_get_alignment(method),
                tilt=single_axis(parent.tomogram.tilt_range, axis="y"),
            )
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    @Refinement.wraps
    @set_design(text="Align all (multi-template)")
    @dask_worker.with_progress(desc=_pdesc.fmt_layers("Multi-template alignment of {}"))  # fmt: skip
    def align_all_multi_template(
        self,
        layers: _MoleculeLayers,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
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
        layers = _assert_list_of_layers(layers, self.parent_viewer)
        parent = self._get_parent()
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
            tilt=single_axis(parent.tomogram.tilt_range, axis="y"),
        )
        molecules = combiner.split(aligned_loader.molecules, layers)
        t0.toc()
        return self._align_all_on_return.with_args(molecules, layers)

    @Refinement.wraps
    @set_design(text="Viterbi Alignment")
    @dask_worker.with_progress(descs=_pdesc.align_viterbi_fmt)
    def align_all_viterbi(
        self,
        layer: MoleculesLayer,
        template_path: Annotated[Union[str, Path], {"bind": params.template_path}],
        mask_params: Annotated[Any, {"bind": params._get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range: _DistRangeLon = (4.0, 4.28),
        angle_max: Optional[float] = 5.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
    ):
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

    @Refinement.wraps
    @set_design(text="Viterbi Alignment (multi-template)")
    @dask_worker.with_progress(descs=_pdesc.align_viterbi_fmt)
    def align_all_viterbi_multi_template(
        self,
        layer: MoleculesLayer,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range: _DistRangeLon = (4.0, 4.28),
        angle_max: Optional[float] = 5.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
    ):
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

    @Refinement.wraps
    @set_design(text="Simulated annealing")
    @dask_worker.with_progress(descs=_pdesc.align_annealing_fmt)
    def align_all_annealing(
        self,
        layer: MoleculesLayer,
        template_path: Annotated[Union[str, Path], {"bind": params.template_path}],
        mask_params: Annotated[Any, {"bind": params._get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range_long: _DistRangeLon = (4.0, 4.28),
        distance_range_lat: _DistRangeLat = (5.1, 5.3),
        angle_max: _AngleMaxLon = 5.0,
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
        {angle_max}{upsample_factor}{random_seeds}
        """
        kwargs = locals()
        kwargs.pop("self")
        t0 = timer("align_all_annealing")
        out = yield from self._align_all_annealing(**kwargs)
        t0.toc()
        return out

    @Refinement.wraps
    @set_design(text="Simulated annealing (multi-template)")
    @dask_worker.with_progress(descs=_pdesc.align_annealing_fmt)
    def align_all_annealing_multi_template(
        self,
        layer: MoleculesLayer,
        template_paths: _ImagePaths,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}] = None,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        distance_range_long: _DistRangeLon = (4.0, 4.28),
        distance_range_lat: _DistRangeLat = (5.1, 5.3),
        angle_max: _AngleMaxLon = 5.0,
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
        {angle_max}{upsample_factor}{random_seeds}
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
        mask_params: Annotated[Any, {"bind": params._get_mask_params}] = None,
        max_shifts: tuple[nm, nm, nm] = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: float = 0.5,
        interpolation: int = 3,
        upsample_factor: int = 5,
    ):
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
        parent = self._get_parent()
        layer = _assert_layer(layer, self.parent_viewer)
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
                tilt=single_axis(parent.tomogram.tilt_range, axis="y"),
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
        from dask import delayed, compute

        parent = self._get_parent()
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
        npfs = mole.features[Mole.pf].unique(maintain_order=True)

        slices = [(mole.features[Mole.pf] == i).to_numpy() for i in npfs]
        viterbi_tasks = [
            delayed(landscape[sl].run_viterbi)(distance_range, angle_max)
            for sl in slices
        ]
        vit_out: list[ViterbiResult] = compute(viterbi_tasks)[0]

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
        angle_max: float = 10.0,
        upsample_factor: int = 5,
        random_seeds: Iterable[int] = range(5),
        return_all: bool = False,
    ):
        parent = self._get_parent()
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
                        metadata={"annealing-result": result},
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
                    metadata={"annealing-result": results[0]},
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

    @Subtomogram_analysis.wraps
    @set_design(text="Calculate FSC")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Calculating FSC of {!r}"))
    def calculate_fsc(
        self,
        layer: MoleculesLayer,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: _FSCFreq = None,
    ):
        """
        Calculate Fourier Shell Correlation using the selected monomer layer.

        Parameters
        ----------
        {layer}{mask_params}{size}
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
        layer = _assert_layer(layer, self.parent_viewer)
        parent = self._get_parent()
        mole = layer.molecules

        loader = parent.tomogram.get_subtomogram_loader(mole, order=interpolation)
        template, mask = loader.normalize_input(
            template=self.params._get_template(allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        fsc, avg = loader.reshape(
            template=template if size is None else None,
            mask=mask,
            shape=None if size is None else (parent.tomogram.nm2pixel(size),) * 3,
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
            _Logger.print_html(f"<b>Fourier Shell Correlation of {layer.name!r}</b>")
            with _Logger.set_plt():
                widget_utils.plot_fsc(
                    freq,
                    fsc_mean,
                    fsc_std,
                    [crit_0143, crit_0500],
                    parent.tomogram.scale,
                )

            _Logger.print_html(f"Resolution at FSC=0.5 ... <b>{res0500:.3f} nm</b>")
            _Logger.print_html(f"Resolution at FSC=0.143 ... <b>{res0143:.3f} nm</b>")

            if img_avg is not None:
                _rec_layer: "Image" = self._show_rec(
                    img_avg,
                    name=f"[AVG]{layer.name}",
                )
                _rec_layer.metadata["fsc"] = widget_utils.FscResult(
                    freq, fsc_mean, fsc_std, res0143, res0500
                )

        return _calculate_fsc_on_return

    @Subtomogram_analysis.wraps
    @set_design(text="PCA/K-means classification")
    @dask_worker.with_progress(descs=_pdesc.classify_pca_fmt)
    def classify_pca(
        self,
        layer: MoleculesLayer,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
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
        layer = _assert_layer(layer, self.parent_viewer)
        parent = self._get_parent()

        loader = self._get_loader(
            binsize=bin_size, molecules=layer.molecules, order=interpolation
        )
        _, mask = loader.normalize_input(
            template=self.params._get_template(allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        out, pca = loader.reshape(
            mask=mask,
            shape=None if size is None else (parent.tomogram.nm2pixel(size),) * 3,
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

    @Subtomogram_analysis.wraps
    @set_design(text="Seam search")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Seam search of {!r}"))
    def seam_search(
        self,
        layer: MoleculesLayer,
        template_path: Annotated[Union[str, Path], {"bind": params.template_path}],
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        npf: Annotated[Optional[int], {"text": "Use global properties"}] = None,
        show_average: Annotated[str, {"label": "Show averages as", "choices": AVG_CHOICES}] = AVG_CHOICES[2],
        cutoff: _CutoffFreq = 0.25,
    ):  # fmt: skip
        """
        Search for the best seam position.

        Try all patterns of seam positions and compare cross correlation values. If molecule
        assembly has 13 protofilaments, this method will try 26 patterns.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{interpolation}
        npf : int, optional
            Number of protofilaments. By default the global properties stored in the
            corresponding spline will be used.
        show_average : bool, default is True
            If true, all the subtomogram averages will be shown.
        {cutoff}
        """
        t0 = timer("seam_search")
        layer = _assert_layer(layer, self.parent_viewer)
        loader, npf = self._seam_search_input(layer, npf, interpolation)
        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
        )

        seam_searcher = CorrelationSeamSearcher(npf)
        result = seam_searcher.search(
            loader=loader,
            template=ip.asarray(template, axes="zyx"),
            mask=mask,
            cutoff=cutoff,
        )

        layer.features = layer.molecules.features.with_columns(
            pl.Series(Mole.isotype, result.get_label(loader.molecules.count()))
        )
        layer.metadata["seam-search-score"] = result.scores

        t0.toc()

        @thread_worker.callback
        def _seam_search_on_return():
            if show_average is not None:
                if show_average == AVG_CHOICES[2]:
                    sigma = 0.25 / loader.scale
                    result.averages.gaussian_filter(sigma=sigma, update=True)
                self._show_rec(result.averages, layer.name, store=False)
                self.sub_viewer.layers[-1].metadata["Score"] = result.scores

            # plot all the correlation
            _Logger.print_html("<code>seam_search</code>")
            with _Logger.set_plt():
                _Logger.print(f"layer = {layer.name!r}")
                _Logger.print(f"template = {str(template_path)!r}")
                widget_utils.plot_seam_search_result(result.scores, npf)

        return _seam_search_on_return

    @Subtomogram_analysis.wraps
    @set_design(text="Seam search (by fiducials)")
    @dask_worker.with_progress(desc=_pdesc.fmt_layer("Seam search (by fiducials) of {!r}"))  # fmt: skip
    def seam_search_by_fiducials(
        self,
        layer: MoleculesLayer,
        mask_params: Annotated[Any, {"bind": params._get_mask_params}],
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        npf: Annotated[Optional[int], {"text": "Use global properties"}] = None,
        show_average: Annotated[str, {"label": "Show averages as", "choices": AVG_CHOICES}] = AVG_CHOICES[2],
    ):  # fmt: skip
        t0 = timer("seam_search_by_fiducials")
        layer = _assert_layer(layer, self.parent_viewer)
        loader, npf = self._seam_search_input(layer, npf, interpolation)
        seam_searcher = FiducialSeamSearcher(npf)
        _, weight = loader.normalize_input(
            self.params._get_template(allow_none=True),
            self.params._get_mask(mask_params),
        )
        if weight is None:
            raise ValueError("Mask is required for seam search by fiducials.")
        weight = ip.asarray(weight, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        result = seam_searcher.search(loader=loader, weight=weight)
        layer.features = layer.molecules.features.with_columns(
            pl.Series(Mole.isotype, result.get_label(loader.molecules.count()))
        )
        layer.metadata["seam-search-score"] = result.scores
        t0.toc()

        @thread_worker.callback
        def _seam_search_on_return():
            if show_average is not None:
                if show_average == AVG_CHOICES[2]:
                    sigma = 0.25 / loader.scale
                    result.averages.gaussian_filter(sigma=sigma, update=True)
                self._show_rec(result.averages, layer.name, store=False)
                self.sub_viewer.layers[-1].metadata["Score"] = result.scores

            # plot all the correlation
            _Logger.print_html("<code>seam_search_by_fiducials</code>")
            with _Logger.set_plt():
                _Logger.print(f"layer = {layer.name!r}")
                widget_utils.plot_seam_search_result(result.scores, npf)

        return _seam_search_on_return

    @Subtomogram_analysis.wraps
    @set_design(text="Seam search (by feature)")
    def seam_search_by_feature(
        self,
        layer: MoleculesLayer,
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
        layer = _assert_layer(layer, self.parent_viewer)
        feat = layer.features
        if by not in feat.columns:
            raise ValueError(f"Column {by} does not exist.")
        npf = utils.roundint(layer.molecules.features[Mole.pf].max() + 1)
        seam_searcher = BooleanSeamSearcher(npf)
        result = seam_searcher.search(feat[by])
        new_feat = pl.Series(
            Mole.isotype, result.get_label(feat.shape[0]), dtype=pl.Int8
        )
        layer.features = layer.molecules.features.with_columns(new_feat)
        return undo_callback(layer.feature_setter(feat, layer.colormap_info))

    def _seam_search_input(
        self, layer: MoleculesLayer, npf: int, order: int
    ) -> tuple[SubtomogramLoader, int]:
        parent = self._get_parent()
        mole = layer.molecules
        loader = parent.tomogram.get_subtomogram_loader(mole, order=order)
        if npf is None:
            npf = mole.features[Mole.pf].unique().len()
        return loader, npf

    @Subtomogram_analysis.wraps
    @set_design(text="Save last average")
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
        parent = self._get_parent()
        new_layers = []
        for mole, layer in zip(molecules, old_layers):
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
        parent = self._get_parent()
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


def _assert_list_of_layers(
    layers: Any, viewer: "napari.Viewer"
) -> list[MoleculesLayer]:
    if len(layers) == 0:
        raise ValueError("No layer selected.")
    if isinstance(layers, (MoleculesLayer, str)):
        layers = [layers]
    layer_normed: list[MoleculesLayer] = []
    for layer in layers:
        if isinstance(layer, str):
            layer_normed.append(viewer.layers[layer])
        elif isinstance(layer, MoleculesLayer):
            layer_normed.append(layer)
        else:
            raise TypeError(f"Layer {layer!r} is not a MoleculesLayer.")
    return layer_normed


def _assert_layer(layer: Any, viewer: "napari.Viewer") -> MoleculesLayer:
    if isinstance(layer, str):
        return viewer.layers[layer]
    elif isinstance(layer, MoleculesLayer):
        return layer
    else:
        raise TypeError(f"Layer {layer!r} is not a MoleculesLayer.")


def _get_slice_for_average_subset(method: str, nmole: int, number: int):
    if nmole < number:
        raise ValueError(f"There are only {nmole} subtomograms.")
    if method == "steps":
        step = nmole // number
        sl = slice(0, step * number, step)
    elif method == "first":
        sl = slice(0, number)
    elif method == "last":
        sl = slice(-number, -1)
    elif method == "random":
        sl_all = np.arange(nmole, dtype=np.uint32)
        np.random.shuffle(sl_all)
        sl = sl_all[:number]
    else:
        raise ValueError(f"method {method!r} not supported.")
    return sl


def _default_align_averaged_shifts(mole: Molecules) -> np.ndarray:
    npf = mole.features[Mole.pf].max() + 1
    dy = np.sqrt(np.sum((mole.pos[0] - mole.pos[1]) ** 2))  # axial shift
    dx = np.sqrt(np.sum((mole.pos[0] - mole.pos[npf]) ** 2))  # lateral shift
    return (dy * 0.6, dy * 0.6, dx * 0.6)


def _update_mole_pos(new: Molecules, old: Molecules, spl: "CylSpline") -> Molecules:
    """
    Update the "position-nm" feature of molecules.

    Feature "position-nm" is the coordinate of molecules along the source spline.
    After alignment, this feature should be updated accordingly. This fucntion
    will do this.
    """
    new_pos = new.pos
    old_pos = old.pos
    _u = old.features[Mole.position] / spl.length()
    vec = spl.map(_u, der=1)
    vec_norm = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    dy = np.sum((new_pos - old_pos) * vec_norm, axis=1)
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
