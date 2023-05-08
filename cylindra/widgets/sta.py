from typing import Iterable, Union, TYPE_CHECKING, Annotated, NamedTuple
import re
from scipy.spatial.transform import Rotation
from magicclass import (
    do_not_record,
    magicclass,
    magicmenu,
    field,
    vfield,
    MagicTemplate,
    set_design,
    abstractapi,
)
from magicclass.widgets import HistoryFileEdit, Separator
from magicclass.types import OneOf, Optional, Path, Bound
from magicclass.utils import thread_worker
from magicclass.ext.dask import dask_thread_worker
from magicclass.logging import getLogger
from magicclass.undo import undo_callback

from acryo import Molecules, SubtomogramLoader, alignment, pipe

import numpy as np
import impy as ip
import polars as pl
import napari

from cylindra import utils
from cylindra.types import MoleculesLayer
from cylindra.const import ALN_SUFFIX, MoleculesHeader as Mole, nm
from cylindra.components import CylSpline

from .widget_utils import FileFilter, timer
from . import widget_utils, _shared_doc, _progress_desc as _pdesc

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from napari.layers import Image
    from cylindra._cylindra_ext import CylindricAnnealingModel

# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_ZRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_YRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_XRotation = Annotated[tuple[float, float], {"options": {"max": 90.0, "step": 0.1}}]
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
_Logger = getLogger("cylindra")


def _get_alignment(method: str):
    if method == "zncc":
        return alignment.ZNCCAlignment
    elif method == "pcc":
        return alignment.PCCAlignment
    else:
        raise ValueError(f"Method {method!r} is unknown.")


MASK_CHOICES = ("No mask", "Blur template", "From file")


@magicclass(
    layout="horizontal",
    widget_type="groupbox",
    name="Parameters",
    visible=False,
    record=False,
)
@_shared_doc.update_cls
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
    sigma = vfield(0.3, record=False).with_options(max=20, step=0.1)


@magicclass(layout="horizontal", widget_type="frame", visible=False, record=False)
class mask_path(MagicTemplate):
    """Path to the mask image."""

    mask_path = vfield(Path.Read[FileFilter.IMAGE])


@magicmenu
class SubtomogramAnalysis(MagicTemplate):
    """Analysis of subtomograms."""

    average_all = abstractapi()
    average_subset = abstractapi()
    split_and_average = abstractapi()
    sep0 = field(Separator)
    calculate_fsc = abstractapi()
    classify_pca = abstractapi()
    sep1 = field(Separator)
    seam_search = abstractapi()
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
    align_all_boltzmann = abstractapi()


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
    tilt_range : tuple of float, options
        Tilt range (degree) of the tomogram.
    """

    template_path = vfield(
        Optional[
            Annotated[Path.Read[FileFilter.IMAGE], {"widget_type": HistoryFileEdit}]
        ],
        label="Template",
    ).with_options(text="Use last averaged image", value=Path(""))
    mask_choice = vfield(OneOf[MASK_CHOICES], label="Mask", record=False)
    params = field(MaskParameters, name="Mask parameters")
    mask_path = field(mask_path)
    tilt_range = vfield(
        Optional[tuple[nm, nm]], label="Tilt range (deg)", record=False
    ).with_options(
        value=(-60.0, 60.0),
        text="No missing-wedge",
        options={"options": {"min": -90.0, "max": 90.0, "step": 1.0}},
    )

    _last_average: ip.ImgArray = None  # the global average result

    def __post_init__(self):
        self._template: ip.ImgArray = None
        self._viewer: Union[napari.Viewer, None] = None
        self.mask_choice = MASK_CHOICES[0]

    @mask_choice.connect
    def _on_mask_switch(self):
        v = self.mask_choice
        self.params.visible = v == MASK_CHOICES[1]
        self.mask_path.visible = v == MASK_CHOICES[2]

    def _get_template(self, path: Union[Path, None] = None, allow_none: bool = False):
        if path is None:
            path = self.template_path
        else:
            self.template_path = path

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
        else:
            if params is None:
                self.mask_choice = MASK_CHOICES[0]
            elif isinstance(params, tuple):
                self.mask_choice = MASK_CHOICES[1]
            else:
                self.mask_path.mask_path = params

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
            from .function_menu import Volume

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
        input_image = utils.normalize_image(image)
        thr = threshold_yen(input_image.value)
        layer = self._viewer.add_image(
            input_image,
            scale=image.scale,
            name=name,
            rendering="iso",
            iso_threshold=thr,
            blending="opaque",
        )
        return layer


@magicclass(widget_type="scrollable")
@_shared_doc.update_cls
class SubtomogramAveraging(MagicTemplate):
    """
    Widget for subtomogram averaging.

    Attributes
    ----------
    template_path : Path
        Path to the template (reference) image file, or layer name of reconstruction.
    mask_path : str
        Select how to create a mask.
    tilt_range : tuple of float, options
        Tilt range (degree) of the tomogram.
    """

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
        self._show_reconstruction(self.template, name="Template image", store=False)

    @Buttons.wraps
    @set_design(text="Show mask")
    @do_not_record
    def show_mask(self):
        """Load and show mask image in the scale of the tomogram."""
        self._show_reconstruction(self.mask, name="Mask image", store=False)

    @property
    def template(self) -> "ip.ImgArray | None":
        """Template image."""
        loader = self._get_loader(binsize=1, molecules=Molecules.empty())
        template, _ = loader.normalize_input(self.params._get_template())
        return ip.asarray(template, axes="zyx").set_scale(zyx=loader.scale, unit="nm")

    @property
    def mask(self) -> "ip.ImgArray | None":
        """Mask image."""
        loader = self._get_loader(binsize=1, molecules=Molecules.empty())
        _, mask = loader.normalize_input(
            self.params._get_template(allow_none=True), self.params._get_mask()
        )
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

    def _show_reconstruction(
        self, image: ip.ImgArray, name: str, store: bool = True
    ) -> "Image":
        return self.params._show_reconstruction(image, name, store)

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
        from .main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget).tomogram.get_subtomogram_loader(
            molecules,
            binsize=binsize,
            order=order,
            shape=shape,
        )

    def _get_parent(self):
        from .main import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)

    def _get_available_binsize(self, _=None) -> list[int]:
        parent = self._get_parent()
        if parent.tomogram is None:
            return [1]
        out = [x[0] for x in parent.tomogram.multiscaled]
        if 1 not in out:
            out = [1] + out
        return out

    @Subtomogram_analysis.wraps
    @set_design(text="Average all molecules")
    @dask_thread_worker.with_progress(
        desc=_pdesc.fmt_layer_name("Subtomogram averaging of {!r}")
    )
    def average_all(
        self,
        layer: MoleculesLayer,
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Subtomogram averaging using all the molecules in the selected layer.

        Parameters
        ----------
        {layer}{size}{interpolation}{bin_size}
        """
        t0 = timer("average_all")
        parent = self._get_parent()
        molecules = layer.molecules
        tomo = parent.tomogram
        shape = self._get_shape_in_nm(size)
        loader = tomo.get_subtomogram_loader(
            molecules, shape, binsize=bin_size, order=interpolation
        )
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale, unit="nm")
        t0.toc()
        return thread_worker.to_callback(
            self._show_reconstruction, img, f"[AVG]{layer.name}"
        )

    @Subtomogram_analysis.wraps
    @set_design(text="Average subset of molecules")
    @dask_thread_worker.with_progress(
        desc=_pdesc.fmt_layer_name("Subtomogram averaging (subset) of {!r}")
    )
    def average_subset(
        self,
        layer: MoleculesLayer,
        size: _SubVolumeSize = None,
        method: OneOf["steps", "first", "last", "random"] = "steps",
        number: int = 64,
        bin_size: OneOf[_get_available_binsize] = 1,
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
        return thread_worker.to_callback(
            self._show_reconstruction, img, f"[AVG(n={number})]{layer.name}"
        )

    @Subtomogram_analysis.wraps
    @set_design(text="Split molecules and average")
    @dask_thread_worker.with_progress(
        desc=_pdesc.fmt_layer_name("Split-and-averaging of {!r}")
    )
    def split_and_average(
        self,
        layer: MoleculesLayer,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        bin_size: OneOf[_get_available_binsize] = 1,
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
        return thread_worker.to_callback(
            self._show_reconstruction, img, f"[Split]{layer.name}"
        )

    @Refinement.wraps
    @set_design(text="Align average to template")
    @dask_thread_worker.with_progress(descs=_pdesc.align_averaged_fmt)
    def align_averaged(
        self,
        layer: MoleculesLayer,
        template_path: Bound[params.template_path],
        mask_params: Bound[params._get_mask_params],
        z_rotation: _ZRotation = (0.0, 0.0),
        y_rotation: _YRotation = (15.0, 1.0),
        x_rotation: _XRotation = (3.0, 1.0),
        bin_size: OneOf[_get_available_binsize] = 1,
        method: OneOf[METHOD_CHOICES] = "zncc",
    ):
        """
        Align the averaged image at current monomers to the template image.

        This function creates a new layer with transformed monomers, which should
        align well with template image.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{z_rotation}{y_rotation}{x_rotation}{bin_size}{method}
        """
        t0 = timer("align_averaged")
        parent = self._get_parent()
        mole = layer.molecules
        loader = self._get_loader(bin_size, mole, order=1)
        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
        )

        _scale = parent.tomogram.scale * bin_size

        npf = mole.features[Mole.pf].max() + 1
        dy = np.sqrt(np.sum((mole.pos[0] - mole.pos[1]) ** 2))  # axial shift
        dx = np.sqrt(np.sum((mole.pos[0] - mole.pos[npf]) ** 2))  # lateral shift

        model = _get_alignment(method)(
            template,
            mask,
            cutoff=1.0,
            rotations=(z_rotation, y_rotation, x_rotation),
            tilt_range=None,  # NOTE: because input is an average
        )
        img_trans, result = model.fit(
            loader.average(template.shape),
            max_shifts=tuple(np.array([dy, dy, dx]) / _scale * 0.6),
        )

        rotator = Rotation.from_quat(result.quat)
        shift_nm = result.shift * _scale
        mole_trans = mole.linear_transform(
            result.shift * _scale, rotator
        ).with_features([pl.col(Mole.position) + shift_nm[1]])

        # logging
        t0.toc()
        _Logger.print_html(
            "{rotvec_str} = {rot_str}, {vec_str} = {shift_nm_str}".format(
                vec_str=", ".join(f"{x}<sub>shift</sub>" for x in "XYZ"),
                rotvec_str=", ".join(f"{x}<sub>rot</sub>" for x in "XYZ"),
                shift_nm_str=", ".join(f"{s:.2f} nm" for s in shift_nm[::-1]),
                rot_str=", ".join(f"{s:.2f}" for s in rotator.as_rotvec()[::-1]),
            )
        )

        parent._need_save = True

        @thread_worker.to_callback
        def _align_averaged_on_return():
            points = parent.add_molecules(
                mole_trans,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
                source=layer.source_component,
            )
            img_norm = utils.normalize_image(img_trans)
            temp_norm = utils.normalize_image(template)
            merge = np.stack([img_norm, temp_norm, img_norm], axis=-1)
            layer.visible = False
            _Logger.print_html(f"{layer.name!r} &#8594; {points.name!r}")
            with _Logger.set_plt():
                widget_utils.plot_projections(merge)
            return (
                undo_callback(parent._try_removing_layer)
                .with_args(points)
                .with_redo(lambda: parent.parent_viewer.add_layer(points))
            )

        return _align_averaged_on_return

    @Refinement.wraps
    @set_design(text="Align all molecules")
    @dask_thread_worker.with_progress(desc=_pdesc.fmt_layer_name("Alignment of {!r}"))
    def align_all(
        self,
        layer: MoleculesLayer,
        template_path: Bound[params.template_path],
        mask_params: Bound[params._get_mask_params],
        tilt_range: Bound[params.tilt_range] = None,
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        z_rotation: _ZRotation = (0.0, 0.0),
        y_rotation: _YRotation = (0.0, 0.0),
        x_rotation: _XRotation = (0.0, 0.0),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Align all the molecules for subtomogram averaging.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{tilt_range}{max_shifts}{z_rotation}{y_rotation}
        {x_rotation}{cutoff}{interpolation}{method}{bin_size}
        """
        t0 = timer("align_all")
        parent = self._get_parent()

        loader = self._get_loader(
            binsize=bin_size,
            molecules=layer.molecules,
            order=interpolation,
        )
        aligned_loader = loader.align(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=_get_alignment(method),
            tilt_range=tilt_range,
        )

        t0.toc()
        parent._need_save = True
        return self._align_all_on_return(aligned_loader, layer)

    @Refinement.wraps
    @set_design(text="Align all (template-free)")
    @dask_thread_worker.with_progress(descs=_pdesc.align_template_free_fmt)
    def align_all_template_free(
        self,
        layer: MoleculesLayer,
        mask_params: Bound[params._get_mask_params],
        tilt_range: Bound[params.tilt_range] = None,
        size: _SubVolumeSize = 12.0,
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        z_rotation: _ZRotation = (0.0, 0.0),
        y_rotation: _YRotation = (0.0, 0.0),
        x_rotation: _XRotation = (0.0, 0.0),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Align all the molecules for subtomogram averaging.

        Parameters
        ----------
        {layer}{mask_params}{tilt_range}{size}{max_shifts}{z_rotation}{y_rotation}{x_rotation}
        {cutoff}{interpolation}{method}{bin_size}
        """
        t0 = timer("align_all_template_free")
        parent = self._get_parent()
        molecules = layer.molecules
        mask = self.params._get_mask(params=mask_params)
        if size is None:
            shape = None
        else:
            shape = tuple(parent.tomogram.nm2pixel(self._get_shape_in_nm(size)))

        loader = self._get_loader(
            binsize=bin_size, molecules=molecules, order=interpolation
        ).reshape(mask, shape=shape)

        aligned_loader = loader.align_no_template(
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=_get_alignment(method),
            tilt_range=tilt_range,
        )

        t0.toc()
        parent._need_save = True
        return self._align_all_on_return(aligned_loader, layer)

    @Refinement.wraps
    @set_design(text="Align all (multi-template)")
    @dask_thread_worker.with_progress(
        desc=_pdesc.fmt_layer_name("Multi-template alignment of {!r}")
    )
    def align_all_multi_template(
        self,
        layer: MoleculesLayer,
        template_path: Bound[params.template_path],
        other_templates: Path.Multiple[FileFilter.IMAGE],
        mask_params: Bound[params._get_mask_params],
        tilt_range: Bound[params.tilt_range] = None,
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        z_rotation: _ZRotation = (0.0, 0.0),
        y_rotation: _YRotation = (0.0, 0.0),
        x_rotation: _XRotation = (0.0, 0.0),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
        bin_size: OneOf[_get_available_binsize] = 1,
    ):
        """
        Align all the molecules for subtomogram averaging.

        Parameters
        ----------
        {layer}{template_path}
        other_templates : list of Path or str
            Path to other template images.
        {mask_params}{tilt_range}{max_shifts}{z_rotation}{y_rotation}{x_rotation}{cutoff}
        {interpolation}{method}{bin_size}
        """
        t0 = timer("align_all_multi_template")
        parent = self._get_parent()
        molecules = layer.molecules
        templates = [self.params._get_template(path=template_path)]
        for path in other_templates:
            templates.append(pipe.from_file(path))

        loader = self._get_loader(
            binsize=bin_size,
            molecules=molecules,
            order=interpolation,
        )
        aligned_loader = loader.align_multi_templates(
            templates=templates,
            mask=self.params._get_mask(params=mask_params),
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model=_get_alignment(method),
            tilt_range=tilt_range,
        )
        t0.toc()
        parent._need_save = True
        return self._align_all_on_return(aligned_loader, layer)

    @Refinement.wraps
    @set_design(text="1D Viterbi Alignment")
    @dask_thread_worker.with_progress(descs=_pdesc.align_viterbi_fmt)
    def align_all_viterbi(
        self,
        layer: MoleculesLayer,
        template_path: Bound[params.template_path],
        mask_params: Bound[params._get_mask_params] = None,
        tilt_range: Bound[params.tilt_range] = None,
        max_shifts: _MaxShifts = (0.6, 0.6, 0.6),
        z_rotation: _ZRotation = (0.0, 0.0),
        y_rotation: _YRotation = (0.0, 0.0),
        x_rotation: _XRotation = (0.0, 0.0),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        distance_range: _DistRangeLon = (3.9, 4.4),
        max_angle: Optional[float] = 6.0,
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
        {layer}{template_path}{mask_params}{tilt_range}{max_shifts}{z_rotation}{y_rotation}
        {x_rotation}{cutoff}{interpolation}
        distance_range : tuple of float, default is (3.9, 4.4)
            Range of allowed distance between monomers.
        upsample_factor : int, default is 5
            Upsampling factor of ZNCC landscape. Be careful not to set this parameter too
            large. Calculation will take much longer for larger ``upsample_factor``.
            Doubling ``upsample_factor`` results in 2^6 = 64 times longer calculation time.
        """
        from dask import array as da
        from dask import delayed
        from cylindra._cylindra_ext import ViterbiGrid

        t0 = timer("align_all_viterbi")
        parent = self._get_parent()
        molecules = layer.molecules
        shape_nm = self._get_shape_in_nm()
        loader = parent.tomogram.get_subtomogram_loader(
            molecules, shape=shape_nm, order=interpolation
        )
        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        if max_angle is not None:
            max_angle = np.deg2rad(max_angle)
        max_shifts_px = tuple(s / parent.tomogram.scale for s in max_shifts)
        search_size = tuple(int(px * upsample_factor) * 2 + 1 for px in max_shifts_px)
        _Logger.print(f"Search size (px): {search_size}")

        model = alignment.ZNCCAlignment.with_params(
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            tilt_range=tilt_range,
        )

        score_dsk = loader.construct_landscape(
            template,
            mask=mask,
            max_shifts=max_shifts,
            upsample=upsample_factor,
            alignment_model=model,
        )

        if not model.has_rotation:
            score = score_dsk.compute()
        else:
            tasks = da.max(score_dsk, axis=1)
            argmax = da.argmax(score_dsk, axis=1)
            out = da.compute([tasks, argmax], argmax)[0]
            score, argmax = out

        scale = parent.tomogram.scale
        npf = molecules.features[Mole.pf].max() + 1

        slices = [np.asarray(molecules.features[Mole.pf] == i) for i in range(npf)]
        molecules_origin = molecules.translate_internal(
            -(np.array(max_shifts) - scale) / 2
        )
        # split each protofilament
        mole_list = [molecules_origin.subset(sl) for sl in slices]

        dist_min, dist_max = np.array(distance_range) / scale * upsample_factor
        scores = [score[sl] for sl in slices]

        @delayed
        def _run_viterbi(s: np.ndarray, m: Molecules, dist_min, dist_max, max_angle):
            origin = (m.pos / scale * upsample_factor).astype(np.float32)
            zvec = m.z.astype(np.float32)
            yvec = m.y.astype(np.float32)
            xvec = m.x.astype(np.float32)
            grid = ViterbiGrid(s, origin, zvec, yvec, xvec)
            return grid.viterbi(dist_min, dist_max, max_angle)

        viterbi_tasks = [
            _run_viterbi(s, m, dist_min, dist_max, max_angle)
            for s, m in zip(scores, mole_list)
        ]
        vit_out: list[tuple[np.ndarray, float]] = da.compute(viterbi_tasks)[0]

        offset = (np.array(max_shifts_px) * upsample_factor).astype(np.int32)
        inds = np.empty((len(molecules), 3), dtype=np.int32)
        for i, (shift, _) in enumerate(vit_out):
            _check_viterbi_shift(shift, max_shifts_px, i)
            inds[slices[i], :] = shift
        all_shifts = ((inds - offset) / upsample_factor * scale).astype(np.float32)

        opt_score = np.fromiter(
            (score[i, iz, iy, ix] for i, (iz, iy, ix) in enumerate(inds)),
            dtype=np.float32,
        )
        molecules_opt = molecules.translate_internal(all_shifts)
        if model.has_rotation:
            quats = np.zeros((len(molecules), 4), dtype=np.float32)
            for i, (shift, _) in enumerate(vit_out):
                quats = np.stack(
                    [
                        model.quaternions[argmax[i, iz, iy, ix]]
                        for i, (iz, iy, ix) in enumerate(inds)
                    ],
                    axis=0,
                )

            molecules_opt = molecules_opt.rotate_by_quaternion(quats)

            rotvec = Rotation.from_quat(quats).as_rotvec().astype(np.float32)
            molecules_opt.features = molecules_opt.features.with_columns(
                pl.Series("align-dzrot", rotvec[:, 0]),
                pl.Series("align-dyrot", rotvec[:, 1]),
                pl.Series("align-dxrot", rotvec[:, 2]),
            )

        molecules_opt.features = molecules_opt.features.with_columns(
            pl.Series("align-dz", all_shifts[:, 0]),
            pl.Series("align-dy", all_shifts[:, 1]),
            pl.Series("align-dx", all_shifts[:, 2]),
            pl.Series(Mole.score, opt_score),
        )
        t0.toc()
        parent._need_save = True
        aligned_loader = SubtomogramLoader(
            parent.tomogram.image.value,
            molecules_opt,
            order=interpolation,
            output_shape=template.shape,
        )
        return self._align_all_on_return(aligned_loader, layer)

    @Refinement.wraps
    @set_design(text="2D Boltzmann Alignment")
    @dask_thread_worker.with_progress(descs=_pdesc.align_boltzmann_fmt)
    def align_all_boltzmann(
        self,
        layer: MoleculesLayer,
        template_path: Bound[params.template_path],
        mask_params: Bound[params._get_mask_params] = None,
        tilt_range: Bound[params.tilt_range] = None,
        max_shifts: _MaxShifts = (0.6, 0.6, 0.6),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        distance_range_long: _DistRangeLon = (3.9, 4.4),
        distance_range_lat: _DistRangeLat = (4.7, 5.3),
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        ntrial: int = 6,
    ):
        """
        Subtomogram alignment using 2D Boltzmann alignment.

        2D Boltzmann alignment is an alignment algorithm that considers the distance between
        every adjacent monomers. Two-dimensionally connected optimization can be approximated
        by the simulated annealing algorithm.

        Parameters
        ----------
        {layer}{template_path}{mask_params}{tilt_range}{max_shifts}{cutoff}{interpolation}
        distance_range_long : tuple of float
            Range of allowed distance between longitudianlly consecutive monomers.
        distance_range_lat : tuple of float
            Range of allowed distance between laterally consecutive monomers.
        upsample_factor : int, default is 5
            Upsampling factor of ZNCC landscape.
        """
        from cylindra._cylindra_ext import CylindricAnnealingModel

        t0 = timer("align_all_boltzmann")
        parent = self._get_parent()
        molecules = layer.molecules
        shape_nm = self._get_shape_in_nm()
        loader = parent.tomogram.get_subtomogram_loader(
            molecules, shape=shape_nm, order=interpolation
        )
        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        max_shifts_px = tuple(s / parent.tomogram.scale for s in max_shifts)
        search_size = tuple(int(px * upsample_factor) * 2 + 1 for px in max_shifts_px)
        _ny, _npf, _nrise = utils.infer_geometry_from_molecules(molecules)

        model = alignment.ZNCCAlignment.with_params(
            cutoff=cutoff,
            tilt_range=tilt_range,
        )

        score_dsk = loader.construct_landscape(
            template,
            mask=mask,
            max_shifts=max_shifts,
            upsample=upsample_factor,
            alignment_model=model,
        )

        score: np.ndarray = score_dsk.compute()
        scale = parent.tomogram.scale
        m0 = molecules.translate_internal(-(np.array(max_shifts) - scale) / 2)

        dist_lon = np.array(distance_range_long) / scale * upsample_factor
        dist_lat = np.array(distance_range_lat) / scale * upsample_factor

        energy = -score

        time_const = molecules.pos.size * np.product(search_size)
        initial_temperature = np.std(energy) * 2

        # construct the annealing model
        annealing = (
            CylindricAnnealingModel()
            .set_graph(
                energy.reshape((_ny, _npf) + search_size),
                (m0.pos / scale * upsample_factor).reshape(_ny, _npf, 3),
                m0.z.reshape(_ny, _npf, 3).astype(np.float32),
                m0.y.reshape(_ny, _npf, 3).astype(np.float32),
                m0.x.reshape(_ny, _npf, 3).astype(np.float32),
                _nrise,
            )
            .set_reservoir(
                temperature=initial_temperature,
                time_constant=time_const,
            )
            .set_box_potential(
                *dist_lon,
                *dist_lat,
            )
        )

        results = _get_annealing_results(
            annealing, scale / upsample_factor, initial_temperature, range(ntrial)
        )
        best_model = sorted(results, key=lambda r: r.energy)[0].model

        offset = (np.array(max_shifts_px) * upsample_factor).astype(np.int32)
        inds = best_model.shifts().reshape(-1, 3)
        opt_score = np.fromiter(
            (score[i, iz, iy, ix] for i, (iz, iy, ix) in enumerate(inds)),
            dtype=np.float32,
        )

        all_shifts = ((inds - offset) / upsample_factor * scale).reshape(-1, 3)

        molecules_opt = molecules.translate_internal(all_shifts)
        molecules_opt.features = molecules_opt.features.with_columns(
            pl.Series("align-dz", all_shifts[:, 0]),
            pl.Series("align-dy", all_shifts[:, 1]),
            pl.Series("align-dx", all_shifts[:, 2]),
            pl.Series(Mole.score, opt_score),
        )
        t0.toc()
        parent._need_save = True
        aligned_loader = SubtomogramLoader(
            parent.tomogram.image.value,
            molecules_opt,
            order=interpolation,
            output_shape=template.shape,
        )

        @thread_worker.to_callback
        def _on_return():
            mole = aligned_loader.molecules
            points = parent.add_molecules(
                mole,
                name=_coerce_aligned_name(layer.name, self.parent_viewer),
                source=layer.source_component,
            )
            layer.visible = False
            _Logger.print_html(f"{layer.name!r} &#8594; {points.name!r}")
            with _Logger.set_plt(rc_context={"font.size": 15}):
                import matplotlib.pyplot as plt

                for i, r in enumerate(results):
                    _x = np.arange(r.energies.size) / 1000
                    plt.plot(_x, -r.energies, label=f"{i}", alpha=0.5)
                plt.xlabel("Repeat (x10^6)")
                plt.ylabel("Score")
                plt.legend()
                plt.show()

            @undo_callback
            def out():
                parent._try_removing_layer(points)
                layer.visible = True

            @out.with_redo
            def out():
                parent.parent_viewer.add_layer(points)

            return out

        return _on_return

    @Subtomogram_analysis.wraps
    @set_design(text="Calculate FSC")
    @dask_thread_worker.with_progress(
        desc=_pdesc.fmt_layer_name("Calculating FSC of {!r}")
    )
    def calculate_fsc(
        self,
        layer: MoleculesLayer,
        mask_params: Bound[params._get_mask_params],
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
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
            If true, subtomogram averaging will be shown after FSC calculation.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        t0 = timer("calculate_fsc")
        parent = self._get_parent()
        mole = layer.molecules

        loader = parent.tomogram.get_subtomogram_loader(mole, order=interpolation)
        template, mask = loader.normalize_input(
            template=self.params._get_template(allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        fsc, avg = loader.reshape(
            template=template,
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

        @thread_worker.to_callback
        def _calculate_fsc_on_return():
            _Logger.print_html(f"<b>Fourier Shell Correlation of {layer.name!r}</b>")
            with _Logger.set_plt(rc_context={"font.size": 15}):
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
                _rec_layer: "Image" = self._show_reconstruction(
                    img_avg,
                    name=f"[AVG]{layer.name}",
                )
                _rec_layer.metadata["fsc"] = widget_utils.FscResult(
                    freq, fsc_mean, fsc_std, res0143, res0500
                )

        return _calculate_fsc_on_return

    @Subtomogram_analysis.wraps
    @set_design(text="PCA/K-means classification")
    @dask_thread_worker.with_progress(descs=_pdesc.classify_pca_fmt)
    def classify_pca(
        self,
        layer: MoleculesLayer,
        mask_params: Bound[params._get_mask_params],
        size: Annotated[
            Optional[nm],
            {
                "text": "Use mask shape",
                "options": {"value": 12.0, "max": 100.0},
                "label": "size (nm)",
            },
        ] = None,
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        bin_size: OneOf[_get_available_binsize] = 1,
        n_components: Annotated[int, {"min": 2, "max": 20}] = 2,
        n_clusters: Annotated[int, {"min": 2, "max": 100}] = 2,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
    ):
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
        t0 = timer("classify_pca")
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

        @thread_worker.to_callback
        def _on_return():
            from .pca import PcaViewer

            pca_viewer = PcaViewer(pca)
            pca_viewer.native.setParent(self.native, pca_viewer.native.windowFlags())
            pca_viewer.show()
            self._show_reconstruction(avgs, name=f"[PCA]{layer.name}", store=False)

        return _on_return

    @Subtomogram_analysis.wraps
    @set_design(text="Seam search")
    @dask_thread_worker.with_progress(desc=_pdesc.fmt_layer_name("Seam search of {!r}"))
    def seam_search(
        self,
        layer: MoleculesLayer,
        template_path: Bound[params.template_path],
        mask_params: Bound[params._get_mask_params],
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        npf: Annotated[Optional[int], {"text": "Use global properties"}] = None,
        cutoff: _CutoffFreq = 0.25,
    ):
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
        {cutoff}
        """
        t0 = timer("seam_search")
        parent = self._get_parent()
        mole = layer.molecules
        shape = self._get_shape_in_nm()
        loader = parent.tomogram.get_subtomogram_loader(
            mole, shape, order=interpolation
        )
        if npf is None:
            npf = mole.features[Mole.pf].max() + 1

        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
        )

        corrs, img_ave, all_labels = utils.try_all_seams(
            loader=loader.replace(output_shape=template.shape),
            npf=npf,
            template=ip.asarray(template, axes="zyx"),
            mask=mask,
            cutoff=cutoff,
        )

        # calculate score and the best PF position
        corr1, corr2 = corrs[:npf], corrs[npf:]
        score = np.empty_like(corrs)
        score[:npf] = corr1 - corr2
        score[npf:] = corr2 - corr1
        imax = np.argmax(score)
        layer.features = layer.molecules.features.with_columns(
            pl.Series(Mole.isotype, all_labels[imax].astype(np.uint8))
        )
        layer.metadata["seam-search-score"] = score

        parent._need_save = True

        @thread_worker.to_callback
        def _seam_search_on_return():
            self._show_reconstruction(img_ave, layer.name, store=False)

            # plot all the correlation
            t0.toc()
            _Logger.print_html("<code>Seam_search</code>")
            with _Logger.set_plt(rc_context={"font.size": 15}):
                _Logger.print(f"layer = {layer.name!r}")
                _Logger.print(f"template = {str(template_path)!r}")
                widget_utils.plot_seam_search_result(score, npf)

            self.sub_viewer.layers[-1].metadata["Correlation"] = corrs
            self.sub_viewer.layers[-1].metadata["Score"] = score

        return _seam_search_on_return

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

    @thread_worker.to_callback
    def _align_all_on_return(
        self, aligned_loader: SubtomogramLoader, layer: MoleculesLayer
    ):
        parent = self._get_parent()
        mole = aligned_loader.molecules

        points = parent.add_molecules(
            mole,
            name=_coerce_aligned_name(layer.name, self.parent_viewer),
            source=layer.source_component,
        )
        layer.visible = False
        _Logger.print_html(f"{layer.name!r} &#8594; {points.name!r}")

        @undo_callback
        def out():
            parent._try_removing_layer(points)
            layer.visible = True

        @out.with_redo
        def out():
            parent.parent_viewer.add_layer(points)

        return out


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


class _AnnealingResult(NamedTuple):
    model: "CylindricAnnealingModel"
    energies: "NDArray[np.float32]"
    energy: float


def _get_annealing_results(
    annealing: "CylindricAnnealingModel",
    factor: float,
    initial_temperature: float,
    seeds: Iterable[int],
) -> list[_AnnealingResult]:
    from dask import array as da, delayed

    def _logging_distances(a: "NDArray[np.float32]", prefix: str):
        _mean = a.mean() * factor
        _std = a.std() * factor
        _Logger.print_html(f"{prefix}: {_mean:.2f} &#177; {_std:.2f} nm")

    _logging_distances(annealing.longitudinal_distances(), "Longitudinal")
    _logging_distances(annealing.lateral_distances(), "Lateral")

    @delayed
    def _run(seed: int) -> _AnnealingResult:
        _model = annealing.with_seed(seed)
        energies = [_model.energy()]
        while (
            _model.temperature() > initial_temperature * 1e-4
            and _model.optimization_state() == "not_converged"
        ):
            _model.simulate(1000)
            energies.append(_model.energy())
        return _AnnealingResult(_model, np.array(energies), energies[-1])

    tasks = [_run(s) for s in seeds]
    results: list[_AnnealingResult] = da.compute(tasks)[0]
    if all(result.model.optimization_state() == "failed" for result in results):
        raise RuntimeError(
            "Failed to optimize for all trials. You may check the distance range."
        )
    elif not any(
        result.model.optimization_state() == "converged" for result in results
    ):
        _Logger.print("Optimization did not converge for any trial.")

    _Logger.print_table(
        {
            "Iteration": [r.model.iteration() for r in results],
            "Score": [-r.energy for r in results],
            "State": [r.model.optimization_state() for r in results],
        }
    )
    return results
