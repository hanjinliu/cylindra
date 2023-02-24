from typing import Annotated, Union

import re
from acryo import BatchLoader, pipe

from magicgui.widgets import Container
from magicclass import (
    magicclass, do_not_record, field, vfield, MagicTemplate, set_design, abstractapi
)
from magicclass.types import OneOf, Optional, Bound, ExprStr, Path
from magicclass.utils import thread_worker
from magicclass.widgets import ConsoleTextEdit, HistoryFileEdit
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.polars import DataFrameView

import numpy as np
import impy as ip
import polars as pl
import napari

from cylindra.const import nm, ALN_SUFFIX, MoleculesHeader as Mole
from cylindra.utils import roundint

from .. import widget_utils
from ..widget_utils import FileFilter
from ..sta import INTERPOLATION_CHOICES, METHOD_CHOICES, MASK_CHOICES, _get_alignment

from .menus import BatchSubtomogramAnalysis, BatchRefinement
from ._loaderlist import LoaderList, LoaderInfo

# annotated types
_CutoffFreq = Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.05}]
_ZRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_YRotation = Annotated[tuple[float, float], {"options": {"max": 180.0, "step": 0.1}}]
_XRotation = Annotated[tuple[float, float], {"options": {"max": 90.0, "step": 0.1}}]
_MaxShifts = Annotated[tuple[nm, nm, nm], {"options": {"max": 10.0, "step": 0.1}, "label": "Max shifts (nm)"}]
_SubVolumeSize = Annotated[Optional[nm], {"text": "Use template shape", "options": {"value": 12., "max": 100.}, "label": "size (nm)"}]

POLARS_NAMESPACE = {"pl": pl, "int": int, "float": float, "str": str, "np": np, "__builtins__": {}}

@magicclass(layout="horizontal", widget_type="groupbox", name="Parameters", visible=False, record=False)
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
    template_path = vfield(Optional[Annotated[Path.Read[FileFilter.IMAGE], {"widget_type": HistoryFileEdit}]], label="Template").with_options(
        text="Use last averaged image", value=Path("")
    )
    mask_choice = vfield(OneOf[MASK_CHOICES], label="Mask", record=False)
    params = field(MaskParameters, name="Mask parameters")
    mask_path = field(mask_path)
    tilt_range = vfield(Optional[tuple[nm, nm]], label="Tilt range (deg)", record=False).with_options(
        value=(-60., 60.), text="No missing-wedge", options={"options": {"min": -90.0, "max": 90.0, "step": 1.0}}
    )
    
    _last_average: ip.ImgArray = None  # the global average result

    def __post_init__(self):
        self._template: ip.ImgArray= None
        self._viewer: "napari.Viewer | None" = None
        self.mask_choice = MASK_CHOICES[0]

    @mask_choice.connect
    def _on_mask_switch(self):
        v = self.mask_choice
        self.params.visible = (v == MASK_CHOICES[1])
        self.mask_path.visible = (v == MASK_CHOICES[2])

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
    
    def _show_reconstruction(self, image: ip.ImgArray, name: str, store: bool = True):
        from cylindra import instance
        ui = instance()
        return ui.sta._show_reconstruction(image, name, store)
        

@magicclass(name="Batch Subtomogram Analysis")
class BatchSubtomogramAveraging(MagicTemplate):
    def _get_parent(self):
        from .main import CylindraBatchWidget
        
        return self.find_ancestor(CylindraBatchWidget, cache=True)

    def _get_loader_names(self, _=None) -> list[str]:
        try:
            parent = self._get_parent()
        except Exception:
            return []
        return [info.name for info in parent._loaders]
    
    # Menus
    BatchSubtomogramAnalysis = field(BatchSubtomogramAnalysis, name="Subtomogram Analysis")
    BatchRefinement = field(BatchRefinement, name="Refinement")

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class Header(MagicTemplate):
        loader_name = abstractapi()
        show_loader_info = abstractapi()
        remove_loader = abstractapi()
        
    loader_name = Header.vfield(str).with_choices(choices=_get_loader_names)
    
    def _get_current_loader_name(self, _=None) -> str:
        return self.loader_name
    
    @Header.wraps
    @set_design(text="??", max_width=36)
    @do_not_record
    def show_loader_info(self):
        """Show information about this loader"""
        loaderlist = self._get_parent()._loaders
        info = loaderlist[self.loader_name]
        loader = info.loader
        img_info = "\n" + "\n".join(
            f"{img_id}: {img_path}" for img_id, img_path in info.image_paths.items()
        )
            
        info_text = (
            f"name: {info.name}\nmolecule: n={loader.count()}\nimages:{img_info}"
        )
        view = DataFrameView(value=loader.molecules.to_dataframe())
        txt = ConsoleTextEdit(value=info_text)
        txt.read_only = True
        cnt = Container(
            widgets=[txt, view],
            layout="horizontal",
            labels=False
        )
        cnt.native.setParent(self.native, cnt.native.windowFlags())
        cnt.show()
    
    @Header.wraps
    @set_design(text="✕", max_width=36)
    def remove_loader(self, loader_name: Bound[_get_current_loader_name]):
        """Remove this loader"""
        loaderlist = self._get_parent()._loaders
        del loaderlist[loader_name]

    params = StaParameters
    
    @BatchSubtomogramAnalysis.wraps
    @set_design(text="Filter loader")
    def filter_loader(
        self,
        loader_name: Bound[_get_current_loader_name],
        expression: ExprStr.In[POLARS_NAMESPACE],
    ):
        """
        Filter the selected loader and add the filtered one to the list.

        Parameters
        ----------
        loader_name : str
            Name of the input loader
        expression : str
            polars expression that will be used to filter the loader. For example,
            `pl.col("score") > 0.7` will filter out all low-score molecules.
        """
        if expression == "":
            raise ValueError("Predicate is not given.")
        loaderlist = self._get_parent()._loaders
        info = loaderlist[loader_name]
        loader = info.loader
        pl_expr: pl.Expr = eval(str(expression), POLARS_NAMESPACE, {})
        new = loader.filter(pl_expr)
        existing_id = set(new.features[Mole.image])
        loaderlist.append(
            LoaderInfo(
                new,
                name=f"{info.name}-Filt",
                image_paths={k: v for k, v in info.image_paths.items() if v in existing_id},
            )
        )
        return None

    @BatchSubtomogramAnalysis.wraps
    @set_design(text="Average all molecules")
    @dask_thread_worker.with_progress(desc="Averaging all molecules in projects")
    def average_all(
        self,
        loader_name: Bound[_get_current_loader_name],
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
    ):
        loaderlist = self._get_parent()._loaders
        loader = loaderlist[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape, order=interpolation)
        img = ip.asarray(loader.average(), axes="zyx")
        img.set_scale(zyx=loader.scale)
        
        return thread_worker.to_callback(
            self.params._show_reconstruction, img, f"[AVG]Collection"
        )
    
    @BatchRefinement.wraps
    @set_design(text="Align all molecules")
    @dask_thread_worker.with_progress(desc="Aligning all molecules")
    def align_all(
        self,
        loader_name: Bound[_get_current_loader_name],
        template_path: Bound[params.template_path],
        mask_params: Bound[params._get_mask_params],
        tilt_range: Bound[params.tilt_range] = None,
        max_shifts: _MaxShifts = (1., 1., 1.),
        z_rotation: _ZRotation = (0., 0.),
        y_rotation: _YRotation = (0., 0.),
        x_rotation: _XRotation = (0., 0.),
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
    ):
        loaderlist = self._get_parent()._loaders
        info = loaderlist[loader_name]
        loader = info.loader
        template, mask = loader.normalize_input(
            template=self.params._get_template(path=template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        loader = loader.replace(output_shape=template.shape, order=interpolation)
        aligned = loader.align(
            template=template,
            mask=mask,
            max_shifts=max_shifts,
            rotations=(z_rotation, y_rotation, x_rotation),
            cutoff=cutoff,
            alignment_model= _get_alignment(method),
            tilt_range=tilt_range,
        )
        loaderlist.append(
            LoaderInfo(
                aligned,
                name=_coerce_aligned_name(info.name, loaderlist),
                image_paths=info.image_paths,
            )
        )
        return None

    @BatchSubtomogramAnalysis.wraps
    @set_design(text="Calculate FSC")
    @dask_thread_worker.with_progress(desc="Calculating FSC")
    def calculate_fsc(
        self,
        loader_name: Bound[_get_current_loader_name],
        mask_params: Bound[params._get_mask_params],
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        n_set: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: Annotated[Optional[float], {"label": "Frequency precision", "text": "Choose proper value", "options": {"min": 0.005, "max": 0.1, "step": 0.005, "value": 0.02}}] = None,
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
        loaderlist = self._get_parent()._loaders
        loader = loaderlist[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape, order=interpolation)
        _, mask = loader.normalize_input(
            template=self.params._get_template(allow_none=True),
            mask=self.params._get_mask(params=mask_params)
        )
    
        if mask is None:
            mask = 1.
        if dfreq is None:
            dfreq = 1.5 / min(shape) * loader.scale
        img = ip.asarray(
            loader.average_split(n_set=n_set, seed=seed, squeeze=False),
            axes="ipzyx",
        )
        
        # NOTE: images added with a big constant offset cause strong correlation at the
        # masked edges. Here to avoid it, normalize images to minimize the artifact.
        img -= img.mean()
        img: ip.ImgArray

        fsc_all: list[np.ndarray] = []
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = ip.fsc(img0 * mask, img1 * mask, dfreq=dfreq)
            fsc_all.append(fsc)
        if show_average:
            img_avg = ip.asarray(img[0, 0] + img[0, 1], axes="zyx") / img.shape.i
            img_avg.set_scale(zyx=loader.scale)
        else:
            img_avg = None

        fsc_all = np.stack(fsc_all, axis=1)
        
        @thread_worker.to_callback
        def _calculate_fsc_on_return():
            from cylindra import instance
            fsc_mean = np.mean(fsc_all, axis=1)
            fsc_std = np.std(fsc_all, axis=1)
            crit_0143 = 0.143
            crit_0500 = 0.500
            
            parent = instance()
            parent.log.print_html("<b>Fourier Shell Correlation of the collection</b>")
            with parent.log.set_plt(rc_context={"font.size": 15}):
                widget_utils.plot_fsc(freq, fsc_mean, fsc_std, [crit_0143, crit_0500], loader.scale)
            
            resolution_0143 = widget_utils.calc_resolution(freq, fsc_mean, crit_0143, loader.scale)
            resolution_0500 = widget_utils.calc_resolution(freq, fsc_mean, crit_0500, loader.scale)
            
            parent.log.print_html(f"Resolution at FSC=0.5 ... <b>{resolution_0500:.3f} nm</b>")
            parent.log.print_html(f"Resolution at FSC=0.143 ... <b>{resolution_0143:.3f} nm</b>")
            parent._LoggerWindow.show()
            
            if img_avg is not None:
                _rec_layer = self.params._show_reconstruction(
                    img_avg, name = "[AVG]Collection",
                )
                _rec_layer.metadata["fsc"] = widget_utils.FscResult(
                    freq, fsc_mean, fsc_std, resolution_0143, resolution_0500
                )
        return _calculate_fsc_on_return
    
    
    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class Buttons(MagicTemplate):
        show_template = abstractapi()
        show_mask = abstractapi()
    
    @Buttons.wraps
    @set_design(text="Show template")
    @do_not_record
    def show_template(self):
        """Load and show template image in the scale of the tomogram."""
        self.params._show_reconstruction(self.template, name="Template image", store=False)
    
    @Buttons.wraps
    @set_design(text="Show mask")
    @do_not_record
    def show_mask(self):
        """Load and show mask image in the scale of the tomogram."""
        self.params._show_reconstruction(self.mask, name="Mask image", store=False)
    
    @property
    def template(self) -> "ip.ImgArray | None":
        """Template image."""
        loader = self._get_parent()._loaders[self.loader_name].loader
        template, _ = loader.normalize_input(self.params._get_template())
        return ip.asarray(template, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
    
    @property
    def mask(self) -> "ip.ImgArray | None":
        """Mask image."""
        loader = self._get_parent()._loaders[self.loader_name].loader
        _, mask = loader.normalize_input(
            self.params._get_template(allow_none=True), self.params._get_mask()
        )
        return ip.asarray(mask, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
    
    
    def _get_shape_in_px(self, default: "nm | None", loader: BatchLoader) -> tuple[int, ...]:
        if default is None:
            tmp = loader.normalize_template(self.params._get_template())
            return tmp.shape
        else:
            return (roundint(default / loader.scale),) * 3

def _coerce_aligned_name(name: str, loaders: LoaderList):
    num = 1
    if re.match(fr".*-{ALN_SUFFIX}(\d)+", name):
        try:
            *pre, suf = name.split(f"-{ALN_SUFFIX}")
            num = int(suf) + 1
            name = "".join(pre)
        except Exception:
            num = 1

    existing_names = set(info.name for info in loaders)
    while name + f"-{ALN_SUFFIX}{num}" in existing_names:
        num += 1
    return name + f"-{ALN_SUFFIX}{num}"

