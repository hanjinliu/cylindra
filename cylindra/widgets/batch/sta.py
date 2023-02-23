from typing import Annotated, TYPE_CHECKING

import re
from acryo import BatchLoader

from magicgui.widgets import Container
from magicclass import (
    magicclass, do_not_record, field, vfield, MagicTemplate, set_design, abstractapi
)
from magicclass.types import OneOf, Optional, Bound, ExprStr
from magicclass.utils import thread_worker
from magicclass.widgets import ConsoleTextEdit
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.polars import DataFrameView

import numpy as np
import impy as ip
import polars as pl

from cylindra.const import nm, ALN_SUFFIX
from cylindra.utils import roundint

from .. import widget_utils
from ..sta import StaParameters, INTERPOLATION_CHOICES, METHOD_CHOICES, _get_alignment

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
        if (parent := info.get_parent()) is not None:
            parent_name = parent.name
        else:
            parent_name = "None"
        if info.paths:
            paths_list: list[str] = []
            for imgpath, moles in info.paths:
                paths_list.append(f"  - {imgpath}")
                for molepath in moles:
                    paths_list.append(f"    - {molepath}")
            paths = "\n" + "\n".join(paths_list)
        else:
            paths = "None"
        info_text = (
            f"name: {info.name}\npaths: {paths}\npredicate: {info.predicate!r}\n"
            f"parent: {parent_name}\nmolecule: n={loader.count()}"
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
        loaderlist.append(
            LoaderInfo(
                new,
                name=f"{info.name}-Filt",
                predicate=expression,
                parent=loader,
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
                parent=loader
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

