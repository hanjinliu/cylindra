import re
from typing import Annotated, Any

import impy as ip
import numpy as np
from acryo import BatchLoader
from magicclass import (
    MagicTemplate,
    abstractapi,
    do_not_record,
    field,
    magicclass,
    magictoolbar,
    nogui,
    set_design,
    setup_function_gui,
    vfield,
)
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.polars import DataFrameView
from magicclass.logging import getLogger
from magicclass.types import Optional, Path
from magicclass.utils import thread_worker
from magicclass.widgets import ConsoleTextEdit
from magicgui.widgets import Container

from cylindra import _shared_doc
from cylindra.const import ALN_SUFFIX, nm
from cylindra.const import MoleculesHeader as Mole
from cylindra.core import ACTIVE_WIDGETS
from cylindra.utils import roundint
from cylindra.widget_utils import FscResult, PolarsExprStr, norm_expr, timer
from cylindra.widgets._annotated import FSCFreq
from cylindra.widgets._widget_ext import RotationsEdit
from cylindra.widgets.batch._loaderlist import LoaderList
from cylindra.widgets.batch.menus import (
    BatchLoaderMenu,
    BatchRefinement,
    BatchSubtomogramAnalysis,
)
from cylindra.widgets.sta import (
    INTERPOLATION_CHOICES,
    METHOD_CHOICES,
    StaParameters,
    _get_alignment,
)


def _classify_pca_fmt():
    yield "(0/5) Caching all the subtomograms"
    yield "(1/5) Creating template image for PCA clustering"
    yield "(2/5) Fitting PCA model"
    yield "(3/5) Transforming all the images"
    yield "(4/5) Creating average images for each cluster"
    yield "(5/5) Finishing"


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
    },
]
_BINSIZE = Annotated[int, {"choices": [1, 2, 3, 4, 5, 6, 7, 8]}]

_Logger = getLogger("cylindra")


@magicclass(name="Batch Subtomogram Analysis")
@_shared_doc.update_cls
class BatchSubtomogramAveraging(MagicTemplate):
    def _get_parent(self):
        from .main import CylindraBatchWidget

        return self.find_ancestor(CylindraBatchWidget, cache=True)

    def _get_loader_names(self, _=None) -> list[str]:
        try:
            parent = self._get_parent()
        except Exception:
            return []
        return parent.loader_infos.names()

    # Menus
    BatchSubtomogramAnalysis = field(
        BatchSubtomogramAnalysis, name="Subtomogram Analysis"
    )
    BatchRefinement = field(BatchRefinement, name="Refinement")
    BatchLoaderMenu = field(BatchLoaderMenu, name="Loader")

    @magicclass(layout="horizontal", properties={"margins": (0, 0, 0, 0)})
    class Header(MagicTemplate):
        loader_name = abstractapi()
        show_loader_info = abstractapi()
        remove_loader = abstractapi()

    loader_name = vfield(str, location=Header, record=False).with_choices(
        choices=_get_loader_names
    )

    def _get_current_loader_name(self, _=None) -> str:
        return self.loader_name

    @set_design(text="??", max_width=36, location=Header)
    @do_not_record
    def show_loader_info(self):
        """Show information about this loader"""
        info = self._get_parent().loader_infos[self.loader_name]
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
        cnt = Container(widgets=[txt, view], layout="horizontal", labels=False)
        cnt.native.setParent(self.native, cnt.native.windowFlags())
        cnt.show()

    @set_design(text="✕", max_width=36, location=Header)
    def remove_loader(
        self, loader_name: Annotated[str, {"bind": _get_current_loader_name}]
    ):
        """Remove this loader"""
        del self._get_parent().loader_infos[loader_name]

    params = field(StaParameters)

    def _get_selected_loader_choice(self, *_) -> list[str]:
        try:
            loader = self.get_loader(self.loader_name)
            return loader.molecules.features.columns
        except Exception:
            return []

    def _get_template_path(self, _=None):
        return self.params.template_path.value

    def _get_mask_params(self, _=None):
        return self.params._get_mask_params()

    @set_design(text="Split loader", location=BatchLoaderMenu)
    def split_loader(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        by: Annotated[str, {"choices": _get_selected_loader_choice}],
        delete_old: bool = False,
    ):
        """
        Split the selected loader by the values of the given column.

        Parameters
        ----------
        loader_name : str
            Name of the input loader
        by : str
            Column name to split the loader
        delete_old : bool, default False
            If true, the original loader will be deleted.
        """
        parent = self._get_parent()
        loaders = parent._loaders
        batch_info = loaders[loader_name]
        batch_loader = batch_info.loader
        n_unique = batch_loader.molecules.features[by].n_unique()
        if n_unique > 48:
            raise ValueError(
                f"Too many groups ({n_unique}). Did you choose a float column?"
            )
        for _key, loader in batch_loader.groupby(by):
            existing_id = set(loader.features[Mole.image])
            image_paths = {
                k: v for k, v in batch_info.image_paths.items() if v in existing_id
            }
            invert = {k: v for k, v in batch_info.invert.items() if v in existing_id}
            parent._add_loader(loader, f"{loader_name}_{_key}", image_paths, invert)

        if delete_old:
            idx = -1
            for i, info in enumerate(loaders):
                if info.loader is batch_loader:
                    idx = i
                    break
            else:
                idx = -1
            if idx < 0:
                raise RuntimeError("Loader not found.")
            del loaders[idx]

    @set_design(text="Filter loader", location=BatchLoaderMenu)
    def filter_loader(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        expression: PolarsExprStr,
    ):
        """
        Filter the selected loader and add the filtered one to the list.

        Parameters
        ----------
        loader_name : str
            Name of the input loader
        expression : str
            polars expression that will be used to filter the loader. For example,
            `col("score") > 0.7` will filter out all low-score molecules.
        """
        loaderlist = self._get_parent()._loaders
        info = loaderlist[loader_name]
        loader = info.loader
        new = loader.filter(norm_expr(expression))
        existing_id = set(new.features[Mole.image])
        loaderlist.add_loader(
            new,
            name=f"{info.name}-Filt",
            image_paths={k: v for k, v in info.image_paths.items() if v in existing_id},
            invert={k: v for k, v in info.invert.items() if v in existing_id},
        )
        return None

    @nogui
    def get_loader(self, name: str) -> BatchLoader:
        """Return the acryo.BatchLoader object with the given name"""
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name).__name__}")
        return self._get_parent().loader_infos[name].loader

    @set_design(text="Average all molecules", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(desc="Averaging all molecules in projects")
    def average_all(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: _BINSIZE = 1,
    ):
        t0 = timer()
        loader = self._get_parent().loader_infos[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        img = ip.asarray(
            loader.replace(output_shape=shape, order=interpolation)
            .binning(bin_size, compute=False)
            .average(),
            axes="zyx",
        ).set_scale(zyx=loader.scale * bin_size, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{loader_name}")

    @set_design(text="Average group-wise", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(desc="Grouped subtomogram averaging")
    def average_groups(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        size: _SubVolumeSize = None,
        by: PolarsExprStr = "col('pf-id')",
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: _BINSIZE = 1,
    ):
        """
        Groupwise subtomogram averaging using molecules grouped by the given expression.

        This method first group molecules by its features, and then average each group.
        This method is useful for such as get average of each protofilament and
        segmented subtomogram averaging.

        Parameters
        ----------
        {loader_name}{size}
        by : str or polars expression
            Expression to group molecules.
        {interpolation}{bin_size}
        """
        t0 = timer()
        loader = self._get_parent().loader_infos[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        img = ip.asarray(
            loader.replace(output_shape=shape, order=interpolation)
            .binning(bin_size, compute=False)
            .groupby(norm_expr(by))
            .average()
            .value_stack(axis=0),
            axes="pzyx",
        ).set_scale(zyx=loader.scale * bin_size, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[AVG]{loader_name}", store=False)

    @set_design(text="Split and average molecules", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(desc="Split-and-average")
    def split_and_average(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        n_pairs: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        size: _SubVolumeSize = None,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        bin_size: _BINSIZE = 1,
    ):
        """
        Split molecules into two groups and average separately.

        Parameters
        ----------
        {loader_name}{size}
        n_pairs : int, default is 1
            How many pairs of average will be calculated.
        {size}{interpolation}{bin_size}
        """
        t0 = timer()
        loader = self._get_parent().loader_infos[loader_name].loader
        shape = self._get_shape_in_px(size, loader)

        axes = "ipzyx" if n_pairs > 1 else "pzyx"
        img = ip.asarray(
            loader.replace(output_shape=shape, order=interpolation)
            .binning(bin_size, compute=False)
            .average_split(n_pairs),
            axes=axes,
        ).set_scale(zyx=loader.scale * bin_size, unit="nm")
        t0.toc()
        return self._show_rec.with_args(img, f"[Split]{loader_name}", store=False)

    @set_design(text="Align all molecules", location=BatchRefinement)
    @dask_thread_worker.with_progress(desc="Aligning all molecules")
    def align_all(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        max_shifts: _MaxShifts = (1.0, 1.0, 1.0),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        bin_size: _BINSIZE = 1,
    ):  # fmt: skip
        """
        Align all the molecules in the selected loader.

        Parameters
        ----------
        {loader_name}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{method}{bin_size}
        """
        t0 = timer()
        loaderlist = self._get_parent()._loaders
        info = loaderlist[loader_name]
        loader = info.loader
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        aligned = (
            loader.replace(output_shape=template.shape, order=interpolation)
            .binning(bin_size, compute=False)
            .align(
                template=template,
                mask=mask,
                max_shifts=max_shifts,
                rotations=rotations,
                cutoff=cutoff,
                alignment_model=_get_alignment(method),
            )
        )
        loaderlist.add_loader(
            aligned,
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()
        return None

    @set_design(text="Calculate FSC", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(desc="Calculating FSC")
    def calculate_fsc(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        template_path: Annotated[str | Path | None, {"bind": _get_template_path}] = None,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 1,
        n_pairs: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: FSCFreq = None,
    ):  # fmt: skip
        """
        Calculate Fourier Shell Correlation using the selected loader.

        Parameters
        ----------
        {loader_name}{template_path}{mask_params}{size}
        seed : int, optional
            Random seed used for subtomogram sampling.
        {interpolation}
        n_pairs : int, default 1
            How many sets of image pairs will be generated to average FSC.
        show_average : bool, default True
            If true, subtomogram averaging will be shown after FSC calculation.
        dfreq : float, default 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be
            calculated at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        t0 = timer()
        loader = (
            self._get_parent()
            .loader_infos[loader_name]
            .loader.replace(order=interpolation)
        )

        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path, allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )

        fsc, avg = loader.reshape(
            template=template if size is None else None,
            mask=mask,
            shape=None if size is None else self._get_shape_in_px(size, loader),
        ).fsc_with_average(mask=mask, seed=seed, n_set=n_pairs, dfreq=dfreq)

        if show_average:
            img_avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale)
        else:
            img_avg = None

        result = FscResult.from_dataframe(fsc, loader.scale)
        criteria = [0.5, 0.143]
        t0.toc()

        @thread_worker.callback
        def _calculate_fsc_on_return():
            _Logger.print_html(f"<b>Fourier Shell Correlation of {loader_name!r}</b>")
            with _Logger.set_plt():
                result.plot(criteria)
            for _c in criteria:
                _r = result.get_resolution(_c)
                _Logger.print_html(f"Resolution at FSC={_c:.3f} ... <b>{_r:.3f} nm</b>")

            if img_avg is not None:
                _rec_layer = self._show_rec(img_avg, name=f"[AVG]{loader_name}")
                _rec_layer.metadata["fsc"] = result

        return _calculate_fsc_on_return

    @set_design(text="PCA/K-means classification", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(descs=_classify_pca_fmt)
    def classify_pca(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        template_path: Annotated[str | Path | None, {"bind": _get_template_path}] = None,
        mask_params: Annotated[Any, {"bind": _get_mask_params}] = None,
        size: Annotated[Optional[nm], {"text": "Use mask shape", "options": {"value": 12.0, "max": 100.0}, "label": "size (nm)"}] = None,
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        bin_size: _BINSIZE = 1,
        n_components: Annotated[int, {"min": 2, "max": 20}] = 2,
        n_clusters: Annotated[int, {"min": 2, "max": 100}] = 2,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
    ):  # fmt: skip
        """
        Classify molecules in the loader using PCA and K-means clustering.

        Parameters
        ----------
        {loader_name}{template_path}{mask_params}{size}{cutoff}{interpolation}{bin_size}
        n_components : int, default 2
            The number of PCA dimensions.
        n_clusters : int, default 2
            The number of clusters.
        seed : int, default
            Random seed.
        """
        from cylindra.widgets.subwidgets import PcaViewer

        t0 = timer()
        loader = self._get_parent().loader_infos[loader_name].loader
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path, allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        shape = None
        if mask is None:
            shape = self._get_shape_in_px(size, loader)
        out, pca = (
            loader.reshape(
                template=template if mask is None and shape is None else None,
                mask=mask,
                shape=shape,
            )
            .replace(order=interpolation)
            .binning(binsize=bin_size, compute=False)
            .classify(
                mask=mask,
                seed=seed,
                cutoff=cutoff,
                n_components=n_components,
                n_clusters=n_clusters,
                label_name="cluster",
            )
        )

        avgs = ip.asarray(
            out.groupby("cluster").average().value_stack(axis=0),
            axes=["cluster", "z", "y", "x"],
        ).set_scale(zyx=loader.scale, unit="nm")

        t0.toc()

        @thread_worker.callback
        def _on_return():
            loader.molecules.features = out.molecules.features
            pca_viewer = PcaViewer(pca)
            pca_viewer.native.setParent(self.native, pca_viewer.native.windowFlags())
            pca_viewer.show()
            self._show_rec(avgs, name=f"[PCA]{loader_name}", store=False)

            ACTIVE_WIDGETS.add(pca_viewer)

        return _on_return

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
        _input = self.params._get_template_input(allow_multiple=True)
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
        loader = self.get_loader(self.loader_name)
        _, mask = loader.normalize_input(
            self.params._norm_template_param(
                self.params._get_template_input(allow_multiple=False),
                allow_none=True,
            ),
            self.params._get_mask(),
        )
        if mask is None:
            raise ValueError("No mask to show.")
        mask = ip.asarray(mask, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        self._show_rec(mask, name="Mask image", store=False)

    @thread_worker.callback
    def _show_rec(self, img: ip.ImgArray, name: str, store: bool = True):
        return self.params._show_reconstruction(img, name, store)

    def _get_shape_in_px(
        self, default: "nm | None", loader: BatchLoader
    ) -> tuple[int, ...]:
        if default is None:
            tmp = self._get_template_image()
            return tmp.sizesof("zyx")
        else:
            return (roundint(default / loader.scale),) * 3

    def _get_template_image(self) -> ip.ImgArray:
        scale = self.get_loader(self.loader_name).scale

        template = self.params._norm_template_param(
            self.params._get_template_input(allow_multiple=True),
            allow_none=False,
            allow_multiple=True,
        ).provide(scale)
        if isinstance(template, list):
            template = ip.asarray(np.stack(template, axis=0), axes="zyx")
        else:
            template = ip.asarray(template, axes="zyx")
        return template.set_scale(zyx=scale, unit="nm")

    @setup_function_gui(split_loader)
    def _(self, gui):
        gui[0].changed.connect(gui[1].reset_choices)


def _coerce_aligned_name(name: str, loaders: LoaderList):
    num = 1
    if re.match(rf".*-{ALN_SUFFIX}(\d)+$", name):
        try:
            name, suf = name.rsplit(f"-{ALN_SUFFIX}", 1)
            num = int(suf) + 1
        except Exception:
            num = 1

    existing_names = {info.name for info in loaders}
    while name + f"-{ALN_SUFFIX}{num}" in existing_names:
        num += 1
    return name + f"-{ALN_SUFFIX}{num}"
