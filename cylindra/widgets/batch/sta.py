from typing import Annotated, Any

import re
from acryo import BatchLoader

from magicgui.widgets import Container
from magicclass import (
    magicclass,
    do_not_record,
    field,
    magictoolbar,
    nogui,
    vfield,
    MagicTemplate,
    set_design,
    abstractapi,
    setup_function_gui,
)
from magicclass.types import OneOf, Optional, Path
from magicclass.utils import thread_worker
from magicclass.logging import getLogger
from magicclass.widgets import ConsoleTextEdit
from magicclass.ext.dask import dask_thread_worker
from magicclass.ext.polars import DataFrameView

import impy as ip

from cylindra.const import nm, ALN_SUFFIX, MoleculesHeader as Mole
from cylindra.core import ACTIVE_WIDGETS
from cylindra.utils import roundint
from cylindra.widgets import _shared_doc
from cylindra.widgets._widget_ext import RotationsEdit
from cylindra.widgets._annotated import FSCFreq
from cylindra.widgets.main import widget_utils
from cylindra.widgets.sta import StaParameters
from cylindra.widgets.widget_utils import timer, PolarsExprStr, norm_expr
from cylindra.widgets.sta import (
    INTERPOLATION_CHOICES,
    METHOD_CHOICES,
    _get_alignment,
)

from .menus import BatchLoaderMenu, BatchSubtomogramAnalysis, BatchRefinement
from ._loaderlist import LoaderList, LoaderInfo


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
_BINSIZE = OneOf[(1, 2, 3, 4, 5, 6, 7, 8)]

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
        return [info.name for info in parent._loaders]

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

    loader_name = vfield(str, location=Header).with_choices(choices=_get_loader_names)

    def _get_current_loader_name(self, _=None) -> str:
        return self.loader_name

    @set_design(text="??", max_width=36, location=Header)
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
        cnt = Container(widgets=[txt, view], layout="horizontal", labels=False)
        cnt.native.setParent(self.native, cnt.native.windowFlags())
        cnt.show()

    @set_design(text="✕", max_width=36, location=Header)
    def remove_loader(
        self, loader_name: Annotated[str, {"bind": _get_current_loader_name}]
    ):
        """Remove this loader"""
        loaderlist = self._get_parent()._loaders
        del loaderlist[loader_name]

    params = field(StaParameters)

    def _get_selected_loader_choice(self, *_) -> list[str]:
        try:
            loader = self.get_loader(self.loader_name)
            return loader.molecules.features.columns
        except Exception:
            return []

    def _get_template_path(self, _=None):
        return self.params.template_path

    def _get_mask_params(self, _=None):
        return self.params._get_mask_params()

    @set_design(text="Split loader", location=BatchLoaderMenu)
    def split_loader(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        by: Annotated[str, {"choices": _get_selected_loader_choice}],
        delete_old: bool = False,
    ):
        parent = self._get_parent()
        batch_info = parent._loaders[loader_name]
        batch_loader = batch_info.loader
        n_unique = batch_loader.molecules.features[by].n_unique()
        if n_unique > 48:
            raise ValueError(
                f"Too many groups ({n_unique}). Did you choose a float column?"
            )
        loaders = parent._loaders
        for _key, loader in batch_loader.groupby(by):
            existing_id = set(loader.features[Mole.image])
            image_paths = {
                k: v for k, v in batch_info.image_paths.items() if v in existing_id
            }
            parent._add_loader(loader, f"{loader_name}_{_key}", image_paths)

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
        loaderlist.append(
            LoaderInfo(
                new,
                name=f"{info.name}-Filt",
                image_paths={
                    k: v for k, v in info.image_paths.items() if v in existing_id
                },
            )
        )
        return None

    @nogui
    def get_loader(self, name: str) -> BatchLoader:
        """Return the acryo.BatchLoader object with the given name"""
        info = self._get_parent()._loaders[name]
        return info.loader

    @set_design(text="Average all molecules", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(desc="Averaging all molecules in projects")
    def average_all(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        size: _SubVolumeSize = None,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        bin_size: _BINSIZE = 1,
    ):
        t0 = timer()
        loaderlist = self._get_parent()._loaders
        loader = loaderlist[loader_name].loader
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
        Group-wise subtomogram averaging using molecules grouped by the given expression.

        This method first group molecules by its features, and then average each group.
        This method is useful for such as get average of each protofilament and segmented
        subtomogram averaging.

        Parameters
        ----------
        {loader_name}{size}
        by : str or polars expression
            Expression to group molecules.
        {interpolation}{bin_size}
        """
        t0 = timer()
        loaderlist = self._get_parent()._loaders
        loader = loaderlist[loader_name].loader
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
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        method: OneOf[METHOD_CHOICES] = "zncc",
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
        loaderlist.append(
            LoaderInfo(
                aligned,
                name=_coerce_aligned_name(info.name, loaderlist),
                image_paths=info.image_paths,
            )
        )
        t0.toc()
        return None

    @set_design(text="Calculate FSC", location=BatchSubtomogramAnalysis)
    @dask_thread_worker.with_progress(desc="Calculating FSC")
    def calculate_fsc(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: _SubVolumeSize = None,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 1,
        n_pairs: Annotated[int, {"min": 1, "label": "number of image pairs"}] = 1,
        show_average: bool = True,
        dfreq: FSCFreq = None,
    ):
        """
        Calculate Fourier Shell Correlation using the selected loader.

        Parameters
        ----------
        {loader_name}{mask_params}{size}
        seed : int, optional
            Random seed used for subtomogram sampling.
        {interpolation}
        n_pairs : int, default is 1
            How many sets of image pairs will be generated to average FSC.
        show_average : bool, default is True
            If true, subtomogram averaging will be shown after FSC calculation.
        dfreq : float, default is 0.02
            Precision of frequency to calculate FSC. "0.02" means that FSC will be calculated
            at frequency 0.01, 0.03, 0.05, ..., 0.45.
        """
        t0 = timer()
        loaderlist = self._get_parent()._loaders
        loader = loaderlist[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape, order=interpolation)
        if mask_params is None:
            mask = None
        else:
            _, mask = loader.normalize_input(
                template=self.params._norm_template_param(allow_none=True),
                mask=self.params._get_mask(params=mask_params),
            )

        fsc, avg = loader.reshape(mask=mask, shape=shape).fsc_with_average(
            mask=mask, seed=seed, n_set=n_pairs, dfreq=dfreq
        )

        if show_average:
            img_avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale)
        else:
            img_avg = None

        result = widget_utils.FscResult.from_dataframe(fsc, loader.scale)
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
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: Annotated[Optional[nm], {"text": "Use mask shape", "options": {"value": 12.0, "max": 100.0}, "label": "size (nm)"}] = None,
        cutoff: _CutoffFreq = 0.5,
        interpolation: OneOf[INTERPOLATION_CHOICES] = 3,
        bin_size: _BINSIZE = 1,
        n_components: Annotated[int, {"min": 2, "max": 20}] = 2,
        n_clusters: Annotated[int, {"min": 2, "max": 100}] = 2,
        seed: Annotated[Optional[int], {"text": "Do not use random seed."}] = 0,
    ):  # fmt: skip
        """
        Classify molecules in the loader using PCA and K-means clustering.

        Parameters
        ----------
        {loader_name}{mask_params}{size}{cutoff}{interpolation}{bin_size}
        n_components : int, default is 2
            The number of PCA dimensions.
        n_clusters : int, default is 2
            The number of clusters.
        seed : int, default is 0
            Random seed.
        """
        from cylindra.widgets.subwidgets import PcaViewer

        t0 = timer()
        loaderlist = self._get_parent()._loaders
        loader = loaderlist[loader_name].loader
        shape = self._get_shape_in_px(size, loader)

        _, mask = loader.normalize_input(
            template=self.params._norm_template_param(allow_none=True),
            mask=self.params._get_mask(params=mask_params),
        )
        out, pca = (
            loader.reshape(mask=mask, shape=shape)
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

        loader.molecules.features = out.molecules.features
        t0.toc()

        @thread_worker.callback
        def _on_return():
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
        loader = self._get_parent()._loaders[self.loader_name].loader
        template, _ = loader.normalize_input(self.params._norm_template_param())
        if template is None:
            raise ValueError("No template to show.")
        template = ip.asarray(template, axes="zyx").set_scale(
            zyx=loader.scale, unit="nm"
        )
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
        loader = self._get_parent()._loaders[self.loader_name].loader
        _, mask = loader.normalize_input(
            self.params._norm_template_param(allow_none=True), self.params._get_mask()
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
            tmp = loader.normalize_template(self.params._norm_template_param())
            return tmp.shape
        else:
            return (roundint(default / loader.scale),) * 3

    @setup_function_gui(split_loader)
    def _(self, gui):
        gui[0].changed.connect(gui[1].reset_choices)


def _coerce_aligned_name(name: str, loaders: LoaderList):
    num = 1
    if re.match(rf".*-{ALN_SUFFIX}(\d)+", name):
        try:
            *pre, suf = name.split(f"-{ALN_SUFFIX}")
            num = int(suf) + 1
            name = "".join(pre)
        except Exception:
            num = 1

    existing_names = {info.name for info in loaders}
    while name + f"-{ALN_SUFFIX}{num}" in existing_names:
        num += 1
    return name + f"-{ALN_SUFFIX}{num}"
