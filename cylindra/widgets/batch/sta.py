import re
import tempfile
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Iterator

import impy as ip
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from acryo import BackProjectionLoader, BatchLoader, Molecules, SubtomogramLoader
from acryo.loader import ExtractedSubvolumeLoader
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
from magicgui.widgets import Container, FunctionGui

from cylindra import _shared_doc, template_free
from cylindra.components.landscape import Landscape
from cylindra.components.spline import CylSpline
from cylindra.const import ALN_SUFFIX, nm
from cylindra.const import MoleculesHeader as Mole
from cylindra.utils import create_random_seeds, roundint
from cylindra.widget_utils import (
    DistExprStr,
    FscResult,
    PolarsExprStr,
    norm_polars_expr,
    timer,
)
from cylindra.widgets import _annealing
from cylindra.widgets import _progress_desc as _pdesc
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
    _plot_current_fsc,
    _store_halfmaps,
)

if TYPE_CHECKING:
    from magicclass._gui._function_gui import FunctionGuiPlus
    from napari.layers import Image


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
_MaxRotations = Annotated[
    tuple[float, float, float],
    {
        "options": {"max": 180.0, "step": 0.1},
        "label": "max rotations (ZYX deg)",
    },
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

_DistRangeLon = Annotated[
    tuple[DistExprStr, DistExprStr],
    {"label": "longitudinal range (nm)"},
]
_DistRangeLat = Annotated[
    tuple[DistExprStr, DistExprStr],
    {"label": "lateral range (nm)"},
]
_AngleMaxLon = Annotated[
    float, {"max": 90.0, "step": 0.5, "label": "maximum angle (deg)"}
]
_SeedType = Annotated[
    int,
    {
        "min": 0,
        "max": 1_000_000,
        "step": 1,
    },
]
_Logger = getLogger("cylindra")
# key of the layer metadata to store the intermediate results of template-free
# alignment
TEMPLATE_FREE_HISTORY = "template_free_history"


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
        """Split the selected loader by the values of the given column.

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
        """Filter the selected loader and add the filtered one to the list.

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
        new = loader.filter(norm_polars_expr(expression))
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
        """Average all the molecules in the selected loader.

        Parameters
        ----------
        {loader_name}{size}{interpolation}{bin_size}
        """
        t0 = timer()
        loader = (
            self._get_parent()
            .loader_infos[loader_name]
            .loader.replace(order=interpolation)
            .binning(bin_size, compute=False)
        )
        shape = self._get_shape_in_px(size, loader)
        img = ip.asarray(
            loader.replace(output_shape=shape).order_optimize().average(),
            axes="zyx",
        ).set_scale(zyx=loader.scale, unit="nm")
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
        """Groupwise subtomogram averaging using molecules grouped by the given expression.

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
        loader = (
            self._get_parent()
            .loader_infos[loader_name]
            .loader.replace(order=interpolation)
            .binning(bin_size, compute=False)
        )
        shape = self._get_shape_in_px(size, loader)
        img = ip.asarray(
            loader.replace(output_shape=shape)
            .order_optimize()
            .groupby(norm_polars_expr(by))
            .average()
            .value_stack(axis=0),
            axes="pzyx",
        ).set_scale(zyx=loader.scale, unit="nm")
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
        loader = info.loader.replace(order=interpolation).binning(
            bin_size, compute=False
        )
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        _Logger.print(f"Aligning {loader.molecules.count()} molecules ...")
        aligned = (
            loader.replace(output_shape=template.shape)
            .order_optimize()
            .align(
                template=template,
                mask=mask,
                max_shifts=max_shifts,
                rotations=rotations,
                cutoff=cutoff,
                alignment_model=_get_alignment(method),
            )
            .order_restore()
        )
        loaderlist.add_loader(
            aligned,
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()

    @set_design(text="Align all (template-free)", location=BatchRefinement)
    @dask_thread_worker.with_progress(desc="Align all started")
    def align_all_template_free(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: _SubVolumeSize = 12.0,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        max_rotations: _MaxRotations = (3.0, 3.0, 3.0),
        min_rotation_step: Annotated[float, {"min": 0.1, "step": 0.1}] = 1.0,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        bin_size: _BINSIZE = 1,
        max_num_iters: Annotated[int, {"min": 3, "max": 100}] = 20,
    ):  # fmt: skip
        """Create initial model by iteratively aligning molecules without template.

        This method iteratively calculate average, evaluate using FSC, and align
        molecules to the current average. Molecules are randomly split into two half
        sets, and the molecules of each half set are aligned to the average of the same
        half set (gold-standard). As FSC between the half sets is not inflated by the
        noise shared through the same template, the templates are weighted by FSC.

        Parameters
        ----------
        {loader_name}{mask_params}{size}{max_shifts}{max_rotations}{min_rotation_step}
        {interpolation}{method}{bin_size}{max_num_iters}
        """
        t0 = timer()
        rng = np.random.default_rng(1428)
        loaderlist = self._get_parent()._loaders
        info = loaderlist[loader_name]
        mask = self.params._get_mask(params=mask_params)
        loader = info.loader.replace(order=interpolation).binning(
            bin_size, compute=False
        )
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape).order_optimize()
        _Logger.print(f"Aligning {loader.molecules.count()} molecules ...")
        _alignment_state = template_free.AlignmentState(
            rng=rng,
            mask=mask,
            alignment_model=_get_alignment(method),
            min_rotation_step=min_rotation_step,
        )
        # loaders of each tomogram, which have their own tilt models
        sub_loaders = list(loader.loaders.iter_loaders())
        num_images = len(sub_loaders)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Subtomograms are extracted to files only once, as spline coefficients so
            # that they can be resampled without the prefilter. In each iteration, the
            # extracted subtomograms are resampled at the aligned molecules for both
            # alignment and FSC calculation.
            extracted: dict[Hashable, ExtractedSubvolumeLoader] = {}
            for ith, (img_id, sub) in enumerate(sub_loaders):
                yield thread_worker.description(
                    f"Extracting subtomograms ({ith + 1}/{num_images})"
                )
                extracted[img_id] = sub.extract_subtomograms(
                    Path(tmpdir) / f"image-{ith}",
                    chunksize=64,
                    apply_prefilter=True,
                )
            # the image IDs are kept, as the aligned molecules are the output
            joined = template_free.assign_halves(join_extracted_loaders(extracted), rng)
            yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
            result = _alignment_state.fsc_step_init(joined, max_shifts, max_rotations)
            _Logger.print(f"Iteration {int(_alignment_state.num_iter)}")
            yield _plot_current_fsc.with_args(result.fsc, result.avg).with_desc(_pdesc.align_tf_1(_alignment_state))  # fmt: skip
            while True:
                _exceeded = max_num_iters <= _alignment_state.num_iter
                if _alignment_state.is_converged() or _exceeded:
                    _Logger.print("Maximum iteration exceeded" if _exceeded else "FSC converged.")  # fmt: skip
                    break
                _Logger.print(_alignment_state.results[-1].params.format())
                # the extracted subtomograms are resampled at the aligned molecules
                joined = _alignment_state.align_step(joined)
                yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
                result = _alignment_state.fsc_step(joined)
                _Logger.print(f"Iteration {int(_alignment_state.num_iter)}")
                yield _plot_current_fsc.with_args(result.fsc, result.avg).with_desc(_pdesc.align_tf_1(_alignment_state))  # fmt: skip

        mole_aligned = template_free.as_alignment_output(
            loader.molecules, [joined.molecules]
        )
        loaderlist.add_loader(
            loader.replace(molecules=mole_aligned).order_restore(),
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()
        return self._show_template_free_average.with_args(result, f"[AVG]{loader_name}")

    @set_design(text="RMA alignment", location=BatchRefinement)
    @dask_thread_worker.with_progress(desc="Simulated annealing (RMA)")
    def align_all_rma(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        template_path: Annotated[str | Path, {"bind": _get_template_path}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        rotations: _Rotations = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        cutoff: _CutoffFreq = 0.5,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        range_long: _DistRangeLon = (4.0, 4.28),
        range_lat: _DistRangeLat = ("d.mean() - 0.1", "d.mean() + 0.1"),
        angle_max: _AngleMaxLon = 5.0,
        bin_size: _BINSIZE = 1,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        lj_const: Annotated[float, {"min": 0.00, "max": 1000.0, "step": 0.1, "label": "LJ const"}] = 0.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        num_trials: Annotated[int, {"min": 1, "max": 100}] = 5,
        seed: _SeedType = 0,
    ):  # fmt: skip
        """Run RMA alignment on all molecules in the selected loader.

        Parameters
        ----------
        {loader_name}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{method}{range_long}{range_lat}{angle_max}{bin_size}
        {temperature_time_const}{lj_const}{upsample_factor}{num_trials}{seed}
        """
        t0 = timer()
        batch = self._get_parent()
        loaderlist = batch._loaders
        info = loaderlist[loader_name]
        loader = info.loader.replace(order=interpolation).binning(
            bin_size, compute=False
        )
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        loader = loader.replace(output_shape=template.shape)
        aln = _get_alignment(method)
        sub_inputs = list(self._group_loader_by_spline(loader_name, loader))
        num_splines = len(sub_inputs)
        _Logger.print(f"{num_splines} splines found for RMA alignment.")

        @thread_worker.callback
        def _plot_annealing_result(results):
            with _Logger.set_plt():
                _annealing.plot_annealing_result(results)

        for ith, inputs in enumerate(sub_inputs):
            _p_ind = f"({ith + 1}/{num_splines})"
            yield thread_worker.description(f"Landscape construction {_p_ind}")
            landscape = Landscape.from_loader(
                loader=inputs.loader,
                template=template,
                mask=mask,
                max_shifts=max_shifts,
                upsample_factor=upsample_factor,
                alignment_model=aln.with_params(
                    rotations=rotations,
                    cutoff=cutoff,
                    tilt=inputs.loader.tilt_model,
                ),
            ).normed()
            yield thread_worker.description(f"RMA alignment {_p_ind}")
            mole, results = landscape.run_annealing_along_spline(
                inputs.spline,
                range_long,
                range_lat,
                angle_max=angle_max,
                temperature_time_const=temperature_time_const,
                lj_nstd=None if lj_const < 1e-3 else 1 / lj_const,
                random_seeds=create_random_seeds(num_trials, seed),
            )
            yield _plot_annealing_result.with_args(results)
            mole = mole.with_features(
                pl.lit(inputs.image_id).alias(Mole.image),
                pl.lit(inputs.molecule_id).alias(Mole.id),
            )
            inputs.loader = inputs.loader.replace(molecules=mole)

        loaderlist.add_loader(
            join_loaders(loader, sub_inputs),
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()

    @set_design(text="RMA alignment (template-free)", location=BatchRefinement)
    @dask_thread_worker.with_progress(desc="RMA alignment started")
    def align_all_rma_template_free(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: _SubVolumeSize = 12.0,
        max_shifts: _MaxShifts = (0.8, 0.8, 0.8),
        max_rotations: _MaxRotations = (0.0, 0.0, 0.0),
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        range_long: _DistRangeLon = (4.0, 4.28),
        range_lat: _DistRangeLat = ("d.mean() - 0.1", "d.mean() + 0.1"),
        angle_max: _AngleMaxLon = 5.0,
        bin_size: _BINSIZE = 1,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 3,
        num_trials: Annotated[int, {"min": 1, "max": 100}] = 5,
        max_num_iters: Annotated[int, {"min": 3, "max": 100}] = 8,
        seed: _SeedType = 0,
    ):
        """Create initial model by iteratively aligning molecules by RMA without template.

        The annealing schedule of RMA is split into `max_num_iters` iterations. In
        each iteration, annealing is resumed from the last state and temperature of
        the previous iteration using the current average as the template, and then the
        average is updated and evaluated using FSC. As the molecules are perturbed at
        high temperature, the template is updated only if the resolution at FSC=0.143
        is improved. If FSC converged, the rest of the annealing will be finished in
        the next iteration. Splines are split into two half sets, and the molecules of
        each half set are aligned to the average of the same half set (gold-standard).
        As FSC between the half sets is not inflated by the noise shared through the
        same template, the templates are weighted by FSC. At least two splines are
        required.

        Parameters
        ----------
        {loader_name}{mask_params}{size}{max_shifts}{max_rotations}
        {interpolation}{method}{range_long}{range_lat}{angle_max}{bin_size}
        {temperature_time_const}{upsample_factor}{num_trials}{max_num_iters}{seed}
        """
        t0 = timer()
        rng = np.random.default_rng(seed)
        batch = self._get_parent()
        loaderlist = batch._loaders
        info = loaderlist[loader_name]
        mask = self.params._get_mask(params=mask_params)
        loader = info.loader.replace(order=interpolation).binning(
            bin_size, compute=False
        )
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape)

        _alignment_state = template_free.RMAAlignmentState(
            rng=rng,
            mask=mask,
            alignment_model=_get_alignment(method),
            keep_best_template=True,
            max_num_iters=max_num_iters,
        )
        sub_inputs = list(self._group_loader_by_spline(loader_name, loader))
        num_splines = len(sub_inputs)
        _Logger.print(f"{num_splines} splines found for RMA alignment.")
        if num_splines < 2:
            raise ValueError(
                "At least two splines are required for gold-standard alignment, got "
                f"{num_splines}."
            )
        # the half set (0 or 1) of each spline for gold-standard alignment
        counts = [inputs.loader.molecules.count() for inputs in sub_inputs]
        halves = template_free.split_into_halves(counts, rng)
        _n0, _n1 = (
            sum(c for c, h in zip(counts, halves, strict=True) if h == half)
            for half in (0, 1)
        )
        _Logger.print(
            f"Splines split into half sets of {halves.count(0)} splines "
            f"({_n0} molecules) and {halves.count(1)} splines ({_n1} molecules)."
        )
        annealings: list[template_free.ResumableAnnealing | None] = [None] * num_splines
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Subtomograms are extracted to files only once, as spline coefficients so
            # that they can be resampled without the prefilter. Landscapes are always
            # constructed around the initial molecules so that the annealing can be
            # resumed in the next iteration, and the averages for FSC are resampled
            # from the extracted subtomograms at the aligned molecules.
            for ith, inputs in enumerate(sub_inputs):
                yield thread_worker.description(
                    f"Extracting subtomograms ({ith + 1}/{num_splines})"
                )
                inputs.loader = inputs.loader.extract_subtomograms(
                    Path(tmpdir) / f"spline-{ith}", chunksize=64, apply_prefilter=True
                )
            extracted = [inputs.loader for inputs in sub_inputs]
            yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
            result = _alignment_state.fsc_step_init(
                _join_extracted_halves(extracted, halves), max_shifts, max_rotations,
                upsample_factor, temperature_time_const, num_trials,
            )  # fmt: skip
            _final = False
            while True:
                _Logger.print(f"Iteration {int(_alignment_state.num_iter)}")
                yield _plot_current_fsc.with_args(result.fsc, result.avg).with_desc(_pdesc.align_tf_1(_alignment_state))  # fmt: skip
                if _final:
                    break
                if (_tmpl := _alignment_state.template_result) is not result:
                    _Logger.print(
                        "FSC not improved. The template of iteration "
                        f"{_tmpl.params.num_iter} is used."
                    )
                if _final := _alignment_state.is_final_iteration():
                    _converged = _alignment_state.is_converged()
                    _Logger.print("FSC converged." if _converged else "Last iteration.")
                _Logger.print(_alignment_state.results[-1].params.format())
                _ratio = _alignment_state.temperature_ratio(_final)
                _Logger.print(f"Annealing until temperature reaches {_ratio:.2e} of T0")
                _iter = _alignment_state.num_iter + 1
                molecules: list[Molecules] = []
                for ith, inputs in enumerate(sub_inputs):
                    _p_ind = f"({ith + 1}/{num_splines})"
                    yield thread_worker.description(
                        f"Landscape construction {_p_ind} of iteration {_iter}"
                    )
                    landscape = _alignment_state.built_landscape_step(
                        inputs.loader, half=halves[ith]
                    )
                    annealing = annealings[ith]
                    if annealing is None:
                        annealing = _alignment_state.prep_annealing_step(
                            landscape, inputs.spline, range_long, range_lat, angle_max
                        )
                        annealings[ith] = annealing
                    else:
                        annealing.update_landscape(landscape)
                    yield thread_worker.description(
                        f"RMA step {_p_ind} of iteration {_iter}"
                    )
                    mole = _alignment_state.rma_step(annealing, final=_final)
                    molecules.append(
                        mole.with_features(
                            pl.lit(inputs.image_id).alias(Mole.image),
                            pl.lit(inputs.molecule_id).alias(Mole.id),
                        )
                    )
                yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
                result = _alignment_state.fsc_step(
                    _join_extracted_halves(extracted, halves, molecules)
                )

        loaderlist.add_loader(
            join_loaders(loader, sub_inputs, molecules),
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()
        return self._show_template_free_average.with_args(result, f"[AVG]{loader_name}")

    @set_design(text="RFA alignment (template-free)", location=BatchRefinement)
    @dask_thread_worker.with_progress(desc="RFA alignment started")
    def align_all_rfa_template_free(
        self,
        loader_name: Annotated[str, {"bind": _get_current_loader_name}],
        mask_params: Annotated[Any, {"bind": _get_mask_params}],
        size: _SubVolumeSize = 12.0,
        max_shifts: _MaxShifts = (2, 2, 2),
        max_rotations: _MaxRotations = (0.0, 180.0, 0.0),
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        range_long: _DistRangeLon = (4.0, 10.0),
        angle_max: _AngleMaxLon = 5.0,
        optimize_polarity: bool = True,
        randomize_roll_angle: bool = False,
        bin_size: _BINSIZE = 1,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 50.0}] = 5.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 3,
        num_trials: Annotated[int, {"min": 1, "max": 100}] = 5,
        max_num_iters: Annotated[int, {"min": 3, "max": 100}] = 8,
        seed: _SeedType = 0,
    ):
        """Create initial model of filaments by iterative RFA without template.

        This method is useful to create an initial model from molecules mapped along
        manually initialized splines (such as by `map_along_spline`), where the
        lattice parameters (and the polarity of the filaments) are unknown.

        Parameters
        ----------
        {loader_name}{mask_params}{size}{max_shifts}{max_rotations}
        {interpolation}{method}{range_long}{angle_max}
        optimize_polarity : bool, default True
            If true, the polarity of each filament is optimized by running annealing
            in both orientations. Otherwise, the orientation of the input molecules
            is kept.
        randomize_roll_angle : bool, default False
            If true, molecules are rotated around their y-axis (the filament axis) by
            random angles before alignment, so that the missing wedges of the
            subtomograms are distributed around the axis in the average. Rotation
            around the y-axis should be searched (such as `max_rotations=(0, 180, 0)`)
            to recover the roll angles.
        {bin_size}{temperature_time_const}{upsample_factor}{num_trials}
        {max_num_iters}{seed}
        """
        t0 = timer()
        rng = np.random.default_rng(seed)
        batch = self._get_parent()
        loaderlist = batch._loaders
        info = loaderlist[loader_name]
        mask = self.params._get_mask(params=mask_params)
        loader = info.loader.replace(order=interpolation).binning(
            bin_size, compute=False
        )
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape)

        _alignment_state = template_free.RFAAlignmentState(
            rng=rng,
            mask=mask,
            alignment_model=_get_alignment(method),
            max_num_iters=max_num_iters,
        )
        sub_inputs = list(self._group_loader_by_spline(loader_name, loader))
        num_splines = len(sub_inputs)
        _Logger.print(f"{num_splines} splines found for RFA alignment.")
        # annealings in the forward (and reverse) orientations
        annealings: list[tuple[template_free.ResumableAnnealing, ...] | None] = [
            None
        ] * num_splines
        # molecules around which the first landscapes are built, in the forward (and
        # reverse) orientations
        initial_refs: list[tuple[Molecules, ...]] = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Same as RMA, subtomograms are extracted only once. Landscapes are always
            # built at the initial positions of molecules, but in the orientations of
            # the best results of the last iteration, so that the annealing is resumed
            # with rotations searched around them. Subvolumes in any orientation,
            # including the reverse one, are resampled from the extracted subtomograms.
            for ith, inputs in enumerate(sub_inputs):
                yield thread_worker.description(
                    f"Extracting subtomograms ({ith + 1}/{num_splines})"
                )
                inputs.loader = inputs.loader.extract_subtomograms(
                    Path(tmpdir) / f"spline-{ith}", chunksize=64, apply_prefilter=True
                )
                mole = inputs.loader.molecules
                if randomize_roll_angle:
                    mole = template_free.randomize_roll(mole, rng)
                if optimize_polarity:
                    mole_rev = mole.rotate_by_rotvec_internal([np.pi, 0, 0])
                    initial_refs.append((mole, mole_rev))
                else:
                    initial_refs.append((mole,))
            extracted = [inputs.loader for inputs in sub_inputs]
            yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
            result = _alignment_state.fsc_step_init(
                join_extracted_loaders(extracted, [refs[0] for refs in initial_refs]),
                max_shifts, max_rotations, upsample_factor, temperature_time_const,
                num_trials,
            )  # fmt: skip
            # intermediate results for debugging
            history = [
                _history_entry(result, [refs[0] for refs in initial_refs], sub_inputs)
            ]
            _final = False
            while True:
                _Logger.print(f"Iteration {int(_alignment_state.num_iter)}")
                yield _plot_current_fsc.with_args(result.fsc, result.avg).with_desc(_pdesc.align_tf_1(_alignment_state))  # fmt: skip
                if _final:
                    break
                if _final := _alignment_state.is_final_iteration():
                    _converged = _alignment_state.is_converged()
                    _Logger.print("FSC converged." if _converged else "Last iteration.")
                _Logger.print(_alignment_state.results[-1].params.format())
                _ratio = _alignment_state.temperature_ratio(_final)
                _Logger.print(f"Annealing until temperature reaches {_ratio:.2e} of T0")
                _iter = _alignment_state.num_iter + 1
                molecules: list[Molecules] = []
                step_results: list[template_free.RFAStepResult] = []
                # molecules around which the landscapes are built
                references: list[tuple[Molecules, ...]] = []
                for ith, inputs in enumerate(sub_inputs):
                    _p_ind = f"({ith + 1}/{num_splines})"
                    yield thread_worker.description(
                        f"Landscape construction {_p_ind} of iteration {_iter}"
                    )
                    if (pair := annealings[ith]) is None:
                        refs = initial_refs[ith]
                    else:
                        refs = tuple(annealing.next_reference() for annealing in pair)
                    references.append(refs)
                    # landscapes are not normalized to compare the two orientations
                    landscapes = _alignment_state.built_landscapes_step(
                        [
                            template_free.resample_extracted(extracted[ith], ref)
                            for ref in refs
                        ],
                        norm=not optimize_polarity,
                    )
                    if pair is None:
                        pair = tuple(
                            _alignment_state.prep_annealing_step(
                                landscape, inputs.spline, range_long, angle_max
                            )
                            for landscape in landscapes
                        )
                        annealings[ith] = pair
                    else:
                        for annealing, landscape in zip(pair, landscapes, strict=True):
                            annealing.update_landscape(landscape)
                    yield thread_worker.description(
                        f"RFA step {_p_ind} of iteration {_iter}"
                    )
                    step = _alignment_state.rfa_step(*pair, final=_final)
                    step_results.append(step)
                    molecules.append(
                        step.molecules.with_features(
                            pl.lit(inputs.image_id).alias(Mole.image),
                            pl.lit(inputs.molecule_id).alias(Mole.id),
                        )
                    )
                if optimize_polarity:
                    _num_rev = sum(step.is_reversed for step in step_results)
                    _Logger.print(
                        f"Selected orientations: {num_splines - _num_rev} forward, "
                        f"{_num_rev} reverse"
                    )
                yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
                result = _alignment_state.fsc_step(
                    join_extracted_loaders(extracted, molecules)
                )
                history.append(
                    _history_entry(result, molecules, sub_inputs, references)
                )

        loaderlist.add_loader(
            join_loaders(loader, sub_inputs, molecules),
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()

        @thread_worker.callback
        def _on_return():
            layer = self._show_rec(
                _alignment_state.results[-1].avg, f"[AVG]{loader_name}"
            )
            layer.metadata[TEMPLATE_FREE_HISTORY] = history

        return _on_return

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
        """Calculate Fourier Shell Correlation using the selected loader.

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

        fsc, (img_0, img_1), img_mask = (
            loader.reshape(
                template=template if size is None else None,
                mask=mask,
                shape=None if size is None else self._get_shape_in_px(size, loader),
            )
            .order_optimize()
            .fsc_with_halfmaps(
                mask, seed=seed, n_set=n_pairs, dfreq=dfreq, squeeze=False
            )
        )

        def _as_imgarray(im, axes: str = "zyx") -> ip.ImgArray | None:
            if np.isscalar(im):
                return None
            return ip.asarray(im, axes=axes).set_scale(zyx=loader.scale, unit="nm")

        if show_average:
            avg = (img_0[0] + img_1[0]) / 2
            img_avg = _as_imgarray(avg)
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
                plt.tight_layout()
                plt.show()
            for _c in criteria:
                _r = result.get_resolution(_c)
                _Logger.print_html(f"Resolution at FSC={_c:.3f} ... <b>{_r:.3f} nm</b>")

            if img_avg is not None:
                _imlayer: Image = self._show_rec(img_avg, name=f"[AVG]{loader_name}")
                _imlayer.metadata["fsc"] = result
                _imlayer.metadata["fsc_halfmaps"] = (
                    _as_imgarray(img_0, axes="izyx"),
                    _as_imgarray(img_1, axes="izyx"),
                )
                _imlayer.metadata["fsc_mask"] = _as_imgarray(img_mask)

        return _calculate_fsc_on_return

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
        self._show_rec(mask, name="Mask image", store=False, threshold=0.5)

    @thread_worker.callback
    def _show_rec(
        self, img: ip.ImgArray, name: str, store: bool = True, threshold=None
    ):
        return self.params._show_reconstruction(img, name, store, threshold)

    @thread_worker.callback
    def _show_template_free_average(
        self, result: template_free.AlignmentResult, name: str
    ):
        """Show the average of template-free alignment with the half maps if any."""
        layer = self._show_rec(result.avg, name)
        _store_halfmaps(layer, result)
        return layer

    def _get_shape_in_px(
        self, default: "nm | None", loader: BatchLoader
    ) -> tuple[int, ...]:
        if default is None:
            tmp = self._get_template_image(loader.scale)
            return tmp.sizesof("zyx")
        else:
            return (roundint(default / loader.scale),) * 3

    def _get_template_image(self, scale: nm | None = None) -> ip.ImgArray:
        if scale is None:
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

    def _group_loader_by_spline(
        self, loader_name: str, loader: BatchLoader
    ) -> Iterator["LoaderOnSpline"]:
        """Split `loader` (derived from the loader `loader_name`) by source splines.

        `loader` may be binned. Molecules of the yielded loaders are in the same
        coordinates as `loader`, so they can be joined back by `join_loaders`.
        """
        batch = self._get_parent()
        info = batch.loader_infos[loader_name]
        # TODO: batch.iter_projects() does not properly work if no batch project is
        # loaded beforehand.
        img_project_map = {
            path: prj
            for prj in batch.iter_projects()
            if (path := prj._batch_image_path()) is not None
        }
        for (mole_id, img_id), each in loader.group_by(["molecules-id", "image-id"]):
            img_path = info.image_paths[img_id]
            prj = img_project_map[img_path]
            for mole_info in prj.molecules_info:
                if mole_info.name.split(".")[0] == mole_id:
                    if (spl_id := mole_info.source) is None:
                        raise ValueError(
                            f"Molecule {mole_id} does not have source spline."
                        )
                    spl = prj.load_spline(spl_id)
                    break
            else:
                raise ValueError(f"Spline for molecule {mole_id} not found.")
            # the loader of the tomogram, or of the tilt series, of the image
            loader_single = loader.loaders[img_id].replace(
                molecules=each.molecules.with_features(
                    pl.lit(img_id).alias(Mole.image),
                    pl.lit(mole_id).alias(Mole.id),
                ),
            )
            yield LoaderOnSpline(
                loader=loader_single, image_id=img_id, molecule_id=mole_id, spline=spl
            )


@setup_function_gui(BatchSubtomogramAveraging.split_loader)
def _setup_split_loader(self: BatchSubtomogramAveraging, gui: FunctionGui):
    gui[0].changed.connect(gui[1].reset_choices)


@setup_function_gui(BatchSubtomogramAveraging.align_all_rma)
@setup_function_gui(BatchSubtomogramAveraging.align_all_rma_template_free)
@setup_function_gui(BatchSubtomogramAveraging.align_all_rfa_template_free)
def _init_seed(self: BatchSubtomogramAveraging, gui: "FunctionGuiPlus"):
    @gui.activated.connect
    def _set_random_seed():
        gui.seed.value = np.random.randint(0, 1_000_000)


def _history_entry(
    result: template_free.AlignmentResult,
    molecules: list[Molecules],
    sub_inputs: list["LoaderOnSpline"],
    references: list[tuple[Molecules, ...]] | None = None,
) -> dict[str, Any]:
    """Intermediate result of an iteration of template-free alignment.

    `molecules` are those used for the average, and `references` are the molecules
    around which the landscapes are built in the forward (and reverse) orientations.
    """

    def _concat(moles: list[Molecules]) -> Molecules:
        return Molecules.concat(
            [
                mole.with_features(
                    pl.lit(inputs.image_id).alias(Mole.image),
                    pl.lit(inputs.molecule_id).alias(Mole.id),
                )
                for mole, inputs in zip(moles, sub_inputs, strict=True)
            ]
        )

    entry = {
        "average": result.avg,
        "fsc": result.fsc,
        "params": result.params,
        "molecules": _concat(molecules),
    }
    if references is not None:
        entry["references"] = [
            _concat([refs[i] for refs in references]) for i in range(len(references[0]))
        ]
    return entry


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


@dataclass
class LoaderOnSpline:
    loader: SubtomogramLoader | BackProjectionLoader | ExtractedSubvolumeLoader
    image_id: int
    molecule_id: str
    spline: CylSpline


def join_loaders(
    loader: BatchLoader,
    sub_inputs: list[LoaderOnSpline],
    molecules: list[Molecules] | None = None,
) -> BatchLoader:
    """Join the loaders on splines into a batch loader.

    If `molecules` is given, they are used instead of the molecules of `sub_inputs`.
    """
    if molecules is None:
        molecules = [inputs.loader.molecules for inputs in sub_inputs]
    loader_batch = BatchLoader(
        order=loader.order,
        scale=loader.scale,
        output_shape=loader.output_shape,
    )
    # loaders of the tomograms, or of the tilt series, of the images
    image_loaders = dict(loader.loaders.iter_loaders())
    for inputs, mole in zip(sub_inputs, molecules, strict=True):
        loader_batch.add_loader(
            image_loaders[inputs.image_id].replace(molecules=mole), inputs.image_id
        )
    return loader_batch


def join_extracted_loaders(
    loaders: (
        Sequence[ExtractedSubvolumeLoader] | Mapping[Hashable, ExtractedSubvolumeLoader]
    ),
    molecules: list[Molecules] | None = None,
) -> BatchLoader:
    """Join the extracted loaders for averaging, FSC calculation and alignment.

    If `loaders` is a mapping, its keys are used as the image IDs. If `molecules` is
    given, the extracted subvolumes are resampled at these molecules instead of
    cropping the tomograms again.
    """
    if isinstance(loaders, Mapping):
        image_ids, loaders = list(loaders.keys()), list(loaders.values())
    else:
        image_ids, loaders = None, list(loaders)
    if molecules is not None:
        loaders = [
            template_free.resample_extracted(loader, mole)
            for loader, mole in zip(loaders, molecules, strict=True)
        ]
    first = loaders[0]
    inputs = (
        loaders if image_ids is None else dict(zip(image_ids, loaders, strict=True))
    )
    return BatchLoader.from_loaders(
        inputs, order=first.order, scale=first.scale, output_shape=first.output_shape
    )


def _join_extracted_halves(
    loaders: Sequence[ExtractedSubvolumeLoader],
    halves: list[int],
    molecules: list[Molecules] | None = None,
) -> tuple[BatchLoader, BatchLoader]:
    """Join the extracted loaders of each half set.

    `halves` is the half set (0 or 1) of each loader. See `join_extracted_loaders` for
    the other parameters.
    """
    out = list[BatchLoader]()
    for half in (0, 1):
        idx = [i for i, h in enumerate(halves) if h == half]
        out.append(
            join_extracted_loaders(
                [loaders[i] for i in idx],
                None if molecules is None else [molecules[i] for i in idx],
            )
        )
    return tuple(out)
