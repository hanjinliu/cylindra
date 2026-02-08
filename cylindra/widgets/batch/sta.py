import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Iterator

import impy as ip
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from acryo import BatchLoader, SubtomogramLoader
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
        "options": {"max": 20.0, "step": 0.1},
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
        loader = self._get_parent().loader_infos[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        img = ip.asarray(
            loader.replace(output_shape=shape, order=interpolation)
            .binning(bin_size, compute=False)
            .order_optimize()
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
        loader = self._get_parent().loader_infos[loader_name].loader
        shape = self._get_shape_in_px(size, loader)
        img = ip.asarray(
            loader.replace(output_shape=shape, order=interpolation)
            .binning(bin_size, compute=False)
            .order_optimize()
            .groupby(norm_polars_expr(by))
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
        _Logger.print(f"Aligning {loader.molecules.count()} molecules ...")
        aligned = (
            loader.replace(output_shape=template.shape, order=interpolation)
            .binning(bin_size, compute=False)
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
        min_rotation_step: Annotated[float, {"min": 0.1, "step": 0.1}] = 0.4,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        bin_size: _BINSIZE = 1,
        max_num_iters: Annotated[int, {"min": 3, "max": 100}] = 20,
    ):  # fmt: skip
        """Create initial model by iteratively aligning molecules without template.

        This method iteratively calculate average, evaluate using FSC, and align
        molecules to the current average.

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
        shape = self._get_shape_in_px(size, info.loader)
        loader = (
            info.loader.replace(output_shape=shape, order=interpolation)
            .binning(bin_size, compute=False)
            .order_optimize()
        )
        _Logger.print(f"Aligning {loader.molecules.count()} molecules ...")
        _alignment_state = template_free.AlignmentState(
            rng=rng,
            mask=mask,
            alignment_model=_get_alignment(method),
            min_rotation_step=min_rotation_step,
        )
        yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
        result0 = _alignment_state.fsc_step_init(loader, max_shifts, max_rotations)
        yield _plot_current_fsc.with_args(
            result0.fsc, _alignment_state.num_iter, result0.avg
        ).with_desc(_pdesc.align_tf_1(_alignment_state))
        while True:
            _Logger.print(_alignment_state.next_params(loader.scale).format())
            _exceeded = max_num_iters <= _alignment_state.num_iter
            if _alignment_state.is_converged() or _exceeded:
                _Logger.print("Maximum iteration exceeded" if _exceeded else "FSC converged.")  # fmt: skip
                break
            loader = _alignment_state.align_step(loader)
            yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
            result = _alignment_state.fsc_step(loader)
            yield _plot_current_fsc.with_args(
                result.fsc, _alignment_state.num_iter, result.avg
            ).with_desc(_pdesc.align_tf_1(_alignment_state))

        loaderlist.add_loader(
            loader.order_restore(),
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()
        return self._show_rec.with_args(
            _alignment_state.results[-1].avg, f"[AVG]{loader_name}"
        )

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
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        num_trials: Annotated[int, {"min": 1, "max": 100}] = 5,
        seed: _SeedType = 0,
    ):
        """Run RMA alignment on all molecules in the selected loader.

        Parameters
        ----------
        {loader_name}{template_path}{mask_params}{max_shifts}{rotations}{cutoff}
        {interpolation}{method}{range_long}{range_lat}{angle_max}{bin_size}
        {temperature_time_const}{upsample_factor}{num_trials}{seed}
        """
        t0 = timer()
        batch = self._get_parent()
        loaderlist = batch._loaders
        info = loaderlist[loader_name]
        loader = info.loader
        template, mask = loader.normalize_input(
            template=self.params._norm_template_param(template_path),
            mask=self.params._get_mask(params=mask_params),
        )
        aln = _get_alignment(method)
        sub_inputs = list(self._group_loader_by_spline(loader_name))
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
                loader=inputs.loader.replace(
                    order=interpolation, output_shape=template.shape
                ).binning(bin_size, compute=False),
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
                random_seeds=create_random_seeds(num_trials, seed),
            )
            yield _plot_annealing_result.with_args(results)
            mole = mole.with_features(
                pl.lit(inputs.image_id).alias(Mole.image),
                pl.lit(inputs.molecule_id).alias(Mole.id),
            )
            inputs.loader = inputs.loader.replace(molecules=mole)

        loader_batch = BatchLoader(
            order=loader.order,
            scale=loader.scale,
            output_shape=loader.output_shape,
        )
        for inputs in sub_inputs:
            loader_batch.add_tomogram(
                loader.images[inputs.image_id],
                inputs.loader.molecules,
                inputs.image_id,
                loader._tilt_models[inputs.image_id],
            )
        loaderlist.add_loader(
            loader_batch,
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
        min_rotation_step: Annotated[float, {"min": 0.1, "step": 0.1}] = 0.4,
        interpolation: Annotated[int, {"choices": INTERPOLATION_CHOICES}] = 3,
        method: Annotated[str, {"choices": METHOD_CHOICES}] = "zncc",
        range_long: _DistRangeLon = (4.0, 4.28),
        range_lat: _DistRangeLat = ("d.mean() - 0.1", "d.mean() + 0.1"),
        angle_max: _AngleMaxLon = 5.0,
        bin_size: _BINSIZE = 1,
        temperature_time_const: Annotated[float, {"min": 0.01, "max": 10.0}] = 1.0,
        upsample_factor: Annotated[int, {"min": 1, "max": 20}] = 5,
        max_num_iters: Annotated[int, {"min": 3, "max": 100}] = 20,
        seed: _SeedType = 0,
    ):
        """Create initial model by iteratively aligning molecules by RMA without template.

        This method iteratively calculate average, evaluate using FSC, and align
        molecules using RMA to the current average.

        Parameters
        ----------
        {loader_name}{mask_params}{size}{max_shifts}{max_rotations}{min_rotation_step}
        {interpolation}{method}{range_long}{range_lat}{angle_max}{bin_size}
        {temperature_time_const}{upsample_factor}{max_num_iters}{seed}
        """
        t0 = timer()
        rng = np.random.default_rng(seed)
        batch = self._get_parent()
        loaderlist = batch._loaders
        info = loaderlist[loader_name]
        loader = info.loader
        mask = self.params._get_mask(params=mask_params)
        shape = self._get_shape_in_px(size, loader)
        loader = loader.replace(output_shape=shape, order=interpolation).binning(
            bin_size, compute=False
        )

        _alignment_state = template_free.RMAAlignmentState(
            rng=rng,
            mask=mask,
            alignment_model=_get_alignment(method),
            min_rotation_step=min_rotation_step,
        )
        yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))
        result0 = _alignment_state.fsc_step_init(
            loader, max_shifts, max_rotations, upsample_factor, temperature_time_const
        )

        sub_inputs = list(self._group_loader_by_spline(loader_name))
        num_splines = len(sub_inputs)
        while True:
            yield _plot_current_fsc.with_args(
                result0.fsc, _alignment_state.num_iter, result0.avg
            ).with_desc(_pdesc.align_tf_1(_alignment_state))
            _Logger.print(_alignment_state.next_params(loader.scale).format())
            _exceeded = max_num_iters <= _alignment_state.num_iter
            if _alignment_state.is_converged() or _exceeded:
                _Logger.print(
                    "Maximum iteration exceeded" if _exceeded else "FSC converged."
                )
                break
            _Logger.print(f"{num_splines} splines found for RMA alignment.")
            for ith, inputs in enumerate(sub_inputs):
                _p_ind = f"({ith + 1}/{num_splines})"
                yield thread_worker.description(
                    f"Landscape construction {_p_ind} of iteration {_alignment_state.num_iter + 1}"
                )
                landscape = _alignment_state.built_landscape_step(inputs.loader)
                yield thread_worker.description(
                    f"RMA step {_p_ind} of iteration {_alignment_state.num_iter + 1}"
                )
                inputs.loader = _alignment_state.rma_step(
                    landscape,
                    inputs.loader,
                    inputs.spline,
                    range_long,
                    range_lat,
                    angle_max,
                )
                inputs.loader = inputs.loader.replace(
                    molecules=inputs.loader.molecules.with_features(
                        pl.lit(inputs.image_id).alias(Mole.image),
                        pl.lit(inputs.molecule_id).alias(Mole.id),
                    )
                )
            loader_batch = join_loaders(loader, sub_inputs)
            result0 = _alignment_state.fsc_step(loader_batch)
            yield thread_worker.description(_pdesc.align_tf_0(_alignment_state))

        loaderlist.add_loader(
            loader_batch,
            name=_coerce_aligned_name(info.name, loaderlist),
            image_paths=info.image_paths,
            invert=info.invert,
        )
        t0.toc()
        return self._show_rec.with_args(
            _alignment_state.results[-1].avg, f"[AVG]{loader_name}"
        )

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

    def _group_loader_by_spline(self, loader_name: str) -> Iterator["LoaderOnSpline"]:
        batch = self._get_parent()
        info = batch.loader_infos[loader_name]
        loader = info.loader
        img_project_map = {Path(prj.image): prj for prj in batch.iter_projects()}
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
            loader_single = SubtomogramLoader(
                image=loader.images[img_id],
                molecules=each.molecules.with_features(
                    pl.lit(img_id).alias(Mole.image),
                    pl.lit(mole_id).alias(Mole.id),
                ),
                order=loader.order,
                scale=loader.scale,
                output_shape=loader.output_shape,
                tilt_model=loader._tilt_models[img_id],
            )
            yield LoaderOnSpline(
                loader=loader_single, image_id=img_id, molecule_id=mole_id, spline=spl
            )


@setup_function_gui(BatchSubtomogramAveraging.split_loader)
def _setup_split_loader(self: BatchSubtomogramAveraging, gui: FunctionGui):
    gui[0].changed.connect(gui[1].reset_choices)


@setup_function_gui(BatchSubtomogramAveraging.align_all_rma)
@setup_function_gui(BatchSubtomogramAveraging.align_all_rma_template_free)
def _init_seed(self: BatchSubtomogramAveraging, gui: "FunctionGuiPlus"):
    @gui.activated.connect
    def _set_random_seed():
        gui.seed.value = np.random.randint(0, 1_000_000)


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
    loader: SubtomogramLoader
    image_id: int
    molecule_id: str
    spline: CylSpline


def join_loaders(loader: BatchLoader, sub_inputs: list[LoaderOnSpline]) -> BatchLoader:
    loader_batch = BatchLoader(
        order=loader.order,
        scale=loader.scale,
        output_shape=loader.output_shape,
    )
    for inputs in sub_inputs:
        loader_batch.add_tomogram(
            loader.images[inputs.image_id],
            inputs.loader.molecules,
            inputs.image_id,
            loader._tilt_models[inputs.image_id],
        )
    return loader_batch
