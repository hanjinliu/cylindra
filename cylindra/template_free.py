from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Sequence

import impy as ip
import numpy as np
from acryo import Molecules, SubtomogramLoader, alignment, pipe
from acryo.loader import ExtractedSubvolumeLoader, LoaderBase
from acryo.loader._extracted import SUBVOLUME_INDEX
from numpy.typing import NDArray

from cylindra._dask import Delayed, compute, delayed
from cylindra.components.landscape import (
    AnnealingResult,
    Landscape,
    _to_epoch_size,
    _update_mole_pos,
)
from cylindra.const import nm
from cylindra.widget_utils import FscResult

if TYPE_CHECKING:
    from acryo.alignment import TomographyInput
    from acryo.tilt import TiltSeriesModel

    from cylindra._cylindra_ext import (
        CylindricAnnealingModel,
        FilamentousAnnealingModel,
    )
    from cylindra.components.spline import CylSpline


def adjust_down(
    a: float, a_min: float, num_iter: int, num_iter_offset: int = 3
) -> float:
    if num_iter < num_iter_offset or a <= a_min:
        return a
    diff = a - a_min
    return diff / (num_iter - num_iter_offset + 2) + a_min


def rotation_range(max_rot: float, max_step: float) -> tuple[float, float]:
    """Return (max, step) of the rotation search around an axis.

    Rotation is searched at -max, 0 and max if `max_rot` is not larger than
    `max_step`. Otherwise, the range is divided by the step not larger than `max_step`,
    so that the resolution of the search is kept for large `max_rot`.
    """
    if max_rot <= max_step:
        return max_rot, max_rot
    return max_rot, max_rot / math.ceil(max_rot / max_step)


def narrow_rotation_range(
    last: tuple[float, float],
    initial: float,
    min_step: float,
    max_step: float,
    num_iter: int,
    num_iter_offset: int = 3,
) -> tuple[float, float]:
    """Return (max, step) of the rotation search around an axis in the next iteration.

    Rotations are searched in the initial range for the first `num_iter_offset`
    iterations. After that, the search is narrowed around the best rotation of the
    last iteration in a coarse-to-fine manner: the next range is the last step, or the
    half of the last range if only three rotations were searched, so that the true
    optimum is always in the range. The range is not narrowed below `min_step`.
    """
    if num_iter < num_iter_offset:
        return rotation_range(initial, max_step)
    last_max, last_step = last
    if last_max <= min_step:
        return last_max, last_step
    return rotation_range(max(min(last_step, last_max / 2), min_step), max_step)


@dataclass
class AlignmentParams:
    num_iter: int
    cutoff: float
    max_shifts: tuple[nm, nm, nm]
    rotations: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]  # max/step
    max_shifts_initial: tuple[nm, nm, nm]
    max_rotations_initial: tuple[float, float, float]

    @classmethod
    def init(
        cls,
        cutoff: float,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
        max_rot_step: float = 30.0,
    ):
        return AlignmentParams(
            num_iter=0,
            cutoff=cutoff,
            max_shifts=max_shifts,
            rotations=tuple(rotation_range(a, max_rot_step) for a in max_rotations),
            max_shifts_initial=max_shifts,
            max_rotations_initial=max_rotations,
        )

    @classmethod
    def from_last_result(
        cls,
        scale: nm,
        result: AlignmentResult,
        min_rot_step: float = 0.5,
        max_rot_step: float = 30.0,
    ):
        num_iter = result.params.num_iter + 1
        cutoff = _safe_div(scale, result.fsc.get_resolution(0.143))
        max_shifts = [
            adjust_down(a, scale, num_iter) for a in result.params.max_shifts_initial
        ]
        last_params = result.params
        rotations = [
            narrow_rotation_range(last, a, min_rot_step, max_rot_step, num_iter)
            for last, a in zip(
                last_params.rotations, last_params.max_rotations_initial, strict=True
            )
        ]
        return AlignmentParams(
            num_iter=num_iter,
            cutoff=cutoff,
            max_shifts=tuple(max_shifts),
            rotations=tuple(rotations),
            max_shifts_initial=result.params.max_shifts_initial,
            max_rotations_initial=result.params.max_rotations_initial,
        )

    def format(self) -> str:
        tz, ty, tx = self.max_shifts
        (z_rot, z_step), (y_rot, y_step), (x_rot, x_step) = self.rotations
        return (
            f"Iteration {self.num_iter}:\n"
            f"Relative cutoff frequency = {self.cutoff:.3f}\n"
            f"Max shifts (Z, Y, X) = {tz:.2f} nm, {ty:.2f} nm, {tx:.2f} nm\n"
            f"Max rotations (Z, Y, X) = {z_rot:.2f}°, {y_rot:.2f}°, {x_rot:.2f}° "
            f"... with steps = {z_step:.2f}°, {y_step:.2f}°, {x_step:.2f}°"
        )


@dataclass
class AlignmentResult:
    """Result of single alignment iteration."""

    fsc: FscResult
    avg: ip.ImgArray
    mask: NDArray[np.float32]
    params: AlignmentParams


@dataclass
class RMAAlignmentParams(AlignmentParams):
    upsample_factor: int
    temperature_time_const: float
    num_trials: int

    @classmethod
    def init(
        cls,
        cutoff: float,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
        upsample_factor: int,
        temperature_time_const: float,
        num_trials: int,
        max_rot_step: float = 30.0,
    ):
        return RMAAlignmentParams(
            num_iter=0,
            cutoff=cutoff,
            max_shifts=max_shifts,
            rotations=tuple(rotation_range(a, max_rot_step) for a in max_rotations),
            max_shifts_initial=max_shifts,
            max_rotations_initial=max_rotations,
            upsample_factor=upsample_factor,
            temperature_time_const=temperature_time_const,
            num_trials=num_trials,
        )

    @classmethod
    def from_last_result(
        cls,
        scale: nm,
        result: RMAAlignmentResult,
        min_rot_step: float = 0.5,
        max_rot_step: float = 30.0,
    ):
        # NOTE: Annealing is resumed in the next iteration, so the parameters that
        # determine the shape of the landscape (max_shifts, upsample_factor) and the
        # annealing models (temperature_time_const, num_trials) must not be changed.
        num_iter = result.params.num_iter + 1
        cutoff = _safe_div(scale, result.fsc.get_resolution(0.143))
        last_params = result.params
        rotations = [
            narrow_rotation_range(last, a, min_rot_step, max_rot_step, num_iter)
            for last, a in zip(
                last_params.rotations, last_params.max_rotations_initial, strict=True
            )
        ]
        return replace(
            result.params,
            num_iter=num_iter,
            cutoff=cutoff,
            rotations=tuple(rotations),
        )

    def format(self) -> str:
        tz, ty, tx = self.max_shifts
        (z_rot, z_step), (y_rot, y_step), (x_rot, x_step) = self.rotations
        out = (
            f"Iteration {self.num_iter}:\n"
            f"Relative cutoff frequency = {self.cutoff:.3f}\n"
            f"Max shifts (Z, Y, X) = {tz:.2f} nm, {ty:.2f} nm, {tx:.2f} nm\n"
            f"Max rotations (Z, Y, X) = {z_rot:.2f}°, {y_rot:.2f}°, {x_rot:.2f}° "
            f"... with steps = {z_step:.2f}°, {y_step:.2f}°, {x_step:.2f}°"
        )
        if self.num_iter <= 1:
            out += (
                f"\nUpsample factor = {self.upsample_factor}\n"
                f"Temperature time constant = {self.temperature_time_const:.2f}\n"
                f"Number of trials = {self.num_trials}"
            )
        return out


@dataclass
class RMAAlignmentResult(AlignmentResult):
    params: RMAAlignmentParams


@dataclass
class BaseAlignmentState[R: "AlignmentResult"]:
    """State of the template-free alignment."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    mask: pipe.ImageProvider | pipe.ImageConverter | None = field(default=None)
    alignment_model: type[alignment.BaseAlignmentModel] = field(
        default=alignment.ZNCCAlignment
    )
    min_rotation_step: float = 1.0
    max_rotation_step: float = 30.0
    results: list[R] = field(default_factory=list)
    # alignment model of the last result
    _model_cache: tuple[R, TomographyInput] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def num_iter(self) -> int:
        """Number of completed iterations."""
        return len(self.results) - 1

    def is_converged(self, tol: nm = 0.005) -> bool:
        if self.num_iter < 3:
            return False
        fsc_3 = self.results[-3].fsc
        fsc_2 = self.results[-2].fsc
        fsc_1 = self.results[-1].fsc
        c0143 = 0.143
        c0500 = 0.5
        res0143_3 = fsc_3.get_resolution(c0143)
        res0143_2 = fsc_2.get_resolution(c0143)
        res0143_1 = fsc_1.get_resolution(c0143)
        res0500_3 = fsc_3.get_resolution(c0500)
        res0500_2 = fsc_2.get_resolution(c0500)
        res0500_1 = fsc_1.get_resolution(c0500)

        diff_1 = (res0143_3 - res0143_1 + res0500_3 - res0500_1) / 2
        diff_2 = (res0143_3 - res0143_2 + res0500_3 - res0500_2) / 2
        return diff_1 < tol and diff_2 < tol

    def _prep_mask(
        self, avg: NDArray[np.float32], scale: nm
    ) -> NDArray[np.float32] | None:
        if self.mask is None:
            return None
        elif isinstance(self.mask, pipe.ImageProvider):
            return self.mask.provide(scale)
        else:
            return self.mask.convert(avg, scale)

    def _alignment_model(self, loader: LoaderBase) -> SharedTemplateModel:
        """Return the alignment model factory of the current iteration.

        The alignment model is created only once for each iteration, so that the
        template images are rotated only once and shared by all the loaders.
        """
        result = self.results[-1]
        if self._model_cache is None or self._model_cache[0] is not result:
            template = result.avg.value
            mask = loader.normalize_mask(result.mask, reshape_to=template.shape)
            model = self.alignment_model.with_params(
                rotations=result.params.rotations,
                cutoff=result.params.cutoff,
            )(template, mask)
            self._model_cache = (result, model)
        return SharedTemplateModel(self._model_cache[1])


@dataclass
class AlignmentState(BaseAlignmentState[AlignmentResult]):
    """State of the template-free alignment.

    Template-free alignment proceeds in iterations consisting of FSC step and alignment
    step.
    """

    def fsc_step_init(
        self,
        loader: LoaderBase,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
    ) -> AlignmentResult:
        """Run the first FSC step"""
        fsc_result, avg = loader.fsc_with_average(
            seed=self.rng.integers(0, 2**32),
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        cutoff = _safe_div(loader.scale, fsc.get_resolution(0.143))
        params = AlignmentParams.init(
            cutoff, max_shifts, max_rotations, self.max_rotation_step
        )
        this_result = AlignmentResult(fsc, avg, None, params)
        self.results.append(this_result)
        return this_result

    def next_params(self, scale: float) -> AlignmentParams:
        last_result = self.results[-1]
        return AlignmentParams.from_last_result(
            scale, last_result, self.min_rotation_step, self.max_rotation_step
        )

    def fsc_step(self, loader: LoaderBase) -> AlignmentResult:
        """Run a FSC step and return the result."""
        last_result = self.results[-1]
        mask = self._prep_mask(last_result.avg.value, loader.scale)
        params = self.next_params(loader.scale)
        return self._fsc_step_impl(loader, mask, params)

    def align_step[L: LoaderBase](self, loader: L) -> L:
        """Run an alignment step and return updated loader."""
        result = self.results[-1]
        model = self._alignment_model(loader)
        return loader.align(
            model.template, mask=model.mask, max_shifts=result.params.max_shifts,
            alignment_model=model,
        )  # fmt: skip

    def _fsc_step_impl(
        self,
        loader: LoaderBase,
        mask: NDArray[np.float32] | None,
        params: AlignmentParams,
    ) -> AlignmentResult:
        fsc_result, avg = loader.fsc_with_average(
            mask,
            seed=self.rng.integers(0, 2**32),
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        this_result = AlignmentResult(fsc, avg, mask, params)
        self.results.append(this_result)
        return this_result


@dataclass
class BaseAnnealingAlignmentState(BaseAlignmentState[RMAAlignmentResult]):
    """State of the template-free alignment using resumable annealing.

    Unlike `AlignmentState`, alignment is not restarted from scratch in every
    iteration. In each iteration, the annealing is resumed from the last state of the
    previous iteration using the energy landscape updated with the new template, and
    the temperature is cooled down by `cooling_fraction`, so that it reaches
    `final_temperature_ratio` of the initial one in `max_num_iters` iterations.
    """

    max_num_iters: int = 8
    final_temperature_ratio: float = 1e-5

    def __post_init__(self):
        if self.max_num_iters < 1:
            raise ValueError(
                f"max_num_iters must be positive, got {self.max_num_iters}."
            )
        if not 0 < self.final_temperature_ratio < 1:
            raise ValueError(
                "final_temperature_ratio must be in (0, 1), got "
                f"{self.final_temperature_ratio}."
            )

    @property
    def cooling_fraction(self) -> float:
        """Ratio of the temperature to be cooled down in each iteration."""
        return self.final_temperature_ratio ** (1 / self.max_num_iters)

    def fsc_step_init(
        self,
        loader: LoaderBase,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
        upsample_factor: int,
        temperature_time_const: float,
        num_trials: int,
    ) -> RMAAlignmentResult:
        """Run the first FSC step"""
        fsc_result, avg = loader.fsc_with_average(
            seed=self.rng.integers(0, 2**32),
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        cutoff = _safe_div(loader.scale, fsc.get_resolution(0.143))
        params = RMAAlignmentParams.init(
            cutoff, max_shifts, max_rotations, upsample_factor,
            temperature_time_const, num_trials, self.max_rotation_step,
        )  # fmt: skip
        this_result = RMAAlignmentResult(fsc, avg, None, params)
        self.results.append(this_result)
        return this_result

    def is_final_iteration(self) -> bool:
        """True if the next RMA step should finish the annealing."""
        return self.is_converged() or self.max_num_iters <= self.num_iter + 1

    def temperature_ratio(self, final: bool = False) -> float:
        """Temperature (relative to the initial one) to be reached in the next step."""
        if final:
            return self.final_temperature_ratio
        frac = self.cooling_fraction ** (self.num_iter + 1)
        return max(self.final_temperature_ratio, frac)

    def next_params(self, scale: float) -> RMAAlignmentParams:
        last_result = self.results[-1]
        return RMAAlignmentParams.from_last_result(
            scale, last_result, self.min_rotation_step, self.max_rotation_step
        )

    def fsc_step(self, loader: LoaderBase) -> RMAAlignmentResult:
        """Run a FSC step and return the result."""
        last_result = self.results[-1]
        mask = self._prep_mask(last_result.avg.value, loader.scale)
        params = self.next_params(loader.scale)
        return self._fsc_step_impl(loader, mask, params)

    def built_landscape_step(
        self,
        loader: SubtomogramLoader,
        norm: bool = True,
    ) -> Landscape:
        """Build the correlation landscape and return updated loader."""
        return self.built_landscapes_step([loader], norm=norm)[0]

    def built_landscapes_step(
        self,
        loaders: Sequence[SubtomogramLoader],
        norm: bool = True,
    ) -> list[Landscape]:
        """Build the correlation landscapes of the loaders in parallel."""
        result = self.results[-1]
        model = self._alignment_model(loaders[0])
        landscapes = Landscape.from_loaders(
            loaders,
            template=model.template,
            mask=model.mask,
            max_shifts=result.params.max_shifts,
            upsample_factor=result.params.upsample_factor,
            alignment_model=model,
        )
        if norm:
            landscapes = [landscape.normed() for landscape in landscapes]
        return landscapes

    def _random_seeds(self) -> list[int]:
        num_trials = self.results[-1].params.num_trials
        return self.rng.integers(0, 2**31 - 1, size=num_trials).tolist()

    def _annealing_step(
        self,
        annealing: ResumableAnnealing,
        final: bool = False,
    ) -> AnnealingResult | None:
        """Resume annealing and return the result of the best trial.

        Annealing runs until the temperature reaches the target of this iteration. If
        `final` is true, annealing will be finished regardless of the iteration. None
        is returned if all the trials failed.
        """
        return self._annealing_steps([annealing], final)[0]

    def _annealing_steps(
        self,
        annealings: Sequence[ResumableAnnealing],
        final: bool = False,
    ) -> list[AnnealingResult | None]:
        """Resume all the trials of the annealings in parallel.

        Same as `_annealing_step` but the best trial of each annealing is returned.
        """
        ratio = self.temperature_ratio(final)
        tasks = [
            annealing.tasks(ratio, cool_completely=final) for annealing in annealings
        ]
        out = list[AnnealingResult | None]()
        for results in compute(*tasks):
            if all(result.state == "failed" for result in results):
                out.append(None)
            else:
                out.append(min(results, key=lambda r: r.energies[-1]))
        return out

    def _fsc_step_impl(
        self,
        loader: LoaderBase,
        mask: NDArray[np.float32] | None,
        params: RMAAlignmentParams,
    ) -> RMAAlignmentResult:
        fsc_result, avg = loader.fsc_with_average(
            mask,
            seed=self.rng.integers(0, 2**32),
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        this_result = RMAAlignmentResult(fsc, avg, mask, params)
        self.results.append(this_result)
        return this_result


@dataclass
class RMAAlignmentState(BaseAnnealingAlignmentState):
    """State of the template-free RMA alignment."""

    def prep_annealing_step(
        self,
        landscape: Landscape,
        spl: CylSpline,
        range_long: tuple[nm | str, nm | str],
        range_lat: tuple[nm | str, nm | str],
        angle_max: float,
    ) -> ResumableAnnealing:
        """Prepare annealing models that will be resumed in the following steps."""
        params = self.results[-1].params
        return ResumableAnnealing.from_landscape(
            landscape,
            spl,
            range_long=range_long,
            range_lat=range_lat,
            angle_max=angle_max,
            temperature_time_const=params.temperature_time_const,
            random_seeds=self._random_seeds(),
        )

    def rma_step(self, annealing: ResumableAnnealing, final: bool = False) -> Molecules:
        """Resume annealing and return the molecules of the best trial.

        Annealing runs until the temperature reaches the target of this iteration. If
        `final` is true, annealing will be finished regardless of the iteration.
        """
        if (best := self._annealing_step(annealing, final)) is None:
            raise RuntimeError(
                "Failed to optimize for all trials. You may check the distance range."
            )
        return annealing.set_best(best)


@dataclass
class RFAStepResult:
    """Result of an RFA step of a filament.

    `energy_reverse` is None if the filament is not annealed in the reverse
    orientation.
    """

    molecules: Molecules
    energy_forward: float
    energy_reverse: float | None = None

    @property
    def is_reversed(self) -> bool:
        """True if the molecules in the reversed orientation are selected."""
        if self.energy_reverse is None:
            return False
        return self.energy_reverse < self.energy_forward


@dataclass
class RFAAlignmentState(BaseAnnealingAlignmentState):
    """State of the template-free RFA alignment.

    If the polarity of each filament is to be optimized, annealing is run for the
    molecules in both orientations, and the better one is used to calculate the next
    average. In each iteration, landscapes should be built around the molecules given
    by `ResumableAnnealing.next_reference`, so that rotations are searched around the
    best orientations of the last iteration.
    """

    def prep_annealing_step(
        self,
        landscape: Landscape,
        spl: CylSpline,
        range_long: tuple[nm | str, nm | str],
        angle_max: float,
    ) -> ResumableAnnealing:
        """Prepare annealing models that will be resumed in the following steps."""
        params = self.results[-1].params
        return ResumableAnnealing.from_landscape_filamentous(
            landscape,
            spl,
            range_long=range_long,
            angle_max=angle_max,
            temperature_time_const=params.temperature_time_const,
            random_seeds=self._random_seeds(),
        )

    def rfa_step(
        self,
        forward: ResumableAnnealing,
        reverse: ResumableAnnealing | None = None,
        final: bool = False,
    ) -> RFAStepResult:
        """Resume annealing and return the molecules of the best trial.

        If `reverse` is given, it must be the annealing of the same molecules as
        `forward` with the opposite orientation, and the molecules of the better
        orientation are returned. In this case, the landscapes must not be normalized,
        so that the energies of the two orientations are comparable.
        """
        if reverse is None:
            best_fw, best_rv = self._annealing_step(forward, final), None
        else:
            best_fw, best_rv = self._annealing_steps([forward, reverse], final)
        if best_fw is None and best_rv is None:
            raise RuntimeError(
                "Failed to optimize for all trials. You may check the distance range."
            )
        # the best trials of both orientations are recorded for the next landscapes
        mole_fw = None if best_fw is None else forward.set_best(best_fw)
        energy_fw = np.inf if best_fw is None else float(best_fw.energies[-1])
        if reverse is None:
            return RFAStepResult(mole_fw, energy_fw)
        mole_rv = None if best_rv is None else reverse.set_best(best_rv)
        energy_rv = np.inf if best_rv is None else float(best_rv.energies[-1])
        if energy_rv < energy_fw:
            return RFAStepResult(mole_rv, energy_fw, energy_rv)
        return RFAStepResult(mole_fw, energy_fw, energy_rv)


class SharedTemplateModel:
    """Alignment model factory that shares the rotated templates of a model.

    Loaders create a new alignment model in every call of `align` or
    `construct_landscape`, and the template images are rotated every time. As the
    rotated templates do not depend on the tilt model, which is only used to mask the
    missing wedge of each subvolume, this factory returns a copy of `model` with the
    tilt model of the loader. The copies share the cache of the rotated templates.
    The template and mask passed by the loader are ignored, as they are the same as
    those of `model`.
    """

    def __init__(self, model: TomographyInput):
        self._model = model

    @property
    def template(self) -> NDArray[np.float32]:
        return self._model.template

    @property
    def mask(self) -> NDArray[np.float32]:
        return self._model.mask

    @property
    def quaternions(self) -> NDArray[np.float32]:
        return self._model.quaternions

    @property
    def has_rotation(self) -> bool:
        return self._model.has_rotation

    def __call__(
        self,
        template: NDArray[np.float32],
        mask: NDArray[np.float32] | None = None,
        tilt: TiltSeriesModel | None = None,
    ) -> TomographyInput:
        model = copy.copy(self._model)
        if tilt is not None:
            model._tilt_model = tilt
        return model


@dataclass
class ResumableAnnealing:
    """Independent annealing trials along a spline that can be resumed.

    The energy landscape can be replaced between runs without resetting the states
    (shifts, temperature and the binding potential) of the annealing models. The new
    landscape may be built around the molecules in different orientations, such as the
    best orientations of the last run (`next_reference`).
    """

    landscape: Landscape
    spline: CylSpline
    models: list[CylindricAnnealingModel] | list[FilamentousAnnealingModel]
    temperature0: float
    epoch_size: int
    # molecules of the best trial of the last run
    best_molecules: Molecules | None = None

    @classmethod
    def from_landscape(
        cls,
        landscape: Landscape,
        spl: CylSpline,
        range_long: tuple[nm | str, nm | str],
        range_lat: tuple[nm | str, nm | str],
        angle_max: float,
        temperature_time_const: float = 1.0,
        random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    ) -> ResumableAnnealing:
        """Prepare cylindric annealing (RMA) models."""
        model = landscape.cylindric_annealing_model(
            spl,
            distance_range_long=range_long,
            distance_range_lat=range_lat,
            angle_max=angle_max,
            temperature_time_const=temperature_time_const,
        )
        return cls._from_model(landscape, spl, model, random_seeds)

    @classmethod
    def from_landscape_filamentous(
        cls,
        landscape: Landscape,
        spl: CylSpline,
        range_long: tuple[nm | str, nm | str],
        angle_max: float,
        temperature_time_const: float = 1.0,
        random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    ) -> ResumableAnnealing:
        """Prepare filamentous annealing (RFA) models."""
        model = landscape.filamentous_annealing_model(
            distance_range=range_long,
            angle_max=angle_max,
            temperature_time_const=temperature_time_const,
        )
        return cls._from_model(landscape, spl, model, random_seeds)

    @classmethod
    def _from_model(
        cls,
        landscape: Landscape,
        spl: CylSpline,
        model: CylindricAnnealingModel | FilamentousAnnealingModel,
        random_seeds: Sequence[int],
    ) -> ResumableAnnealing:
        models = [model.with_seed(s) for s in random_seeds]
        for each in models:
            each.init_shift_random()
        return cls(
            landscape=landscape,
            spline=spl,
            models=models,
            temperature0=model.temperature(),
            epoch_size=_to_epoch_size(model.time_constant()),
        )

    def update_landscape(self, landscape: Landscape) -> None:
        """Replace the energy landscape while keeping the annealing states.

        If the new landscape is built around the molecules in different poses, the
        local coordinates of the models are updated, and the current states are moved
        to the nearest grid points of the new landscape so that the positions of the
        molecules are kept.
        """
        if landscape.energies.shape != self.landscape.energies.shape:
            raise ValueError(
                f"Shape of the new landscape {landscape.energies.shape} does not match "
                f"the current one {self.landscape.energies.shape}."
            )
        from cylindra._cylindra_ext import EnergyLandscape

        # Copy the energies only once and share them by all the models.
        energies = EnergyLandscape(landscape.energies)
        if _is_same_pose(landscape.molecules, self.landscape.molecules):
            coords = None
        else:
            coords = landscape.graph_coordinates()
        for model in self.models:
            shifts = model.shifts()
            if coords is not None:
                positions = self.landscape.world_positions(shifts)
                model.set_graph_coordinates(*coords)
                shifts = landscape.nearest_indices(positions)
            # NOTE: set_energy_landscape resets the shifts to the center.
            model.set_energy_landscape(energies)
            model.set_shifts(shifts)
        self.landscape = landscape

    def set_best(self, result: AnnealingResult) -> Molecules:
        """Record the best trial of the last run and return its molecules."""
        self.best_molecules = self.transform_molecules(result.indices)
        return self.best_molecules

    def next_reference(self) -> Molecules:
        """Molecules around which the next landscape should be built.

        The positions are the same as the current landscape, so that the range of the
        shifts is kept, while the orientations are those of the best trial of the last
        run, so that rotations are searched around them.
        """
        ref = self.landscape.molecules
        if self.best_molecules is None:
            return ref
        return Molecules(ref.pos, self.best_molecules.rotator, features=ref.features)

    def run(
        self,
        temperature_ratio: float,
        cool_completely: bool = False,
    ) -> list[AnnealingResult]:
        """Run annealing until the temperature reaches the given ratio."""
        return compute(*self.tasks(temperature_ratio, cool_completely))

    def tasks(
        self,
        temperature_ratio: float,
        cool_completely: bool = False,
    ) -> list[Delayed[AnnealingResult]]:
        """Delayed tasks of `run` for each trial."""
        temperature = self.temperature0 * temperature_ratio
        return [
            _resume_annealing(model, self.epoch_size, temperature, cool_completely)
            for model in self.models
        ]

    def transform_molecules(self, indices: NDArray[np.int32]) -> Molecules:
        """Molecules at the given landscape indices."""
        mole = self.landscape.transform_molecules(self.landscape.molecules, indices)
        return _update_mole_pos(mole, self.landscape.molecules, self.spline)


@delayed
def _resume_annealing(
    model: CylindricAnnealingModel | FilamentousAnnealingModel,
    epoch_size: int,
    temperature: float,
    cool_completely: bool,
) -> AnnealingResult:
    energies = [model.energy()]
    state = "not_converged"
    while model.temperature() > temperature:
        niter = model.iteration()
        model.simulate(epoch_size)
        energies.append(model.energy())
        if model.iteration() - niter < epoch_size:
            # reached the reject limit
            state = model.optimization_state()
            break
    if cool_completely:
        model.cool_completely()
        energies.append(model.energy())
    return AnnealingResult(
        energies=np.array(energies),
        epoch_size=epoch_size,
        time_const=model.time_constant(),
        indices=model.shifts(),
        niter=model.iteration(),
        state=state,
    )


def resample_extracted(
    loader: ExtractedSubvolumeLoader,
    molecules: Molecules,
) -> ExtractedSubvolumeLoader:
    """Resample the extracted subvolumes at the given molecules.

    `molecules` must correspond one by one to the molecules of `loader`. Subvolumes are
    resampled from the extracted ones using the transformation from the extracted
    molecules, without cropping the tomogram again.
    """
    ref = loader.molecules
    if molecules.count() != ref.count():
        raise ValueError(
            f"Number of molecules {molecules.count()} does not match the number of "
            f"extracted subvolumes {ref.count()}."
        )
    return loader.replace(
        molecules=molecules.with_features(ref.features[SUBVOLUME_INDEX])
    )


def _is_same_pose(mole0: Molecules, mole1: Molecules, tol: float = 1e-6) -> bool:
    """True if the molecules are at the same positions in the same orientations."""
    if not np.allclose(mole0.pos, mole1.pos, atol=tol):
        return False
    return bool(np.all((mole0.rotator.inv() * mole1.rotator).magnitude() < tol))


def _safe_div(a, b, default: float = 0.5):
    if b == 0 or np.isnan(b):
        return default
    return a / b


def randomize_roll(molecules: Molecules, rng: np.random.Generator) -> Molecules:
    """Rotate molecules around their y-axis by random angles.

    For molecules along a filament, the y-axis is parallel to the filament axis, so
    that the missing wedges of the subtomograms are distributed around the axis.
    """
    rotvec = np.zeros((molecules.count(), 3))
    rotvec[:, 1] = rng.uniform(-np.pi, np.pi, size=molecules.count())
    return molecules.rotate_by_rotvec_internal(rotvec)
