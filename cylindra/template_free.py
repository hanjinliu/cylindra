from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Sequence

import impy as ip
import numpy as np
import polars as pl
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
from cylindra.const import MoleculesHeader as Mole
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
    """Parameters of an alignment.

    The parameters are those of the alignment using the average of the result that
    they belong to as the template, that is, the alignment in iteration `num_iter + 1`.
    """

    # number of the iterations completed before the alignment
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

    def next_iteration(
        self,
        scale: nm,
        cutoff: float,
        min_rot_step: float = 0.5,
        max_rot_step: float = 30.0,
    ) -> AlignmentParams:
        """Parameters of the alignment in the next iteration.

        `cutoff` should be determined by the FSC of the template of the next alignment
        (see `cutoff_of`).
        """
        num_iter = self.num_iter + 1
        max_shifts = [adjust_down(a, scale, num_iter) for a in self.max_shifts_initial]
        rotations = [
            narrow_rotation_range(last, a, min_rot_step, max_rot_step, num_iter)
            for last, a in zip(self.rotations, self.max_rotations_initial, strict=True)
        ]
        return AlignmentParams(
            num_iter=num_iter,
            cutoff=cutoff,
            max_shifts=tuple(max_shifts),
            rotations=tuple(rotations),
            max_shifts_initial=self.max_shifts_initial,
            max_rotations_initial=self.max_rotations_initial,
        )

    def format(self) -> str:
        tz, ty, tx = self.max_shifts
        (z_rot, z_step), (y_rot, y_step), (x_rot, x_step) = self.rotations
        return (
            f"Iteration {self.num_iter + 1}:\n"
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
    # averages of the half sets if they are aligned independently (gold-standard), in
    # which case the templates are weighted by FSC (see `fsc_weighted`)
    half_avgs: tuple[ip.ImgArray, ip.ImgArray] | None = None

    def template(self, half: int | None = None) -> NDArray[np.float32]:
        """Template image of the next alignment, of all or of the given half set."""
        if half is None:
            return self.avg.value
        if self.half_avgs is None:
            raise ValueError("Half sets are not aligned independently.")
        return self.half_avgs[half].value


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

    def next_iteration(
        self,
        scale: nm,
        cutoff: float,
        min_rot_step: float = 0.5,
        max_rot_step: float = 30.0,
    ) -> RMAAlignmentParams:
        """Parameters of the alignment in the next iteration.

        See `AlignmentParams.next_iteration` for details.
        """
        # NOTE: Annealing is resumed in the next iteration, so the parameters that
        # determine the shape of the landscape (max_shifts, upsample_factor) and the
        # annealing models (temperature_time_const, num_trials) must not be changed.
        num_iter = self.num_iter + 1
        rotations = [
            narrow_rotation_range(last, a, min_rot_step, max_rot_step, num_iter)
            for last, a in zip(self.rotations, self.max_rotations_initial, strict=True)
        ]
        return replace(
            self,
            num_iter=num_iter,
            cutoff=cutoff,
            rotations=tuple(rotations),
        )

    def format(self) -> str:
        tz, ty, tx = self.max_shifts
        (z_rot, z_step), (y_rot, y_step), (x_rot, x_step) = self.rotations
        out = (
            f"Iteration {self.num_iter + 1}:\n"
            f"Relative cutoff frequency = {self.cutoff:.3f}\n"
            f"Max shifts (Z, Y, X) = {tz:.2f} nm, {ty:.2f} nm, {tx:.2f} nm\n"
            f"Max rotations (Z, Y, X) = {z_rot:.2f}°, {y_rot:.2f}°, {x_rot:.2f}° "
            f"... with steps = {z_step:.2f}°, {y_step:.2f}°, {x_step:.2f}°"
        )
        # parameters fixed during annealing are shown only in the first iteration
        if self.num_iter == 0:
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
    # alignment models of the last result, for all (None) or each half set
    _model_cache: tuple[R, dict[int | None, TomographyInput]] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def num_iter(self) -> int:
        """Number of completed iterations."""
        return len(self.results) - 1

    def _next_params[P: AlignmentParams](
        self,
        last_params: P,
        fsc: FscResult,
        scale: nm,
        size: int,
        weighted: bool,
    ) -> P:
        """Parameters of the next alignment after a result of `fsc` is added.

        The cutoff frequency is determined by `fsc`, the FSC of the template of the next
        alignment. See `cutoff_of` for the other parameters.
        """
        return last_params.next_iteration(
            scale,
            cutoff_of(fsc, scale, size, weighted),
            self.min_rotation_step,
            self.max_rotation_step,
        )

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
        return _resolve_mask(self.mask, avg, scale)

    def _alignment_model(
        self, loader: LoaderBase, half: int | None = None
    ) -> SharedTemplateModel:
        """Return the alignment model factory of the current iteration.

        The alignment model is created only once for each iteration (and each half
        set), so that the template images are rotated only once and shared by all the
        loaders. If `half` is given, the average of the half set is used as the
        template instead of the average of all the molecules. The template images are
        the averages of the last result, which are weighted by FSC if the half sets are
        aligned independently, as the FSC is then not inflated.
        """
        result = self.results[-1]
        if self._model_cache is None or self._model_cache[0] is not result:
            self._model_cache = (result, {})
        models = self._model_cache[1]
        if half not in models:
            template = result.template(half)
            if result.half_avgs is not None:
                template = fsc_weighted(template, result.fsc)
            mask = loader.normalize_mask(result.mask, reshape_to=template.shape)
            models[half] = self.alignment_model.with_params(
                rotations=result.params.rotations,
                cutoff=result.params.cutoff,
            )(template, mask)
        return SharedTemplateModel(models[half])


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
        """Run the first FSC step.

        If the molecules are assigned to half sets, they are regarded as the half sets
        aligned independently. See `fsc_step` for details. As there is no template
        yet, the mask is created from the initial average, and used for both the FSC
        and the first alignment.
        """
        fsc, avg, half_avgs, mask = _fsc_with_average(loader, self.mask, self.rng)
        cutoff = cutoff_of(
            fsc, _scale_of(loader), min(avg.shape), half_avgs is not None
        )
        params = AlignmentParams.init(
            cutoff, max_shifts, max_rotations, self.max_rotation_step
        )
        this_result = AlignmentResult(fsc, avg, mask, params, half_avgs)
        self.results.append(this_result)
        return this_result

    def fsc_step(self, loader: LoaderBase) -> AlignmentResult:
        """Run a FSC step and return the result.

        The result has the parameters of the next alignment. If the molecules are
        assigned to half sets (see `assign_halves`), each half set is aligned to its
        own average (gold-standard). FSC is then calculated between the averages of
        the half sets instead of randomly split molecules, so that it is not inflated
        by the noise shared through the same template.
        """
        scale = _scale_of(loader)
        mask = self._prep_mask(self.results[-1].avg.value, scale)
        fsc, avg, half_avgs, _ = _fsc_with_average(loader, mask, self.rng)
        params = self._next_params(
            self.results[-1].params, fsc, scale, min(avg.shape), half_avgs is not None
        )
        this_result = AlignmentResult(fsc, avg, mask, params, half_avgs)
        self.results.append(this_result)
        return this_result

    def align_step[L: LoaderBase](self, loader: L) -> L:
        """Run an alignment step and return updated loader.

        If the molecules are assigned to half sets (see `assign_halves`), the molecules
        of each half set are aligned to the average of the same half set.
        """
        if not _has_halves(loader):
            return self._align(loader)
        labels = loader.molecules.features[HALF_SET].to_numpy()
        indices = [np.flatnonzero(labels == half) for half in (0, 1)]
        aligned = [
            self._align(loader.replace(molecules=loader.molecules[idx]), half)
            for half, idx in enumerate(indices)
        ]
        # molecules are restored to the original order
        mole = Molecules.concat([each.molecules for each in aligned])
        return loader.replace(molecules=mole[np.argsort(np.concatenate(indices))])

    def _align[L: LoaderBase](self, loader: L, half: int | None = None) -> L:
        result = self.results[-1]
        model = self._alignment_model(loader, half)
        return loader.align(
            model.template, mask=model.mask, max_shifts=result.params.max_shifts,
            alignment_model=model,
        )  # fmt: skip


@dataclass
class BaseAnnealingAlignmentState(BaseAlignmentState[RMAAlignmentResult]):
    """State of the template-free alignment using resumable annealing.

    Unlike `AlignmentState`, alignment is not restarted from scratch in every
    iteration. In each iteration, the annealing is resumed from the last state of the
    previous iteration using the energy landscape updated with the new template, and
    the temperature is cooled down by `cooling_fraction`, so that it reaches
    `final_temperature_ratio` of the initial one in `max_num_iters` iterations.

    If `final_landscape_temperature_ratio` is given, the temperature reaches it in
    `max_num_iters - 1` iterations instead, and the rest of the annealing is run in
    the last iteration without rebuilding the landscape (see `rebuilds_landscape`), as
    the average is hardly improved at such low temperature.
    """

    max_num_iters: int = 8
    final_temperature_ratio: float = 1e-5
    final_landscape_temperature_ratio: float | None = None

    def __post_init__(self):
        min_iters = 1 if self.final_landscape_temperature_ratio is None else 2
        if self.max_num_iters < min_iters:
            raise ValueError(
                f"max_num_iters must be at least {min_iters}, got {self.max_num_iters}."
            )
        if not 0 < self.final_temperature_ratio < 1:
            raise ValueError(
                "final_temperature_ratio must be in (0, 1), got "
                f"{self.final_temperature_ratio}."
            )
        if (ratio := self.final_landscape_temperature_ratio) is not None and not (
            self.final_temperature_ratio < ratio < 1
        ):
            raise ValueError(
                "final_landscape_temperature_ratio must be in "
                f"({self.final_temperature_ratio}, 1), got {ratio}."
            )

    def is_converged(self) -> bool:
        return False

    @property
    def cooling_fraction(self) -> float:
        """Ratio of the temperature to be cooled down in each iteration."""
        if (ratio := self.final_landscape_temperature_ratio) is None:
            return self.final_temperature_ratio ** (1 / self.max_num_iters)
        return ratio ** (1 / (self.max_num_iters - 1))

    def fsc_step_init(
        self,
        loader: LoaderBase | tuple[LoaderBase, LoaderBase],
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
        upsample_factor: int,
        temperature_time_const: float,
        num_trials: int,
    ) -> RMAAlignmentResult:
        """Run the first FSC step.

        If the molecules are assigned to half sets, or a pair of loaders of the half
        sets is given, they are regarded as the half sets aligned independently. See
        `fsc_step` for details. As there is no template yet, the mask is created from
        the initial average, and used for both the FSC and the first alignment.
        """
        fsc, avg, half_avgs, mask = _fsc_with_average(loader, self.mask, self.rng)
        cutoff = cutoff_of(
            fsc, _scale_of(loader), min(avg.shape), half_avgs is not None
        )
        params = RMAAlignmentParams.init(
            cutoff, max_shifts, max_rotations, upsample_factor,
            temperature_time_const, num_trials, self.max_rotation_step,
        )  # fmt: skip
        this_result = RMAAlignmentResult(fsc, avg, mask, params, half_avgs)
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
        if (ratio := self.final_landscape_temperature_ratio) is None:
            ratio = self.final_temperature_ratio
        return max(ratio, frac)

    def rebuilds_landscape(self) -> bool:
        """True if the landscape should be rebuilt with the template in the next step.

        If `final_landscape_temperature_ratio` is given, the landscape of the previous
        iteration is used in the last iteration.
        """
        if self.final_landscape_temperature_ratio is None:
            return True
        return not self.is_final_iteration()

    def fsc_step(
        self, loader: LoaderBase | tuple[LoaderBase, LoaderBase]
    ) -> RMAAlignmentResult:
        """Run a FSC step and return the result.

        The result has the parameters of the next alignment. If the molecules are
        assigned to half sets (see `assign_halves`), or a pair of loaders of the half
        sets is given, each half set is aligned to its own average (gold-standard).
        FSC is then calculated between the averages of the half sets instead of
        randomly split molecules, so that it is not inflated by the noise shared
        through the same template.
        """
        scale = _scale_of(loader)
        mask = self._prep_mask(self.results[-1].avg.value, scale)
        fsc, avg, half_avgs, _ = _fsc_with_average(loader, mask, self.rng)
        params = self._next_params(
            self.results[-1].params, fsc, scale, min(avg.shape), half_avgs is not None
        )
        this_result = RMAAlignmentResult(fsc, avg, mask, params, half_avgs)
        self.results.append(this_result)
        return this_result

    def built_landscape_step(
        self,
        loader: SubtomogramLoader,
        norm: bool = True,
        half: int | None = None,
    ) -> Landscape:
        """Build the correlation landscape and return updated loader."""
        return self.built_landscapes_step([loader], norm=norm, half=half)[0]

    def built_landscapes_step(
        self,
        loaders: Sequence[SubtomogramLoader],
        norm: bool = True,
        half: int | None = None,
    ) -> list[Landscape]:
        """Build the correlation landscapes of the loaders in parallel.

        If `half` is given, the average of the half set is used as the template.
        """
        result = self.results[-1]
        model = self._alignment_model(loaders[0], half)
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


@dataclass
class RMAAlignmentState(BaseAnnealingAlignmentState):
    """State of the template-free RMA alignment."""

    final_landscape_temperature_ratio: float | None = 5e-4

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


# features added by the alignment of acryo loaders, in the same order
_ALIGN_FEATURES = [Mole.score] + [f"align-d{c}" for c in "zyx"]
_ALIGN_FEATURES += [f"align-d{c}rot" for c in "zyx"]


def as_alignment_output(original: Molecules, aligned: list[Molecules]) -> Molecules:
    """Concatenate molecules aligned using the extracted subtomograms.

    The temporary features (the subvolume index and the half set) are dropped, and the
    feature columns are ordered in the same way as aligning the original loader
    directly.
    """
    mole = Molecules.concat(aligned).drop_features(
        SUBVOLUME_INDEX, HALF_SET, strict=False
    )
    columns = mole.features.columns
    order = [c for c in original.features.columns if c in columns]
    order += [c for c in _ALIGN_FEATURES if c in columns and c not in order]
    order += [c for c in columns if c not in order]
    mole.features = mole.features.select(order)
    return mole


def _is_same_pose(mole0: Molecules, mole1: Molecules, tol: float = 1e-6) -> bool:
    """True if the molecules are at the same positions in the same orientations."""
    if not np.allclose(mole0.pos, mole1.pos, atol=tol):
        return False
    return bool(np.all((mole0.rotator.inv() * mole1.rotator).magnitude() < tol))


def split_into_halves(counts: Sequence[int], rng: np.random.Generator) -> list[int]:
    """Split items (such as splines) into two half sets of balanced total counts.

    Items are assigned in descending order of the counts (in random order for ties) to
    the half set with fewer counts in total. The index of the half set (0 or 1) of each
    item is returned.
    """
    if len(counts) < 2:
        raise ValueError(f"At least two items are required, got {len(counts)}.")
    order = sorted(rng.permutation(len(counts)).tolist(), key=lambda i: -counts[i])
    totals = [0, 0]
    halves = [0] * len(counts)
    for i in order:
        half = int(totals[1] < totals[0])
        halves[i] = half
        totals[half] += counts[i]
    return halves


# feature of the half set (0 or 1) of each molecule during alignment
HALF_SET = "align-halfset-id"


def assign_halves[L: LoaderBase](loader: L, rng: np.random.Generator) -> L:
    """Randomly assign the molecules of a loader to two half sets.

    The half set of each molecule is recorded in the `HALF_SET` feature, so that the
    half sets are aligned and averaged in the same loader. The numbers of molecules of
    the half sets differ by one at most. The feature is temporary and dropped by
    `as_alignment_output`.
    """
    num = loader.molecules.count()
    if num < 2:
        raise ValueError(
            f"At least two molecules are required to be split into half sets, got {num}."
        )
    labels = rng.permutation(np.arange(num) % 2).astype(np.int8)
    return loader.replace(
        molecules=loader.molecules.with_features(pl.Series(HALF_SET, labels))
    )


def _has_halves(loader: LoaderBase) -> bool:
    """True if the molecules are assigned to half sets by `assign_halves`."""
    return HALF_SET in loader.molecules.features.columns


def _fsc_with_average(
    loader: LoaderBase | tuple[LoaderBase, LoaderBase],
    mask: NDArray[np.float32] | pipe.ImageProvider | pipe.ImageConverter | None,
    rng: np.random.Generator,
) -> tuple[
    FscResult,
    ip.ImgArray,
    tuple[ip.ImgArray, ip.ImgArray] | None,
    NDArray[np.float32] | None,
]:
    """Calculate FSC, the average of all the molecules and those of the half sets.

    If the molecules are assigned to half sets (see `assign_halves`), or a pair of
    loaders of the half sets is given, FSC is calculated between the averages of the
    half sets, which are also returned. Otherwise, molecules are randomly split into
    two sets to calculate FSC, and None is returned for the half sets. If `mask` is an
    image converter, the mask is created from the average. The mask array used for FSC
    is returned as well.
    """
    if isinstance(loader, tuple):
        halves = np.stack([each.average() for each in loader])
        counts = [each.molecules.count() for each in loader]
    elif _has_halves(loader):
        # averages of the half sets are calculated at once
        avgs = loader.groupby(HALF_SET).average()
        halves = np.stack([avgs[0], avgs[1]])
        counts = np.bincount(loader.molecules.features[HALF_SET], minlength=2)
    else:
        fsc_result, (img0, img1), _mask = loader.fsc_with_halfmaps(
            mask, seed=rng.integers(0, 2**32)
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = _as_imgarray((img0 + img1) / 2, loader.scale)
        return fsc, avg, None, None if mask is None else _mask
    any_loader = loader[0] if isinstance(loader, tuple) else loader
    scale = any_loader.scale
    avg = np.average(halves, axis=0, weights=counts).astype(np.float32)
    # the mask is created before normalization, as `fsc_with_halfmaps` of acryo does
    mask = _resolve_mask(mask, avg, scale)
    # same normalization as `fsc_with_halfmaps` of acryo loaders (zero_norm=True)
    offset = halves.mean()
    halves -= offset
    avg -= offset
    shape = halves.shape[1:]
    if mask is not None:
        mask = any_loader.normalize_mask(mask, reshape_to=shape)
    _mask = 1.0 if mask is None else mask
    img0, img1 = (ip.asarray(half * _mask, axes="zyx") for half in halves)
    freq, fsc = ip.fsc(img0, img1, dfreq=1.5 / min(shape))
    return (
        FscResult(freq, fsc, np.zeros_like(fsc), scale),
        _as_imgarray(avg, scale),
        (_as_imgarray(halves[0], scale), _as_imgarray(halves[1], scale)),
        mask,
    )


def _resolve_mask(
    mask: NDArray[np.float32] | pipe.ImageProvider | pipe.ImageConverter | None,
    avg: NDArray[np.float32],
    scale: nm,
) -> NDArray[np.float32] | None:
    """Mask array for the average `avg`."""
    if mask is None or isinstance(mask, np.ndarray):
        return mask
    elif isinstance(mask, pipe.ImageProvider):
        return mask.provide(scale)
    else:
        return mask.convert(avg, scale)


def fsc_weighted(img: NDArray[np.float32], fsc: FscResult) -> NDArray[np.float32]:
    """Weight the frequency components of a template image by its FSC.

    Each frequency is weighted by C_ref = sqrt(2 FSC / (1 + FSC)) (Rosenthal and
    Henderson, 2003), the expected correlation between the average and the true
    structure, where FSC is that between the independent half sets (negative values
    are regarded as zero). Unlike low-pass filtering at the resolution, frequencies
    are down-weighted according to their signal-to-noise ratios, so that frequencies
    beyond the resolution can also be used for alignment (see `cutoff_of`).
    """
    freqs = [np.fft.fftfreq(s) for s in img.shape[:-1]]
    freqs.append(np.fft.rfftfreq(img.shape[-1]))
    grids = np.meshgrid(*freqs, indexing="ij", sparse=True)
    radius = np.sqrt(sum(grid**2 for grid in grids))
    fsc_values = np.clip(np.nan_to_num(fsc.mean), 0.0, 1.0)
    c_ref = np.sqrt(2 * fsc_values / (1 + fsc_values))
    # frequencies beyond the last shell of FSC are not used
    weight = np.interp(radius, fsc.freq, c_ref, right=0.0)
    axes = tuple(range(img.ndim))
    out = np.fft.irfftn(np.fft.rfftn(img) * weight, s=img.shape, axes=axes)
    return out.astype(np.float32)


def cutoff_of(fsc: FscResult, scale: nm, size: int, weighted: bool) -> float:
    """Relative cutoff frequency of the alignment using a template of FSC `fsc`.

    If the template is not weighted by FSC, it is low-pass filtered at its resolution
    at FSC=0.143. Otherwise, as frequencies of low signal-to-noise ratios are
    down-weighted (see `fsc_weighted`), frequencies beyond the resolution are also
    used so that the resolution can be improved in the next iteration. Following
    RELION, the cutoff is the larger of 10 Fourier pixels beyond FSC=0.5 and 5 beyond
    FSC=0.143, where `size` is the box size in pixels. 1.0 (no low-pass filter) is
    returned at most.
    """
    if not weighted:
        return _safe_div(scale, fsc.get_resolution(0.143))
    freq_0500 = _crossing_freq(fsc, 0.5, scale)
    freq_0143 = _crossing_freq(fsc, 0.143, scale)
    return min(max(freq_0500 + 10 / size, freq_0143 + 5 / size), 1.0)


def _crossing_freq(fsc: FscResult, threshold: float, scale: nm) -> float:
    """Frequency (1/pixel) at which FSC drops below `threshold`.

    Infinity is returned if FSC is above the threshold at all frequencies.
    """
    res = fsc.get_resolution(threshold)
    if np.isnan(res):
        return np.inf
    if res == 0:
        return 0.0
    return scale / res


def _scale_of(loader: LoaderBase | tuple[LoaderBase, LoaderBase]) -> nm:
    return loader[0].scale if isinstance(loader, tuple) else loader.scale


def _as_imgarray(img: NDArray[np.float32], scale: nm) -> ip.ImgArray:
    return ip.asarray(img, axes="zyx").set_scale(zyx=scale, unit="nm")


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
