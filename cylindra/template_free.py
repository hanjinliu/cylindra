from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Sequence, TypeVar

import impy as ip
import numpy as np
from acryo import Molecules, SubtomogramLoader, alignment, pipe
from acryo.loader import LoaderBase
from magicclass.logging import getLogger
from numpy.typing import NDArray

from cylindra._dask import compute, delayed
from cylindra.components.landscape import (
    AnnealingResult,
    Landscape,
    _to_epoch_size,
    _update_mole_pos,
)
from cylindra.const import nm
from cylindra.widget_utils import FscResult

if TYPE_CHECKING:
    from cylindra._cylindra_ext import CylindricAnnealingModel
    from cylindra.components.spline import CylSpline

_Logger = getLogger("cylindra")


def adjust_down(
    a: float, a_min: float, num_iter: int, num_iter_offset: int = 3
) -> float:
    if num_iter < num_iter_offset or a <= a_min:
        return a
    diff = a - a_min
    return diff / (num_iter - num_iter_offset + 2) + a_min


_L = TypeVar("_L", bound=LoaderBase)


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
    ):
        return AlignmentParams(
            num_iter=0,
            cutoff=cutoff,
            max_shifts=max_shifts,
            rotations=tuple((a, a) for a in max_rotations),
            max_shifts_initial=max_shifts,
            max_rotations_initial=max_rotations,
        )

    @classmethod
    def from_last_result(
        cls,
        scale: nm,
        result: AlignmentResult,
        min_rot_step: float = 0.5,
    ):
        num_iter = result.params.num_iter + 1
        cutoff = scale / result.fsc.get_resolution(0.143)
        max_shifts = [
            adjust_down(a, scale, num_iter) for a in result.params.max_shifts_initial
        ]
        rotations = []
        for a in result.params.max_rotations_initial:
            rot_ = adjust_down(a, min_rot_step, num_iter)
            rotations.append((rot_, rot_))
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
            f"Max rotations (Z, Y, X) = {z_rot:.2f}°, {y_rot:.2f}°, {x_rot:.2f}°\n"
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
    ):
        return RMAAlignmentParams(
            num_iter=0,
            cutoff=cutoff,
            max_shifts=max_shifts,
            rotations=tuple((a, a) for a in max_rotations),
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
    ):
        # NOTE: Annealing is resumed in the next iteration, so the parameters that
        # determine the shape of the landscape (max_shifts, upsample_factor) and the
        # annealing models (temperature_time_const, num_trials) must not be changed.
        num_iter = result.params.num_iter + 1
        cutoff = scale / result.fsc.get_resolution(0.143)
        rotations = []
        for a in result.params.max_rotations_initial:
            rot_ = adjust_down(a, min_rot_step, num_iter)
            rotations.append((rot_, rot_))
        return replace(
            result.params,
            num_iter=num_iter,
            cutoff=cutoff,
            rotations=tuple(rotations),
        )

    def format(self) -> str:
        tz, ty, tx = self.max_shifts
        (z_rot, z_step), (y_rot, y_step), (x_rot, x_step) = self.rotations
        return (
            f"Iteration {self.num_iter}:\n"
            f"Relative cutoff frequency = {self.cutoff:.3f}\n"
            f"Max shifts (Z, Y, X) = {tz:.2f} nm, {ty:.2f} nm, {tx:.2f} nm\n"
            f"Max rotations (Z, Y, X) = {z_rot:.2f}°, {y_rot:.2f}°, {x_rot:.2f}°\n"
            f"... with steps = {z_step:.2f}°, {y_step:.2f}°, {x_step:.2f}°\n"
            f"Upsample factor = {self.upsample_factor}\n"
            f"Temperature time constant = {self.temperature_time_const:.2f}\n"
            f"Number of trials = {self.num_trials}"
        )


@dataclass
class RMAAlignmentResult(AlignmentResult):
    params: RMAAlignmentParams


_R = TypeVar("_R", bound="AlignmentResult")


@dataclass
class BaseAlignmentState(Generic[_R]):
    """State of the template-free alignment."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    mask: pipe.ImageProvider | pipe.ImageConverter | None = field(default=None)
    alignment_model: type[alignment.BaseAlignmentModel] = field(
        default=alignment.ZNCCAlignment
    )
    min_rotation_step: float = 0.5
    results: list[_R] = field(default_factory=list)

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


@dataclass
class AlignmentState(BaseAlignmentState[AlignmentResult]):
    """State of the template-free alignment.

    Template-free alignment proceeds in iterations consisting of FSC step and alignment
    step.
    """

    def fsc_step_init(
        self,
        loader: SubtomogramLoader,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
    ) -> AlignmentResult:
        """Run the first FSC step"""
        fsc_result, avg = loader.fsc_with_average(
            seed=self.rng.integers(0, 2**32),
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        cutoff = loader.scale / fsc.get_resolution(0.143)
        params = AlignmentParams.init(cutoff, max_shifts, max_rotations)
        this_result = AlignmentResult(fsc, avg, None, params)
        self.results.append(this_result)
        return this_result

    def next_params(self, scale: float) -> AlignmentParams:
        last_result = self.results[-1]
        return AlignmentParams.from_last_result(
            scale, last_result, self.min_rotation_step
        )

    def fsc_step(self, loader: SubtomogramLoader) -> AlignmentResult:
        """Run a FSC step and return the result."""
        last_result = self.results[-1]
        mask = self._prep_mask(last_result.avg.value, loader.scale)
        params = self.next_params(loader.scale)
        return self._fsc_step_impl(loader, mask, params)

    def align_step(self, loader: _L) -> _L:
        """Run an alignment step and return updated loader."""
        result = self.results[-1]
        return loader.align(
            result.avg.value, mask=result.mask, max_shifts=result.params.max_shifts,
            rotations=result.params.rotations, cutoff=result.params.cutoff,
            alignment_model=self.alignment_model,
        )  # fmt: skip

    def _fsc_step_impl(
        self,
        loader: SubtomogramLoader,
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
class RMAAlignmentState(BaseAlignmentState[RMAAlignmentResult]):
    """State of the template-free RMA alignment.

    Unlike `AlignmentState`, alignment is not restarted from scratch in every
    iteration. The annealing schedule from the initial temperature to the final
    temperature is split into `max_num_iters` parts, and in each iteration, the
    annealing is resumed from the last state of the previous iteration using the
    energy landscape updated with the new template.
    """

    max_num_iters: int = 20
    final_temperature_ratio: float = 1e-5

    def fsc_step_init(
        self,
        loader: SubtomogramLoader,
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
        cutoff = loader.scale / fsc.get_resolution(0.143)
        params = RMAAlignmentParams.init(
            cutoff, max_shifts, max_rotations, upsample_factor,
            temperature_time_const, num_trials,
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
        frac = min((self.num_iter + 1) / self.max_num_iters, 1.0)
        return self.final_temperature_ratio**frac

    def next_params(self, scale: float) -> RMAAlignmentParams:
        last_result = self.results[-1]
        return RMAAlignmentParams.from_last_result(
            scale, last_result, self.min_rotation_step
        )

    def fsc_step(self, loader: SubtomogramLoader) -> RMAAlignmentResult:
        """Run a FSC step and return the result."""
        last_result = self.results[-1]
        mask = self._prep_mask(last_result.avg.value, loader.scale)
        params = self.next_params(loader.scale)
        return self._fsc_step_impl(loader, mask, params)

    def built_landscape_step(
        self,
        loader: SubtomogramLoader,
    ) -> Landscape:
        """Build the correlation landscape and return updated loader."""
        result = self.results[-1]
        return Landscape.from_loader(
            loader=loader,
            template=result.avg,
            mask=result.mask,
            max_shifts=result.params.max_shifts,
            upsample_factor=result.params.upsample_factor,
            alignment_model=self.alignment_model.with_params(
                rotations=result.params.rotations,
                cutoff=result.params.cutoff,
                tilt=loader.tilt_model,
            ),
        ).normed()

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
            random_seeds=self.rng.integers(
                0, 2**31 - 1, size=params.num_trials
            ).tolist(),
        )

    def rma_step(self, annealing: ResumableAnnealing, final: bool = False) -> Molecules:
        """Resume annealing and return the molecules of the best trial.

        Annealing runs until the temperature reaches the target of this iteration. If
        `final` is true, annealing will be finished regardless of the iteration.
        """
        results = annealing.run(self.temperature_ratio(final), cool_completely=final)
        if all(result.state == "failed" for result in results):
            raise RuntimeError(
                "Failed to optimize for all trials. You may check the distance range."
            )
        if final:
            _Logger.print_table(
                {
                    "Iteration": [r.niter for r in results],
                    "Score": [f"{-float(r.energies[-1]):.5g}" for r in results],
                    "State": [r.state for r in results],
                }
            )
        best = min(results, key=lambda r: r.energies[-1])
        return annealing.transform_molecules(best.indices)

    def _fsc_step_impl(
        self,
        loader: SubtomogramLoader,
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
class ResumableAnnealing:
    """Independent annealing trials along a spline that can be resumed.

    The energy landscape can be replaced between runs without resetting the states
    (shifts, temperature and the binding potential) of the annealing models.
    """

    landscape: Landscape
    spline: CylSpline
    models: list[CylindricAnnealingModel]
    temperature0: float
    epoch_size: int

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
        model = landscape.cylindric_annealing_model(
            spl,
            distance_range_long=range_long,
            distance_range_lat=range_lat,
            angle_max=angle_max,
            temperature_time_const=temperature_time_const,
        )
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
        """Replace the energy landscape while keeping the annealing states."""
        if landscape.energies.shape != self.landscape.energies.shape:
            raise ValueError(
                f"Shape of the new landscape {landscape.energies.shape} does not match "
                f"the current one {self.landscape.energies.shape}."
            )
        for model in self.models:
            shifts = model.shifts()
            # NOTE: set_energy_landscape resets the shifts to the center.
            model.set_energy_landscape(landscape.energies)
            model.set_shifts(shifts)
        self.landscape = landscape

    def run(
        self,
        temperature_ratio: float,
        cool_completely: bool = False,
    ) -> list[AnnealingResult]:
        """Run annealing until the temperature reaches the given ratio."""
        temperature = self.temperature0 * temperature_ratio
        tasks = [
            _resume_annealing(model, self.epoch_size, temperature, cool_completely)
            for model in self.models
        ]
        return compute(*tasks)

    def transform_molecules(self, indices: NDArray[np.int32]) -> Molecules:
        """Molecules at the given landscape indices."""
        mole = self.landscape.transform_molecules(self.landscape.molecules, indices)
        return _update_mole_pos(mole, self.landscape.molecules, self.spline)


@delayed
def _resume_annealing(
    model: CylindricAnnealingModel,
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
