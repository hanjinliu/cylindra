from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import impy as ip
import numpy as np
from acryo import SubtomogramLoader, alignment, pipe
from acryo.loader import BatchLoader
from numpy.typing import NDArray

from cylindra.components.landscape import Landscape
from cylindra.const import nm
from cylindra.widget_utils import FscResult


def adjust_down(
    a: float, a_min: float, num_iter: int, num_iter_offset: int = 3
) -> float:
    if num_iter < num_iter_offset or a <= a_min:
        return a
    diff = a - a_min
    return diff / (num_iter - num_iter_offset + 2) + a_min


def adjust_up(a: float, a_max: float, num_iter: int, num_iter_offset: int = 3) -> float:
    if num_iter < num_iter_offset or a >= a_max:
        return a
    diff = a_max - a
    return a_max - diff / (num_iter - num_iter_offset + 2)


_L = TypeVar("_L", bound=BatchLoader)


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
    upsample_factor_final: int
    temperature_time_const: float
    temperature_time_const_final: float
    num_trials: int

    @classmethod
    def init(
        cls,
        cutoff: float,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
        upsample_factor_final: int,
        temperature_time_const_final: float,
    ):
        return RMAAlignmentParams(
            num_iter=0,
            cutoff=cutoff,
            max_shifts=max_shifts,
            rotations=tuple((a, a) for a in max_rotations),
            max_shifts_initial=max_shifts,
            max_rotations_initial=max_rotations,
            upsample_factor=2,
            upsample_factor_final=upsample_factor_final,
            temperature_time_const=temperature_time_const_final / 2.5,
            temperature_time_const_final=temperature_time_const_final,
            num_trials=2,
        )

    @classmethod
    def from_last_result(
        cls,
        scale: nm,
        result: RMAAlignmentResult,
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

        upsample_factor = adjust_up(
            2,
            result.params.upsample_factor_final,
            num_iter,
        )
        temperature_time_const = adjust_up(
            result.params.temperature_time_const_final / 1.6,
            result.params.temperature_time_const_final,
            num_iter,
        )
        num_trials = adjust_up(2, 5, num_iter)
        return RMAAlignmentParams(
            num_iter=num_iter,
            cutoff=cutoff,
            max_shifts=tuple(max_shifts),
            rotations=tuple(rotations),
            max_shifts_initial=result.params.max_shifts_initial,
            max_rotations_initial=result.params.max_rotations_initial,
            upsample_factor=int(round(upsample_factor)),
            temperature_time_const=temperature_time_const,
            upsample_factor_final=result.params.upsample_factor_final,
            temperature_time_const_final=result.params.temperature_time_const_final,
            num_trials=int(round(num_trials)),
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
    """State of the template-free alignment."""

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
    def fsc_step_init(
        self,
        loader: SubtomogramLoader,
        max_shifts: tuple[nm, nm, nm],
        max_rotations: tuple[float, float, float],
        upsample_factor_final: int,
        temperature_time_const_final: float,
    ) -> RMAAlignmentResult:
        """Run the first FSC step"""
        fsc_result, avg = loader.fsc_with_average(
            seed=self.rng.integers(0, 2**32),
        )
        fsc = FscResult.from_dataframe(fsc_result, loader.scale)
        avg = ip.asarray(avg, axes="zyx").set_scale(zyx=loader.scale, unit="nm")
        cutoff = loader.scale / fsc.get_resolution(0.143)
        params = RMAAlignmentParams.init(
            cutoff, max_shifts, max_rotations, upsample_factor_final,
            temperature_time_const_final
        )  # fmt: skip
        this_result = RMAAlignmentResult(fsc, avg, None, params)
        self.results.append(this_result)
        return this_result

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

    def rma_step(
        self,
        landscape: Landscape,
        loader: SubtomogramLoader,
        spl,
        range_long,
        range_lat,
        angle_max,
    ) -> SubtomogramLoader:
        result = self.results[-1]
        mole, ann_results = landscape.run_annealing_along_spline(
            spl,
            range_long=range_long,
            range_lat=range_lat,
            angle_max=angle_max,
            temperature_time_const=result.params.temperature_time_const,
            random_seeds=self.rng.integers(
                0, 2**32, size=result.params.num_trials
            ).tolist(),
        )
        return SubtomogramLoader(
            loader.image,
            mole,
            order=loader.order,
            scale=loader.scale,
            output_shape=loader.output_shape,
        )

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
