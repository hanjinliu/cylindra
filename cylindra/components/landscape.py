from __future__ import annotations

from typing import TYPE_CHECKING, SupportsIndex, SupportsInt
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import polars as pl
from acryo import Molecules, pipe, alignment

from cylindra.const import MoleculesHeader as Mole, nm
from .cyl_spline import CylSpline

if TYPE_CHECKING:
    from acryo.loader._base import LoaderBase, TemplateInputType, MaskInputType
    from acryo.alignment._base import ParametrizedModel, TomographyInput
    from dask import array as da
    from cylindra._cylindra_ext import CylindricAnnealingModel


@dataclass
class Landscape:
    """
    Energy landscape array.

    Parameters
    ----------
    energy_array : NDArray[np.float32]
        4D array of energy values.
    molecules : Molecules
        Molecules object.
    argmax : NDArray[np.int32] | None
        Argmax indices to track which template resulted in the best alignment.
    alignment_model : ParametrizedModel | TomographyInput
        Alignment model used.
    scale_factor : float
        Scale factor to convert from pixels to nanometers (upsampling considered).
        ``scale / upsample_factor`` will be passed to this parameter from the GUI.
    """

    energy_array: NDArray[np.float32]
    molecules: Molecules
    argmax: NDArray[np.int32] | None
    alignment_model: ParametrizedModel | TomographyInput
    scale_factor: float

    def __getitem__(
        self, key: slice | list[SupportsIndex] | NDArray[np.integer]
    ) -> Landscape:
        """Subset of the landscape."""
        if not isinstance(key, (slice, list, np.ndarray)):
            raise TypeError(f"Invalid type of key: {type(key)}")
        energy = self.energy_array[key]
        mole = self.molecules.subset(key)
        argmax = self.argmax[key] if self.argmax is not None else None
        return Landscape(energy, mole, argmax, self.alignment_model, self.scale_factor)

    @property
    def offset(self) -> NDArray[np.int32]:
        """Shift from the corner (0, 0, 0) to the center."""
        shift = (np.array(self.energy_array.shape[1:]) - 1) / 2
        return shift.astype(np.int32)

    @property
    def offset_nm(self) -> NDArray[np.float32]:
        """Offset in nm."""
        return self.offset * self.scale_factor

    @classmethod
    def from_loader(
        cls,
        loader: LoaderBase,
        template: TemplateInputType,
        mask: MaskInputType = None,
        max_shifts: tuple[nm, nm, nm] = (0.8, 0.8, 0.8),
        upsample_factor: int = 5,
        alignment_model=alignment.ZNCCAlignment,
    ) -> Landscape:
        """
        Construct a landscape from a loader object.

        Parameters
        ----------
        loader : LoaderBase
            Any loader object from ``acryo``.
        template : template input type
            Template image or a list of template images to be used.
        mask : mask input type, optional
            Mask image to be used, by default None
        max_shifts : (float, float, float), optional
            Maximum shifts in nm, in (Z, Y, X) order.
        upsample_factor : int
            Upsampling factor for landscape construction.
        alignment_model : alignment model object
            Alignment model to be used to evaluate correlation score.
        """
        if isinstance(template, (str, Path)):
            template = pipe.from_file(template)
            multi = False
        elif isinstance(template, (list, tuple)) and isinstance(
            next(iter(template), None), (str, Path)
        ):
            template = pipe.from_files(template)
            multi = True
        elif isinstance(template, np.ndarray):
            multi = template.ndim == 4
        else:
            raise TypeError(f"Invalid type of template: {type(template)}")

        score_dsk = loader.construct_landscape(
            template,
            mask=mask,
            max_shifts=max_shifts,
            upsample=upsample_factor,
            alignment_model=alignment_model,
        )
        score, argmax = _calc_landscape(
            alignment_model, score_dsk, multi_templates=multi
        )
        return cls(
            energy_array=-score,
            molecules=loader.molecules,
            argmax=argmax,
            alignment_model=alignment_model,
            scale_factor=loader.scale / upsample_factor,
        )

    def transform_molecules(
        self, molecules: Molecules, indices: NDArray[np.int32]
    ) -> Molecules:
        """
        Transform the input molecules based on the landscape.

        Parameters
        ----------
        molecules : Molecules
            Molecules object to be transformed.
        indices : integer array
            Indices in the landscape to be used for transformation.

        Returns
        -------
        Molecules
            Transformed molecules.
        """
        offset = self.offset
        shifts = ((indices - offset) * self.scale_factor).astype(np.float32)
        molecules_opt = molecules.translate_internal(shifts)
        if self.alignment_model.has_rotation:
            nrotation = len(self.alignment_model.quaternions)
            quats = np.stack(
                [
                    self.alignment_model.quaternions[
                        self.argmax[i, iz, iy, ix] % nrotation
                    ]
                    for i, (iz, iy, ix) in enumerate(indices)
                ],
                axis=0,
            )
            molecules_opt = molecules_opt.rotate_by_quaternion(quats)

            rotvec = Rotation.from_quat(quats).as_rotvec().astype(np.float32)
            molecules_opt = molecules_opt.with_features(
                pl.Series("align-dzrot", rotvec[:, 0]),
                pl.Series("align-dyrot", rotvec[:, 1]),
                pl.Series("align-dxrot", rotvec[:, 2]),
            )

        opt_score = np.fromiter(
            (
                -self.energy_array[i, iz, iy, ix]
                for i, (iz, iy, ix) in enumerate(indices)
            ),
            dtype=np.float32,
        )
        return molecules_opt.with_features(
            pl.Series("align-dz", shifts[:, 0]),
            pl.Series("align-dy", shifts[:, 1]),
            pl.Series("align-dx", shifts[:, 2]),
            pl.Series(Mole.score, opt_score),
        )

    def run_viterbi(self, dist_range: tuple[nm, nm], angle_max: float | None = None):
        """Run Viterbi alignment."""
        from cylindra._cylindra_ext import ViterbiGrid

        mole = self.molecules.translate_internal(-self.offset_nm)
        origin = (mole.pos / self.scale_factor).astype(np.float32)
        zvec = mole.z.astype(np.float32)
        yvec = mole.y.astype(np.float32)
        xvec = mole.x.astype(np.float32)
        grid = ViterbiGrid(-self.energy_array, origin, zvec, yvec, xvec)
        result = grid.viterbi(*dist_range, angle_max)
        return ViterbiResult(result[0], result[1])

    def annealing_model(
        self,
        spl: CylSpline,
        distance_range_long: tuple[nm, nm],
        distance_range_lat: tuple[nm, nm],
        angle_max: float | None = None,
        time_const: float | None = None,
        temperature: float | None = None,
        cooling_rate: float | None = None,
        reject_limit: int | None = None,
    ) -> CylindricAnnealingModel:
        """Get an annealing model using the landscape."""
        from cylindra._cylindra_ext import CylindricAnnealingModel

        cyl = spl.cylinder_model()
        _nrise, _npf = cyl.nrise, cyl.shape[1]
        molecules = self.molecules
        mole = molecules.translate_internal(-self.offset_nm)
        if angle_max is None:
            angle_max = 90.0

        time_const, temperature, cooling_rate, reject_limit = self._normalize_args(
            time_const, temperature, cooling_rate, reject_limit
        )

        return (
            CylindricAnnealingModel()
            .construct_graph(
                indices=molecules.features.select([Mole.nth, Mole.pf])
                .to_numpy()
                .astype(np.int32),
                npf=_npf,
                nrise=_nrise,
            )
            .set_graph_coordinates(
                origin=mole.pos,
                zvec=mole.z.astype(np.float32) * self.scale_factor,
                yvec=mole.y.astype(np.float32) * self.scale_factor,
                xvec=mole.x.astype(np.float32) * self.scale_factor,
            )
            .set_energy_landscape(self.energy_array)
            .set_reservoir(
                temperature=temperature,
                time_constant=time_const,
            )
            .set_box_potential(
                *distance_range_long,
                *distance_range_lat,
                float(np.deg2rad(angle_max)),
                cooling_rate=cooling_rate,
            )
            .with_reject_limit(reject_limit)
        )

    def run_annealing(
        self,
        spl: CylSpline,
        distance_range_long: tuple[nm, nm],
        distance_range_lat: tuple[nm, nm],
        angle_max: float | None = None,
        time_const: float | None = None,
        temperature: float | None = None,
        cooling_rate: float | None = None,
        reject_limit: int | None = None,
        random_seeds: list[int] = [0],
    ) -> list[AnnealingResult]:
        """Run simulated mesh annealing."""
        from dask import array as da, delayed

        if angle_max is None:
            angle_max = 90.0
        random_seeds = _normalize_random_seeds(random_seeds)
        time_const, temperature, cooling_rate, reject_limit = self._normalize_args(
            time_const, temperature, cooling_rate, reject_limit
        )
        annealing = self.annealing_model(
            spl,
            distance_range_long=distance_range_long,
            distance_range_lat=distance_range_lat,
            angle_max=angle_max,
            time_const=time_const,
            temperature=temperature,
            cooling_rate=cooling_rate,
            reject_limit=reject_limit,
        )

        batch_size = _to_batch_size(time_const)

        @delayed
        def _run(seed: int) -> AnnealingResult:
            _model = annealing.with_seed(seed)
            _model.init_shift_random()
            energies = [_model.energy()]
            while (
                _model.temperature() > temperature * 1e-6
                and _model.optimization_state() == "not_converged"
            ):
                _model.simulate(batch_size)
                energies.append(_model.energy())

            return AnnealingResult(
                energies=np.array(energies),
                batch_size=batch_size,
                time_const=time_const,
                indices=_model.shifts(),
                niter=_model.iteration(),
                state=_model.optimization_state(),
            )

        tasks = [_run(s) for s in random_seeds]
        results: list[AnnealingResult] = da.compute(tasks)[0]
        return results

    def _normalize_args(self, time_const, temperature, cooling_rate, reject_limit):
        nmole = self.molecules.count()
        if time_const is None:
            time_const = nmole * np.product(self.energy_array.shape[1:])
        _energy_std = np.std(self.energy_array)
        if temperature is None:
            temperature = _energy_std * 2
        if cooling_rate is None:
            cooling_rate = _energy_std / time_const * 8
        if reject_limit is None:
            reject_limit = nmole * 100
        return time_const, temperature, cooling_rate, reject_limit


@dataclass
class AnnealingResult:
    """
    Dataclass for annealing results.

    Parameters
    ----------
    energies : np.ndarray
        History of energies of the annealing process.
    batch_size : int
        Batch size used in the annealing process.
    time_const : float
        Time constant used for cooling.
    indices : np.ndarray
        The optimized indices of the molecules.
    niter : int
        Number of iterations.
    state : str
        Optimization state.
    """

    energies: NDArray[np.float32]
    batch_size: int
    time_const: float
    indices: NDArray[np.int32]
    niter: int
    state: str

    def with_debug_info(self, **kwargs) -> AnnealingResult:
        # insider use only
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


def _to_batch_size(time_const: float) -> int:
    return max(int(time_const / 20), 1)


def _normalize_random_seeds(seeds) -> list[int]:
    if isinstance(seeds, SupportsInt):  # noqa
        return [int(seeds)]
    out = list[int]()
    for i, seed in enumerate(seeds):
        if not isinstance(seed, SupportsInt):  # noqa
            raise TypeError(f"{i}-th seed {seed!r} is not an integer.")
        out.append(int(seed))
    if len(out) == 0:
        raise ValueError("No random seed is given.")
    return out


@dataclass
class ViterbiResult:
    indices: NDArray[np.int32]
    score: float


def _calc_landscape(
    model: ParametrizedModel | TomographyInput,
    score_dsk: da.Array,
    multi_templates: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.int32] | None]:
    from dask import array as da

    if not model.has_rotation:
        score = score_dsk.compute()
        if multi_templates:
            score = np.max(score, axis=1)
        argmax = None
    else:
        tasks = da.max(score_dsk, axis=1)
        argmax = da.argmax(score_dsk, axis=1)
        # NOTE: argmax.shape[0] == n_templates * len(model.quaternion)
        score, argmax = da.compute(tasks, argmax)
    return score, argmax
