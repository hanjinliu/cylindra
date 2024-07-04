from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Sequence, SupportsIndex, SupportsInt

import impy as ip
import numpy as np
import polars as pl
from acryo import Molecules, alignment, pipe
from magicclass.logging import getLogger
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes

from cylindra._dask import Delayed, compute, delayed
from cylindra.components._peak import find_peak
from cylindra.components.spline import CylSpline
from cylindra.const import MoleculesHeader as Mole
from cylindra.const import nm

if TYPE_CHECKING:
    from acryo.alignment._base import ParametrizedModel, TomographyInput
    from acryo.loader._base import LoaderBase, MaskInputType, TemplateInputType
    from dask import array as da

    from cylindra._cylindra_ext import CylindricAnnealingModel

    _DistLike = nm | str

_Logger = getLogger("cylindra")


@dataclass
class Landscape:
    """
    Energy landscape array.

    Parameters
    ----------
    energies : NDArray[np.float32]
        4D array of energy values.
    molecules : Molecules
        Molecules object.
    argmax : NDArray[np.int32], optional
        Argmax indices to track which rotation resulted in the best alignment.
    quaternions : NDArray[np.float32]
        Quaternions used for template rotation.
    scale_factor : float
        Scale factor to convert from pixels to nanometers (upsampling considered).
        ``scale / upsample_factor`` will be passed to this parameter from the GUI.
    """

    energies: NDArray[np.float32]
    molecules: Molecules
    argmax: NDArray[np.int32] | None
    quaternions: NDArray[np.float32]
    scale_factor: float

    def __getitem__(
        self, key: slice | list[SupportsIndex] | NDArray[np.integer]
    ) -> Landscape:
        """Subset of the landscape."""
        if not isinstance(key, (slice, list, np.ndarray)):
            raise TypeError(f"Invalid type of key: {type(key)}")
        energy = self.energies[key]
        mole = self.molecules.subset(key)
        argmax = self.argmax[key] if self.argmax is not None else None
        return Landscape(energy, mole, argmax, self.quaternions, self.scale_factor)

    def __repr__(self) -> str:
        eng_repr = f"<{self.energies.shape!r} array>"
        mole_repr = f"<{self.molecules.count()} molecules>"
        argmax_repr = (
            f"<{self.argmax.shape!r} array>" if self.argmax is not None else None
        )
        return (
            f"Landscape(energies={eng_repr}, molecules={mole_repr}, "
            f"argmax={argmax_repr}, quaternion={self.quaternions!r}, "
            f"scale_factor={self.scale_factor:.3g})"
        )

    @property
    def offset(self) -> NDArray[np.int32]:
        """Shift from the corner (0, 0, 0) to the center."""
        shift = (np.array(self.energies.shape[1:], dtype=np.float32) - 1) / 2
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
        alignment_model: alignment.TomographyInput = alignment.ZNCCAlignment,
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
        mole = loader.molecules
        to_drop = set(mole.features.columns) - {Mole.nth, Mole.pf, Mole.position}
        if to_drop:
            mole = mole.drop_features(*to_drop)
        return cls(
            energies=-np.ascontiguousarray(score),
            molecules=mole,
            argmax=argmax,
            quaternions=alignment_model.quaternions,
            scale_factor=loader.scale / upsample_factor,
        )

    def transform_molecules(
        self,
        molecules: Molecules,
        indices: NDArray[np.int32],
        detect_peak: bool = False,
    ) -> Molecules:
        """
        Transform the input molecules based on the landscape.

        Parameters
        ----------
        molecules : Molecules
            Molecules object to be transformed.
        indices : integer array
            Indices in the landscape to be used for transformation.
        detect_peak : bool, default False
            If True, landscape will be sub-sampled to detect the peak in higher
            precision. This should be false for constrained alignment, as the detected
            peak usually does not represent the optimal result.

        Returns
        -------
        Molecules
            Transformed molecules.
        """
        offset = self.offset
        indices_sub = indices.astype(np.float32)
        nmole = self.energies.shape[0]
        opt_energy = np.zeros(nmole, dtype=np.float32)
        nrepeat = 3 if detect_peak else 0
        for i in range(nmole):
            eng = self.energies[i]
            indices_sub[i], opt_energy[i] = find_peak(eng, indices[i], nrepeat=nrepeat)
        opt_score = -opt_energy
        shifts = ((indices_sub - offset) * self.scale_factor).astype(np.float32)
        molecules_opt = molecules.translate_internal(shifts)
        shift_feat = _as_n_series("align-d{}", shifts)
        if (nrots := self.quaternions.shape[0]) > 1:
            quats = np.stack(
                [
                    self.quaternions[self.argmax[(i, *ind)] % nrots]
                    for i, ind in enumerate(indices)
                ],
                axis=0,
            )
            rotator = Rotation.from_quat(quats).inv()
            molecules_opt = molecules_opt.rotate_by(rotator)
            rotvec = rotator.as_rotvec().astype(np.float32)
            rotvec_feat = _as_n_series("align-d{}rot", rotvec)
            molecules_opt = molecules_opt.with_features(*rotvec_feat)

        return molecules_opt.with_features(
            *shift_feat, pl.Series(Mole.score, opt_score)
        )

    def run_min_energy(
        self, spl: CylSpline | None = None
    ) -> tuple[Molecules, MinEnergyResult]:
        """Minimize energy for each local landscape independently."""
        shape = self.energies.shape[1:]
        indices = list[NDArray[np.int32]]()
        engs = list[float]()
        for i in range(self.energies.shape[0]):
            eng = self.energies[i]
            pos = np.unravel_index(np.argmin(eng), shape)
            indices.append(np.array(pos, dtype=np.int32))
            engs.append(eng[pos])
        indices = np.stack(indices, axis=0)
        engs = np.array(engs, dtype=np.float32)
        result = MinEnergyResult(indices, engs)

        mole_opt = self.transform_molecules(
            self.molecules, result.indices, detect_peak=True
        )
        if spl is not None:
            mole_opt = _update_mole_pos(mole_opt, self.molecules, spl)
        return mole_opt, result

    def _norm_dist(self, dist_range: tuple[_DistLike, _DistLike]) -> tuple[nm, nm]:
        dist_min, dist_max = dist_range
        # normalize distance limits
        dist_mean = np.mean(
            np.sqrt(np.sum(np.diff(self.molecules.pos, axis=0) ** 2, axis=1))
        )
        dist_min = _norm_distance(dist_min, dist_mean)
        dist_max = _norm_distance(dist_max, dist_mean)
        return dist_min / self.scale_factor, dist_max / self.scale_factor

    def _prep_viterbi_grid(self):
        from cylindra._cylindra_ext import ViterbiGrid

        mole = self.molecules.translate_internal(-self.offset_nm)
        origin = (mole.pos / self.scale_factor).astype(np.float32)
        zvec = mole.z.astype(np.float32)
        yvec = mole.y.astype(np.float32)
        xvec = mole.x.astype(np.float32)
        return ViterbiGrid(-self.energies, origin, zvec, yvec, xvec)

    def run_viterbi(
        self, dist_range: tuple[_DistLike, _DistLike], angle_max: float | None = None
    ):
        """Run Viterbi alignment."""
        dist_min, dist_max = self._norm_dist(dist_range)
        if angle_max is not None:
            angle_max = np.deg2rad(angle_max)
        grid = self._prep_viterbi_grid()
        return ViterbiResult(*grid.viterbi(dist_min, dist_max, angle_max))

    def run_viterbi_along_spline(
        self,
        spl: CylSpline | None,
        range_long: tuple[nm, nm] = (4.0, 4.28),
        angle_max: float | None = 5.0,
    ):
        mole = self.molecules
        if Mole.pf in mole.features.columns:
            npfs: Sequence[int] = mole.features[Mole.pf].unique(maintain_order=True)
            slices = [(mole.features[Mole.pf] == i).to_numpy() for i in npfs]
            viterbi_tasks = [
                delayed(self[sl].run_viterbi)(range_long, angle_max) for sl in slices
            ]
        else:
            slices = [slice(None)]
            viterbi_tasks = [delayed(self.run_viterbi)(range_long, angle_max)]
        _Logger.info("Running Viterbi alignment")
        _Logger.info(f"  shape: {self.energies.shape[1:]!r}")
        _Logger.info(f"  tasks: {len(viterbi_tasks)}")
        vit_out = compute(*viterbi_tasks)

        inds = np.empty((mole.count(), 3), dtype=np.int32)
        max_shifts_px = (np.array(self.energies.shape[1:]) - 1) // 2
        for i, result in enumerate(vit_out):
            inds[slices[i], :] = _check_viterbi_shift(result.indices, max_shifts_px, i)
        molecules_opt = self.transform_molecules(mole, inds)
        if spl is not None:
            molecules_opt = _update_mole_pos(molecules_opt, mole, spl)
        return molecules_opt

    def run_viterbi_fixed_start(
        self,
        first: NDArray[np.float32],
        range_long: tuple[nm, nm] = (4.0, 4.28),
        angle_max: float | None = 5.0,
    ):
        """Run Viterbi alignment with a fixed start edge."""
        mole = self.molecules
        dist_min, dist_max = self._norm_dist(range_long)
        if angle_max is not None:
            angle_max = np.deg2rad(angle_max)
        grid = self._prep_viterbi_grid()
        mole0 = Molecules(first, mole.rotator[:1]).translate_internal(-self.offset_nm)
        origin = (mole0.pos[0] / self.scale_factor).astype(np.float32)
        res = ViterbiResult(
            *grid.viterbi_fixed_start(
                dist_min, dist_max, first / self.scale_factor, origin, angle_max
            )
        )
        max_shifts_px = (np.array(self.energies.shape[1:]) - 1) // 2
        inds = _check_viterbi_shift(res.indices, max_shifts_px, 0)
        molecules_opt = self.transform_molecules(mole, inds)
        return molecules_opt

    def annealing_model(
        self,
        spl: CylSpline,
        distance_range_long: tuple[_DistLike, _DistLike],
        distance_range_lat: tuple[_DistLike, _DistLike],
        angle_max: float | None = None,
        temperature_time_const: float = 1.0,
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
            temperature_time_const, temperature, cooling_rate, reject_limit
        )
        ind = molecules.features.select([Mole.nth, Mole.pf]).to_numpy().astype(np.int32)
        ind[:, 0] -= ind[:, 0].min()
        ind[:, 1] -= ind[:, 1].min()
        model = CylindricAnnealingModel().construct_graph(
            indices=ind, npf=_npf, nrise=_nrise
        )

        return (
            model.set_graph_coordinates(
                origin=mole.pos,
                zvec=(mole.z * self.scale_factor).astype(np.float32),
                yvec=(mole.y * self.scale_factor).astype(np.float32),
                xvec=(mole.x * self.scale_factor).astype(np.float32),
            )
            .set_energy_landscape(self.energies)
            .set_reservoir(
                temperature=temperature,
                time_constant=time_const,
            )
            .set_box_potential(
                *self._norm_distance_range_long(distance_range_long, model),
                *self._norm_distance_range_lat(distance_range_lat, model),
                float(np.deg2rad(angle_max)),
                cooling_rate=cooling_rate,
            )
            .with_reject_limit(reject_limit)
        )

    def run_annealing(
        self,
        spl: CylSpline,
        distance_range_long: tuple[_DistLike, _DistLike],
        distance_range_lat: tuple[_DistLike, _DistLike],
        angle_max: float | None = None,
        temperature_time_const: float = 1.0,
        temperature: float | None = None,
        cooling_rate: float | None = None,
        reject_limit: int | None = None,
        random_seeds: list[int] = [0],
    ) -> list[AnnealingResult]:
        """Run simulated mesh annealing."""

        if angle_max is None:
            angle_max = 90.0
        random_seeds = _normalize_random_seeds(random_seeds)
        annealing = self.annealing_model(
            spl,
            distance_range_long=distance_range_long,
            distance_range_lat=distance_range_lat,
            angle_max=angle_max,
            temperature_time_const=temperature_time_const,
            temperature=temperature,
            cooling_rate=cooling_rate,
            reject_limit=reject_limit,
        )

        batch_size = _to_batch_size(annealing.time_constant())
        temp0 = annealing.temperature()
        _Logger.info("Running annealing")
        _Logger.info(f"  shape: {self.energies.shape[1:]!r}")
        tasks = [
            _run_annealing(annealing.with_seed(s), batch_size, temp0)
            for s in random_seeds
        ]
        return compute(*tasks)

    def run_annealing_along_spline(
        landscape: Landscape,
        spl: CylSpline,
        range_long: tuple[float, float],
        range_lat: tuple[float | str, float | str],
        angle_max: float,
        temperature_time_const: float = 1.0,
        random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    ):
        mole = landscape.molecules
        results = landscape.run_annealing(
            spl,
            range_long,
            range_lat,
            angle_max,
            temperature_time_const=temperature_time_const,
            random_seeds=random_seeds,
        )
        if all(result.state == "failed" for result in results):
            raise RuntimeError(
                "Failed to optimize for all trials. You may check the distance range."
            )
        elif not any(result.state == "converged" for result in results):
            _Logger.print("Optimization did not converge for any trial.")

        _Logger.print_table(
            {
                "Iteration": [r.niter for r in results],
                "Score": [-float(r.energies[-1]) for r in results],
                "State": [r.state for r in results],
            }
        )
        results = sorted(results, key=lambda r: r.energies[-1])
        mole_opt = landscape.transform_molecules(mole, results[0].indices)
        mole_opt = _update_mole_pos(mole_opt, mole, spl)
        return mole_opt, results

    def _normalize_args(
        self, temperature_time_const, temperature, cooling_rate, reject_limit
    ):
        nmole = self.molecules.count()
        time_const = nmole * np.prod(self.energies.shape[1:]) * temperature_time_const
        _energy_std = np.std(self.energies)
        if temperature is None:
            temperature = _energy_std * 2
        if cooling_rate is None:
            cooling_rate = _energy_std / time_const * 8
        if reject_limit is None:
            reject_limit = nmole * 50
        return time_const, temperature, cooling_rate, reject_limit

    def normed(self, sd: bool = True) -> Landscape:
        """Return a landscape with normalized mean energy."""
        each_mean: NDArray[np.float32] = self.energies.mean(axis=(1, 2, 3))
        all_mean = each_mean.mean()
        sl = (slice(None), np.newaxis, np.newaxis, np.newaxis)
        if sd:
            each_sd: NDArray[np.float32] = self.energies.std(axis=(1, 2, 3))
            all_sd = each_sd.std()
            dif = self.energies - each_mean[sl]
            new_array = dif * all_sd / each_sd[sl] + all_mean
        else:
            new_array = self.energies + all_mean - each_mean[sl]
        return Landscape(
            new_array,
            self.molecules,
            self.argmax,
            self.quaternions,
            self.scale_factor,
        )

    def create_surface(
        self,
        level: float | None = None,
        resolution: nm = 0.25,
        show_min: bool = True,
    ) -> SurfaceData:
        """Create a isosurface data from the landscape"""
        if level is None:
            level = self.energies.mean()
        if show_min:
            intensity = -self.energies
            level = -level
        else:
            intensity = self.energies

        step_size = max(int(resolution / self.scale_factor), 1)
        spacing = (self.scale_factor,) * 3
        center = np.array(intensity.shape[1:]) / 2 + 0.5
        offset = center * spacing
        n_verts = 0
        tasks = list[Delayed[SurfaceData]]()
        for i in range(intensity.shape[0]):
            arr: NDArray[np.float32] = intensity[i]
            tasks.append(delayed_isosurface(arr, level, spacing, step_size=step_size))
        surfs = compute(*tasks)
        for i in range(intensity.shape[0]):
            mole = self.molecules[i]
            surf = surfs[i]
            surf = SurfaceData(
                mole.rotator.apply(surf.vertices - offset) + mole.pos,
                surf.faces + n_verts,
                surf.values,
            )
            surfs[i] = surf  # update
            n_verts += len(surf.vertices)
        vertices = np.concatenate([s.vertices for s in surfs], axis=0)
        faces = np.concatenate([s.faces for s in surfs], axis=0)
        values = np.concatenate([s.values for s in surfs], axis=0)
        return SurfaceData(vertices, faces, values)

    @classmethod
    def from_dir(cls, path: str | Path) -> Landscape:
        """Load a landscape from a directory."""
        path = Path(path)
        if path.suffix != "":
            raise ValueError(f"Must be a directory, got {path}")
        energies = ip.imread(path / "landscape.tif")
        molecules = Molecules.from_parquet(path / "molecules.parquet")
        argmax = None
        if (fp := path / "argmax.parquet").exists():
            argmax = pl.read_parquet(fp).to_series().to_numpy()
        quaternions = np.loadtxt(
            path / "quaternions.txt", delimiter=",", dtype=np.float32
        )
        scale_factor = energies.scale["x"]
        return cls(energies.value, molecules, argmax, quaternions, scale_factor)

    def save(self, path: str | Path) -> None:
        """Save the landscape to a directory."""
        path = Path(path)
        if path.suffix != "":
            raise ValueError(f"Must be a directory, got {path}")
        path.mkdir(exist_ok=False)
        arr = ip.asarray(self.energies, axes="tzyx").set_scale(
            xyz=self.scale_factor, unit="nm"
        )
        arr.imsave(path / "landscape.tif")
        self.molecules.to_parquet(path / "molecules.parquet")
        if self.argmax is not None:
            pl.DataFrame({"argmax": self.argmax}).write_parquet(
                path / "argmax.parquet", compression_level=10
            )
        self.quaternions.tofile(path / "quaternions.txt", sep=",")
        return None

    def _norm_distance_range_long(
        self,
        rng: tuple[nm | str, nm | str],
        model: CylindricAnnealingModel,
    ) -> tuple[nm, nm]:
        rng0, rng1 = rng
        if isinstance(rng0, str) or isinstance(rng1, str):
            long_dist = model.longitudinal_distances().mean()
            if isinstance(rng0, str):
                rng0 = _norm_distance(rng0, long_dist)
            if isinstance(rng1, str):
                rng1 = _norm_distance(rng1, long_dist)
        if not rng0 < rng1:
            raise ValueError(f"Lower is larger than the upper: {(rng0, rng1)}")
        return rng0, rng1

    def _norm_distance_range_lat(
        self,
        rng: tuple[nm | str, nm | str],
        model: CylindricAnnealingModel,
    ) -> tuple[nm, nm]:
        rng0, rng1 = rng
        if isinstance(rng0, str) or isinstance(rng1, str):
            lat_dist = model.lateral_distances().mean()
            if isinstance(rng0, str):
                rng0 = _norm_distance(rng0, lat_dist)
            if isinstance(rng1, str):
                rng1 = _norm_distance(rng1, lat_dist)
        if not rng0 < rng1:
            raise ValueError(f"Lower is larger than the upper: {(rng0, rng1)}")
        return rng0, rng1


def _norm_distance(v: str | nm, ref: nm) -> nm:
    if not isinstance(v, str):
        return v
    if v.startswith(("x", "*")):
        ntimes = float(v[1:])
        if ntimes < 0.0:
            raise ValueError(f"Invalid bound: {v}")
        v = float(v[1:]) * ref
    elif v.startswith("-"):
        v = ref - float(v[1:])
    elif v.startswith("+"):
        v = ref + float(v[1:])
    else:
        raise ValueError(f"Invalid bound: {v}")
    return v


def _check_viterbi_shift(shift: NDArray[np.int32], offset: NDArray[np.int32], i):
    invalid = shift[:, 0] < 0
    if invalid.any():
        invalid_indices = np.where(invalid)[0]
        _Logger.print(
            f"Viterbi alignment could not determine the optimal positions for PF={i!r}"
        )
        for idx in invalid_indices:
            shift[idx, :] = offset

    return shift


def _update_mole_pos(new: Molecules, old: Molecules, spl: CylSpline) -> Molecules:
    """
    Update the "position-nm" feature of molecules.

    Feature "position-nm" is the coordinate of molecules along the source spline.
    After alignment, this feature should be updated accordingly. This fucntion
    will do this.
    """
    if Mole.position not in old.features.columns:
        return new
    _u = spl.y_to_position(old.features[Mole.position])
    vec = spl.map(_u, der=1)  # the tangent vector of the spline
    vec_norm = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    dy = np.sum((new.pos - old.pos) * vec_norm, axis=1)
    return new.with_features(pl.col(Mole.position) + dy)


@delayed
def _run_annealing(
    model: CylindricAnnealingModel,
    batch_size: int,
    temp: float,
) -> AnnealingResult:
    model.init_shift_random()
    energies = [model.energy()]
    while (
        model.temperature() > temp * 1e-5
        and model.optimization_state() == "not_converged"
    ):
        model.simulate(batch_size)
        energies.append(model.energy())
    model.cool_completely()
    energies.append(model.energy())
    return AnnealingResult(
        energies=np.array(energies),
        batch_size=batch_size,
        time_const=model.time_constant(),
        indices=model.shifts(),
        niter=model.iteration(),
        state=model.optimization_state(),
    )


class SurfaceData(NamedTuple):
    """Tuple for storing isosurface data for landscapes."""

    vertices: NDArray[np.float32]
    faces: NDArray[np.int32]
    values: NDArray[np.float32]


@delayed
def delayed_isosurface(
    arr: NDArray[np.float32],
    level: float,
    spacing: tuple[float, float, float],
    step_size: int = 1,
) -> SurfaceData:
    """Create an isosurface from a 3D array using marching cubes algorithm."""
    arr_pad = np.pad(arr, step_size, mode="constant", constant_values=arr.min())
    try:
        verts, faces, _, vals = marching_cubes(
            arr_pad,
            level,
            spacing=spacing,
            gradient_direction="descent",
            step_size=step_size,
        )
        verts -= np.array(spacing)[np.newaxis] * step_size
    except (RuntimeError, ValueError):
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)
        vals = np.zeros((0,), dtype=np.float32)
    return SurfaceData(verts, faces, vals)


@dataclass
class MinEnergyResult:
    indices: NDArray[np.int32]
    energies: NDArray[np.float32]


@dataclass
class AnnealingResult:
    """
    Dataclass for storing the annealing results.

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
    """
    Dataclass for storing the Viterbi alignment results.

    Parameters
    ----------
    indices : np.ndarray
        The optimized indices of the molecules.
    score : float
        The score of the optimal alignment.
    """

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


def _as_n_series(fmt: str, arr: NDArray[np.floating]) -> list[pl.Series]:
    return [
        pl.Series(fmt.format("z"), arr[:, 0]),
        pl.Series(fmt.format("y"), arr[:, 1]),
        pl.Series(fmt.format("x"), arr[:, 2]),
    ]
