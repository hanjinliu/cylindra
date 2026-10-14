import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cylindra._cylindra_ext import FilamentousAnnealingModel


def _build_model(
    seed: int,
    num: int = 6,
    local_shape: tuple[int, int, int] = (3, 5, 5),
    spacing: float = 4.0,
) -> FilamentousAnnealingModel:
    rng = np.random.default_rng(seed)
    nz, ny, nx = local_shape
    center = np.array([nz // 2, ny // 2, nx // 2], dtype=np.float32)

    # Nodes laid out along the world x-axis, `spacing` nm apart, so that each
    # node's "center" shift maps back onto the straight line.
    positions = np.zeros((num, 3), dtype=np.float32)  # columns: z, y, x
    positions[:, 2] = np.arange(num) * spacing
    origin = (positions - center).astype(np.float32)
    zvec = np.tile([1.0, 0.0, 0.0], (num, 1)).astype(np.float32)
    yvec = np.tile([0.0, 1.0, 0.0], (num, 1)).astype(np.float32)
    xvec = np.tile([0.0, 0.0, 1.0], (num, 1)).astype(np.float32)

    # Random internal energy landscape so that some nodes -- including the two
    # tip nodes -- are pulled off the straight line, creating tension with the
    # bending (deforming) energy of the triplet centered at their neighbor.
    energy = rng.uniform(0.0, 2.0, size=(num, nz, ny, nx)).astype(np.float32)

    model = (
        FilamentousAnnealingModel(seed=seed)
        .construct_graph(num)
        .set_graph_coordinates(origin=origin, zvec=zvec, yvec=yvec, xvec=xvec)
        .set_energy_landscape(energy)
        .set_reservoir(temperature=1.0, time_constant=1.0)
        .set_box_potential(2.0, 6.0, ang_max=0.3, cooling_rate=0.005)
    )
    model.init_shift_random()
    return model


def _lattice_neighbors(
    state: list[int], shape: tuple[int, int, int]
) -> list[list[int]]:
    """Single-step face-adjacent moves, mirroring `list_neighbors` (Rust)."""
    neighbors = []
    for axis in range(3):
        s = state[axis]
        upper = shape[axis] - 1
        if 0 < s < upper:
            deltas = (-1, 1)
        elif s == 0:
            deltas = (1,)
        else:
            deltas = (-1,)
        for d in deltas:
            nbr = list(state)
            nbr[axis] += d
            neighbors.append(nbr)
    return neighbors


def _best_improving_shift(model: FilamentousAnnealingModel) -> float:
    """Brute-force search for a single-node, single-step shift that reduces
    `model.energy()` below its current value. Returns the most negative energy
    difference found (0.0 if no node has any improving shift)."""
    shape = model.local_shape()
    base_shifts = model.shifts()
    base_energy = model.energy()
    best_diff = 0.0
    for idx in range(model.node_count()):
        for nbr in _lattice_neighbors(list(base_shifts[idx]), shape):
            trial = base_shifts.copy()
            trial[idx] = nbr
            model.set_shifts(trial)
            diff = model.energy() - base_energy
            if diff < best_diff:
                best_diff = diff
    model.set_shifts(base_shifts)
    return best_diff


@pytest.mark.parametrize("seed", range(10))
def test_filamentous_cool_completely_reaches_local_optimum(seed: int):
    # `energy_diff_by_shift` used to skip updating the bending (deforming)
    # energy when shifting a tip node (idx == 0 or idx == node_count() - 1),
    # even though shifting a tip node still changes the deforming-energy
    # triplet centered at its one neighbor. That made `cool_completely` (and
    # `simulate`, which relies on the same incremental energy calculation)
    # stop at a state that is not actually a local optimum: a single-step
    # shift of a tip node could still reduce the true energy.
    model = _build_model(seed)
    model.simulate(nsteps=4000)
    model.cool_completely()

    best_diff = _best_improving_shift(model)
    assert best_diff == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize(
    "local_shape", [(3, 5, 5), (7, 3, 3), (3, 7, 3), (3, 3, 7), (5, 3, 7)]
)
def test_filamentous_energy_non_cubic(local_shape: tuple[int, int, int]):
    num = 6
    dist_min, dist_max, ang_max, cooling_rate = 3.6, 4.4, 0.2, 0.01
    rng = np.random.default_rng(0)
    mtx = Rotation.random(num, random_state=0).as_matrix().astype(np.float32)
    zvec, yvec, xvec = (mtx[:, :, i] * 0.5 for i in range(3))
    origin = np.zeros((num, 3), dtype=np.float32)
    origin[:, 1] = np.arange(num) * 4.0
    energy = rng.uniform(0.0, 2.0, size=(num, *local_shape)).astype(np.float32)
    model = (
        FilamentousAnnealingModel(seed=0)
        .construct_graph(num)
        .set_graph_coordinates(origin=origin, zvec=zvec, yvec=yvec, xvec=xvec)
        .set_energy_landscape(energy)
        .set_reservoir(temperature=1.0, time_constant=1.0)
        .set_box_potential(dist_min, dist_max, ang_max, cooling_rate=cooling_rate)
    )
    model.simulate(50)  # potentials are effective only after cooling
    slope = cooling_rate * model.iteration()
    for _ in range(10):
        shifts = np.stack(
            [rng.integers(0, s, size=num) for s in local_shape], axis=1
        ).astype(model.shifts().dtype)
        model.set_shifts(shifts)
        # energy calculated from the world coordinates of the nodes
        pos = origin + sum(
            shifts[:, i : i + 1] * v for i, v in enumerate([zvec, yvec, xvec])
        )
        expected = energy[
            np.arange(num), shifts[:, 0], shifts[:, 1], shifts[:, 2]
        ].sum()
        dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        expected += slope * np.sum(
            np.clip(dist_min - dist, 0, None) + np.clip(dist - dist_max, 0, None)
        )
        for i in range(1, num - 1):
            dr1, dr2 = pos[i] - pos[i - 1], pos[i] - pos[i + 1]
            cos = abs(np.dot(dr1, dr2)) / np.linalg.norm(dr1) / np.linalg.norm(dr2)
            ang = np.arccos(min(cos, 1.0))
            expected += slope * max(ang - ang_max, 0.0)
        assert model.energy() == pytest.approx(expected, rel=1e-4, abs=1e-3)
