import numpy as np
import pytest

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
        .set_reservoir(temperature=1.0, time_constant=1.0, min_temperature=0.0)
        .set_box_potential(2.0, 6.0, ang_max=0.3, cooling_rate=0.005)
    )
    model.init_shift_random()
    return model


def _lattice_neighbors(state: list[int], shape: tuple[int, int, int]) -> list[list[int]]:
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
