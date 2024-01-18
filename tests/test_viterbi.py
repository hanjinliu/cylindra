import numpy as np
import pytest
from numpy.testing import assert_array_less, assert_equal

from cylindra._cylindra_ext import ViterbiGrid


def _get_grid(scale: float):
    # there is only one position with score 1.0 for each landscape
    score = np.zeros((10, 5, 5, 5), dtype=np.float32)
    for i in [0, 1, 2, 3, 5, 6, 8, 9]:
        score[i, 0, 0, 0] = 1.0
    score[4, 1, 2, 1] = 1.0
    score[7, 4, 4, 4] = 1.0
    zvec = np.array([[1.0, 0.0, 0.0]] * 10, dtype=np.float32) * scale
    yvec = np.array([[0.0, 1.0, 0.0]] * 10, dtype=np.float32) * scale
    xvec = np.array([[0.0, 0.0, 1.0]] * 10, dtype=np.float32) * scale
    origin = (
        np.array([[i * 5, i * 5, i * 5] for i in range(10)], dtype=np.float32) * scale
    )

    return ViterbiGrid(score, origin, zvec, yvec, xvec)


@pytest.mark.parametrize("scale", [1e-4, 0.01, 1.0, 100.0, 1e4])
@pytest.mark.parametrize("angle_max", [None, 3.14 / 2])
def test_viterbi_1d(scale, angle_max):
    grid = _get_grid(scale)

    states, z = grid.viterbi(0.0, 10000.0 * scale, angle_max=angle_max)
    assert_equal(
        states,
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 2, 1],
                [0, 0, 0],
                [0, 0, 0],
                [4, 4, 4],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert z == 10.0


@pytest.mark.parametrize("scale", [1e-4, 0.01, 1.0, 100.0, 1e4])
@pytest.mark.parametrize("angle_max", [None, 3.14 / 2])
def test_viterbi_1d_lower_bound(scale, angle_max):
    grid = _get_grid(scale)
    states, z = grid.viterbi(2 * np.sqrt(3) * scale, 10000 * scale, angle_max=angle_max)
    assert_equal(
        states,
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 2, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert z == 9.0


@pytest.mark.parametrize("scale", [1e-4, 0.01, 1.0, 100.0, 1e4])
@pytest.mark.parametrize("angle_max", [None, 3.14 / 2])
def test_viterbi_1d_upper_bound(scale, angle_max):
    grid = _get_grid(scale)
    states, z = grid.viterbi(0.0, 7 * np.sqrt(3) * scale, angle_max=angle_max)
    assert_equal(
        states,
        np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 2, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert z == 9.0


@pytest.mark.parametrize("seed", [1, 12, 1234, 12345, 9999])
@pytest.mark.parametrize("angle_max", [None, 1.57, 1.047])  # None, pi/2, pi/3
def test_viterbi_1d_distance(seed: int, angle_max):
    n = 4
    dist_min, dist_max = 2 * np.sqrt(3), 7 * np.sqrt(3)
    rng = np.random.default_rng(seed)
    score = rng.random((n, 5, 5, 5)).astype(np.float32)

    zvec = np.array([[1.0, 0.0, 0.0]] * n, dtype=np.float32)
    yvec = np.array([[0.0, 1.0, 0.0]] * n, dtype=np.float32)
    xvec = np.array([[0.0, 0.0, 1.0]] * n, dtype=np.float32)
    origin = np.array([[i * 5, i * 5, i * 5] for i in range(n)], dtype=np.float32)

    grid = ViterbiGrid(score, origin, zvec, yvec, xvec)
    states, z = grid.viterbi(dist_min, dist_max, angle_max=angle_max)
    assert_array_less(-1, states, "states < -1 not satisfied")
    dist = []
    for i in range(n - 1):
        pos1 = grid.world_pos(i + 1, *states[i + 1])
        pos0 = grid.world_pos(i, *states[i])
        dist.append(np.sqrt(np.sum((pos1 - pos0) ** 2)))
    dist = np.array(dist)

    assert_array_less(dist_min, dist, "dist_min < dist not satisfied")
    assert_array_less(dist, dist_max, "dist_max > dist not satisfied")
