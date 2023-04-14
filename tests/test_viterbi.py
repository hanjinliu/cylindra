import numpy as np
from numpy.testing import assert_equal, assert_array_less
import pytest
from cylindra.components import CylSpline, CylinderModel

@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_viterbi_1d(scale):
    from cylindra._cpp_ext import ViterbiGrid
    
    # there is only one position with score 1.0 for each landscape
    score = np.zeros((10, 5, 5, 5))
    for i in [0, 1, 2, 3, 5, 6, 8, 9]:
        score[i, 0, 0, 0] = 1.0
    score[4, 1, 2, 1] = 1.0
    score[7, 4, 4, 4] = 1.0
    zvec = np.array([[1., 0., 0.]]*10) * scale
    yvec = np.array([[0., 1., 0.]]*10) * scale
    xvec = np.array([[0., 0., 1.]]*10) * scale
    origin = np.array([[i*5, i*5, i*5] for i in range(10)]) * scale

    grid = ViterbiGrid(score, origin, zvec, yvec, xvec)
    states, z = grid.viterbi(0., 10000. * scale)
    assert_equal(
        states, 
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1],
                  [0, 0, 0], [0, 0, 0], [4, 4, 4], [0, 0, 0], [0, 0, 0]])
    )
    assert z == 10.0
    
    states, z = grid.viterbi(2*np.sqrt(3) * scale, 10000 * scale)
    assert_equal(
        states, 
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1],
                  [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert z == 9.0

    states, z = grid.viterbi(0., 7*np.sqrt(3) * scale)
    assert_equal(
        states, 
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1],
                  [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert z == 9.0

@pytest.mark.parametrize("seed", [1, 12, 1234, 12345, 9999])
def test_viterbi_1d_distance(seed: int):
    from cylindra._cpp_ext import ViterbiGrid
    
    n = 4
    dist_min, dist_max = 2*np.sqrt(3), 7*np.sqrt(3)
    rng = np.random.default_rng(seed)
    score = rng.random((n, 5, 5, 5)).astype(np.float32)
    
    zvec = np.array([[1., 0., 0.]] * n)
    yvec = np.array([[0., 1., 0.]] * n)
    xvec = np.array([[0., 0., 1.]] * n)
    origin = np.array([[i*5, i*5, i*5] for i in range(n)])
    
    grid = ViterbiGrid(score, origin, zvec, yvec, xvec)
    states, z = grid.viterbi(dist_min, dist_max)
    
    dist = []
    for i in range(n - 1):
        pos1 = grid.world_pos(i + 1, *states[i + 1])
        pos0 = grid.world_pos(i, *states[i])
        dist.append(np.sqrt(np.sum((pos1 - pos0) ** 2)))
    dist = np.array(dist)

    assert_array_less(dist_min, dist, "dist_min < dist not satisfied")
    assert_array_less(dist, dist_max, "dist_max > dist not satisfied")

@pytest.mark.parametrize("nrise", [2, -2])
def test_viterbi_2d(nrise: int):
    from cylindra._cpp_ext import ViterbiGrid2D
    from timeit import default_timer

    score = np.zeros((4, 3, 5, 5, 5))
    spl = CylSpline.line([0, 0, 0], [3, 3, 3])
    model = CylinderModel((4, 3), (0, 0.2), radius=0.58)
    mole = model.to_molecules(spl)
    zvec = mole.z.reshape(4, 3, -1)
    yvec = mole.y.reshape(4, 3, -1)
    xvec = mole.x.reshape(4, 3, -1)
    origin = mole.pos.reshape(4, 3, -1)

    for i in range(4):
        for j in range(3):
            if i != 1:
                score[i, j, 1, 2, 1] = 1.0
            else:
                score[i, j, 4, 4, 4] = 1.0

    grid = ViterbiGrid2D(score, origin, zvec, yvec, xvec, nrise)
    t0 = default_timer()
    states, z = grid.viterbi(0.0, 10000.0, 0.0, 10000.0)
    msec = (default_timer() - t0) * 1000
    print(f"{msec:2f} msec, score={z}")

    answer = np.zeros((4, 3, 3))
    for i in range(4):
        for j in range(3):
            if i != 1:
                answer[i, j, :] = [1, 2, 1]
            else:
                answer[i, j, :] = [4, 4, 4]
    assert_equal(states, answer)
    assert_array_less(0, np.max(states, axis=(1, 2)))

@pytest.mark.parametrize("seed", [21, 32, 432, 9876, 1010])
@pytest.mark.parametrize(["ny", "npf"], [(5, 4), (20, 4), (5, 12)])
@pytest.mark.parametrize("nrise", [1, -1])
def test_viterbi_2d_distance(seed: int, ny: int, npf: int, nrise: int):
    from cylindra._cpp_ext import ViterbiGrid2D
    from timeit import default_timer

    radius = 20
    yspace = 10
    narr = np.arange(ny * npf).reshape(ny, npf)
    
    lat_dist_avg = np.hypot(np.sqrt(2) * radius, yspace / npf)
    dist_min, dist_max = yspace - 1.2, yspace + 1.2
    lat_dist_min, lat_dist_max = lat_dist_avg - 1.4, lat_dist_avg + 1.4

    rng = np.random.default_rng(seed + ny * npf)
    score = rng.random((ny, npf, 5, 5, 5)).astype(np.float32)

    if nrise >= 0:
        origin = np.stack(
            [radius * np.cos(np.pi / 2 * narr),
            yspace / npf * narr,
            radius * np.sin(np.pi / 2 * narr)],
            axis=-1,
        )  # shape (ny, npf, 3)
    else:
        origin = np.stack(
            [radius * np.cos(-np.pi / 2 * narr),
            -yspace / npf * (narr % npf) + yspace * (narr // npf),
            radius * np.sin(-np.pi / 2 * narr)],
            axis=-1,
        )  # shape (ny, npf, 3)
        score = score[:, ::-1]
    
    def _cross(x, y) -> np.ndarray:  # just for typing
        return -np.cross(x, y, axis=-1)
        
    zvec = np.stack([origin[:, :, 0], np.zeros((ny, npf)), origin[:, :, 2]], axis=-1) / radius
    yvec = np.array([[0., 1., 0.]] * ny * npf).reshape(ny, npf, 3)
    xvec = _cross(yvec, zvec)

    grid = ViterbiGrid2D(score, origin, zvec, yvec, xvec, nrise)
    t0 = default_timer()
    states, z = grid.viterbi(dist_min, dist_max, lat_dist_min, lat_dist_max)
    msec = (default_timer() - t0) * 1000
    print(f"{msec:2f} msec, score={z}")
    
    assert_array_less(-1, states)

    dist = grid.all_longitudinal_distances(states)
    dist_lat = grid.all_lateral_distances(states)

    assert_array_less(dist_min, dist, "dist_min < dist not satisfied")
    assert_array_less(dist, dist_max, "dist < dist_max not satisfied")
    assert_array_less(lat_dist_min, dist_lat, "lat_dist_min < dist_lat not satisfied")
    assert_array_less(dist_lat, lat_dist_max, "dist_lat < lat_dist_max not satisfied")

    assert_array_less(0, np.max(states, axis=(1, 2)))
