import numpy as np
from numpy.testing import assert_allclose
from cylindra.components import CylSpline, CylinderModel

def test_viterbi_1d():
    from cylindra._cpp_ext import ViterbiGrid
    
    # there is only one position with score 1.0 for each landscape
    score = np.zeros((10, 5, 5, 5))
    for i in [0, 1, 2, 3, 5, 6, 8, 9]:
        score[i, 0, 0, 0] = 1.0
    score[4, 1, 2, 1] = 1.0
    score[7, 4, 4, 4] = 1.0
    zvec = np.array([[1., 0., 0.]]*10)
    yvec = np.array([[0., 1., 0.]]*10)
    xvec = np.array([[0., 0., 1.]]*10)
    origin = np.array([[i*5, i*5, i*5] for i in range(10)])

    grid = ViterbiGrid(score, origin, zvec, yvec, xvec)
    states, z = grid.viterbi(0., 10000.)
    assert_allclose(
        states, 
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1],
                  [0, 0, 0], [0, 0, 0], [4, 4, 4], [0, 0, 0], [0, 0, 0]])
    )
    assert z == 10.0
    
    states, z = grid.viterbi(2*np.sqrt(3), 10000)
    assert_allclose(
        states, 
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1],
                  [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert z == 9.0

    states, z = grid.viterbi(0., 7*np.sqrt(3))
    assert_allclose(
        states, 
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 1],
                  [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert z == 9.0

def test_viterbi_2d():
    from cylindra._cpp_ext import ViterbiGrid2D

    score = np.zeros((4, 3, 5, 5, 5))
    spl = CylSpline.line([0, 0, 0], [3, 3, 3])
    model = CylinderModel((4, 3), (0, 0.2), radius=0.58)
    mole = model.to_molecules(spl)
    zvec = mole.z.reshape(4, 3, -1)
    yvec = mole.y.reshape(4, 3, -1)
    xvec = mole.x.reshape(4, 3, -1)
    origin = mole.pos.reshape(4, 3, -1)

    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]:
        score[i, j, 0, 0, 0] = 1.0
    score[0, 2, 1, 2, 1] = 1.0
    score[1, 2, 4, 4, 4] = 1.0
    score[2, 2, 2, 2, 2] = 1.0
    score[3, 2, 1, 1, 1] = 1.0

    grid = ViterbiGrid2D(score, origin, zvec, yvec, xvec, 2)
    states, z = grid.viterbi(0.8, 2, 0.8, 2)
