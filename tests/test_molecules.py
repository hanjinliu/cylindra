from mtprops.molecules import Molecules
from mtprops.const import EulerAxes
import numpy as np
from numpy.testing import assert_allclose
import pytest

Sq3 = np.sqrt(3)
Sq2 = np.sqrt(2)

values = [([1, 0, 0], 
           [0, 1/Sq2, 1/Sq2],
           [[1,     0,      0], 
            [0, 1/Sq2, -1/Sq2], 
            [0, 1/Sq2,  1/Sq2]]),
          ([0, 0, -1],
		    [0, 1, 0],
           [[ 0, 0, 1], 
            [ 0, 1, 0], 
            [-1, 0, 0]]),
          ([1, 0, 0],
		    [0, Sq3/2, 1/2],
           [[1,     0,     0], 
            [0, Sq3/2,  -1/2], 
            [0,   1/2, Sq3/2]]),
          ([  1/2, -Sq3/2, 0],
           [Sq3/2,    1/2, 0],
           [[   1/2, Sq3/2, 0], 
            [-Sq3/2,   1/2, 0], 
            [     0,     0, 1]])
          ]

@pytest.mark.parametrize("zvec, yvec, mat", values)
def test_matrix(zvec, yvec, mat):
    pos = np.array([0, 0, 0])
    zvec = np.array(zvec)
    yvec = np.array(yvec)
    mat = np.array(mat)[np.newaxis]
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.z[0], zvec)
    assert_allclose(mol.y[0], yvec)
    out = mol.matrix()
    assert_allclose(out, mat, rtol=1e-6, atol=1e-6)


def test_euler_const():
    for name in EulerAxes._member_names_:
        assert name == getattr(EulerAxes, name)


def test_euler():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("ZYX", degrees=True), [[45, 0, 0]])
    
    pos = np.array([0, 0, 0])
    zvec = np.array([0, 0, -1])
    yvec = np.array([0, 1, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("zyz", degrees=True), [[0, 90, 0]])


def test_rotvec():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.rot_vector(), [[np.pi/4, 0, 0]])
    
    pos = np.array([0, 0, 0])
    zvec = np.array([0, 0, -1])
    yvec = np.array([0, 1, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.rot_vector(), [[0, np.pi/2, 0]])

# TODO: test from_axes using x=...