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
    assert_allclose(np.cross(mol.y, mol.x, axis=1), mol.z)


def test_euler_const():
    for name in EulerAxes._member_names_:
        assert name == getattr(EulerAxes, name)


def test_euler():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1/Sq2, 1/Sq2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("ZYX", degrees=True), [[45, 0, 0]])
    
    pos = np.array([0, 0, 0])
    zvec = np.array([0, 0, -1])
    yvec = np.array([0, 1, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("zyz", degrees=True), [[0, 90, 0]])
    
    pos = np.array([0, 0, 0])
    zvec = np.array([1/Sq2, 1/Sq2, 0])
    yvec = np.array([-1/Sq2, 1/Sq2, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("zyx", degrees=True), [[0, 0, 45]])


def test_rotvec():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1/Sq2, 1/Sq2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.rotvec(), [[np.pi/4, 0, 0]])
    
    pos = np.array([0, 0, 0])
    zvec = np.array([0, 0, -1])
    yvec = np.array([0, 1, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.rotvec(), [[0, np.pi/2, 0]])


def test_save_and_load_euler_angle():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0.4, 0.1])
    yvec = np.array([0, 1.1, 2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    euler = mol.euler_angle(degrees=True)
    mol2 = Molecules.from_euler(pos, euler, degrees=True)
    assert_allclose(mol2.x, mol.x, rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.y, mol.y, rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.z, mol.z, rtol=1e-8, atol=1e-8)

def test_rotate():
    from scipy.spatial.transform import Rotation
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0.4, 0.1])
    yvec = np.array([0, 1.1, 2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    
    rot = Rotation.from_rotvec([0.1, 0.3, -0.2])
    mol2 = mol.rotate_by(rot)
    
    assert_allclose(rot.apply(mol.z), mol2.z, rtol=1e-8, atol=1e-8)
    assert_allclose(rot.apply(mol.y), mol2.y, rtol=1e-8, atol=1e-8)
    assert_allclose(rot.apply(mol.x), mol2.x, rtol=1e-8, atol=1e-8)
