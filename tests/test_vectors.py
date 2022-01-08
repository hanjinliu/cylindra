from mtprops.vector import VectorField3D
import numpy as np
from numpy.testing import assert_allclose

def test_rotation_matrix():
    vec = np.array([[1, 2, 2], [0, -1, -1]], dtype=np.float32)
    vec /= np.sqrt(np.sum(vec**2, axis=1))[:, np.newaxis]
    vf = VectorField3D(np.zeros((2, 3)), vec)
    
    mat = vf.rot_matrix("z")
    assert mat.shape == (2, 3, 3)
    assert_allclose(np.array([1, 0, 0]) @ mat[0], vf[0], rtol=1e-6, atol=1e-6)
    assert_allclose(np.array([1, 0, 0]) @ mat[1], vf[1], rtol=1e-6, atol=1e-6)
    
    mat = vf.rot_matrix("y")
    assert_allclose(np.array([0, 1, 0]) @ mat[0], vf[0], rtol=1e-6, atol=1e-6)
    assert_allclose(np.array([0, 1, 0]) @ mat[1], vf[1], rtol=1e-6, atol=1e-6)
    
    mat = vf.rot_matrix("x")
    assert_allclose(np.array([0, 0, 1]) @ mat[0], vf[0], rtol=1e-6, atol=1e-6)
    assert_allclose(np.array([0, 0, 1]) @ mat[1], vf[1], rtol=1e-6, atol=1e-6)
    
    src = np.array([1, 2, -1], dtype=np.float32)
    src /= np.sqrt(np.sum(src**2))
    mat = vf.rot_matrix(src)
    assert_allclose(src @ mat[0], vf[0], rtol=1e-6, atol=1e-6)
    assert_allclose(src @ mat[1], vf[1], rtol=1e-6, atol=1e-6)
    
    
    
    