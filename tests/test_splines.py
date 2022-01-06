from mtprops.spline import Spline3D
import numpy as np
from numpy.testing import assert_allclose

spl = Spline3D()
coords = np.array([[0, 0, 0], [0, 1, 0],[0, 2, 0],[0, 3, 0]])
spl.fit(coords)

def test_inverse_mapping():
    coords = np.array([[1, 1.5, 0], 
                       [0, 1.5, 1],
                       [-1, 1.5, 0], 
                       [2, 1.5, 3]])
    
    crds_spl = spl.inv_cartesian(coords)
    
    answer = np.array([[1, 1.5, 0], 
                       [0, 1.5, -1],
                       [-1, 1.5, 0], 
                       [2, 1.5, -3]])
    
    assert_allclose(crds_spl, answer)
    
    coords = np.array([[1, 1.5, 0], 
                       [1, 1.5, np.pi/4],
                       [1, 1.5, np.pi/2],
                       [2, 1.5, np.pi], 
                       [2, 1.5, np.pi*1.5]])
    
    crds_spl = spl.inv_cylindrical(coords)
    
    answer = np.array([[0, 1.5, -1],
                       [np.sqrt(2)/2, 1.5, -np.sqrt(2)/2]
                       [1, 1.5, 0],
                       [0, 1.5, 2],
                       [-2, 1.5, 0]])
    
    assert_allclose(crds_spl, answer)
    