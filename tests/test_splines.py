from mtprops.utils import map_coordinates
from mtprops.spline import Spline3D
import numpy as np
from numpy.testing import assert_allclose
import impy as ip


def test_inverse_mapping():
    spl = Spline3D()
    coords = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    spl.fit(coords)
    coords = np.array([[1, 1.5, 0], 
                       [0, 1.5, 1],
                       [-1, 1.5, 0], 
                       [2, 1.5, 3]])
    
    crds_spl = spl.cartesian_to_world(coords)
    
    answer = np.array([[1, 1.5, 0], 
                       [0, 1.5, -1],
                       [-1, 1.5, 0], 
                       [2, 1.5, -3]])
    
    assert_allclose(crds_spl, answer, rtol=1e-6, atol=1e-6)
    
    coords = np.array([[1, 1.5, 0], 
                       [1, 1.5, np.pi/4],
                       [1, 1.5, np.pi/2],
                       [2, 1.5, np.pi], 
                       [2, 1.5, np.pi*1.5]])
    
    crds_spl = spl.cylindrical_to_world(coords)
    
    answer = np.array([[0, 1.5, -1],
                       [np.sqrt(2)/2, 1.5, -np.sqrt(2)/2],
                       [1, 1.5, 0],
                       [0, 1.5, 2],
                       [-2, 1.5, 0]])
    
    assert_allclose(crds_spl, answer, rtol=1e-6, atol=1e-6)


def test_coordinate_transformation():
    spl = Spline3D()
    coords = np.array([[2, 1, 2], [2, 2, 2], [2, 3, 2], [2, 4, 2]])
    spl.fit(coords)
    
    # Cartesian
    img = ip.array(np.arange(5*6*5).reshape(5, 6, 5), dtype=np.float32, axes="zyx")
    
    crds = spl.cartesian((3, 3))
    crds = np.moveaxis(crds, -1, 0)
    img_tr = map_coordinates(img, crds)
    assert_allclose(img["z=1:4;y=1:5;x=1:4"], img_tr, rtol=1e-6, atol=1e-6)
    
    crds = spl.local_cartesian((3, 3), 4, u=0.5)
    crds = np.moveaxis(crds, -1, 0)
    img_tr = map_coordinates(img, crds)
    assert_allclose(img["z=1:4;y=1:5;x=1:4"], img_tr, rtol=1e-6, atol=1e-6)
    
    # Cylindrical
    img = ip.zeros((5, 4, 5), dtype=np.float32, axes="zyx")
    img["z=2;y=2;x=3"] = 1
    img["z=3;y=2;x=1"] = -1
    
    # 0  0  0  0  0  z
    # 0 -1  0  0  0  ^
    # 0  0  0 +1  0  |
    # 0  0  0  0  0
    # 0  0  0  0  0 -> x
    
    crds = spl.cylindrical((1, 3))
    crds = np.moveaxis(crds, -1, 0)
    img_tr = map_coordinates(img, crds)
    img_tr = ip.asarray(img_tr, axes="rya")
    rmax, ymax, amax = np.unravel_index(np.argmax(img_tr), img_tr.shape)
    rmin, ymin, amin = np.unravel_index(np.argmin(img_tr), img_tr.shape)
    assert amax == 0
    assert ymax == 1
    assert rmax == 1
    assert amin < img_tr.shape[-1]/2
    
    crds = spl.local_cylindrical((1, 3), 4, u=0.5)
    crds = np.moveaxis(crds, -1, 0)
    img_tr = map_coordinates(img, crds)
    img_tr = ip.asarray(img_tr, axes="rya")
    rmax, ymax, amax = np.unravel_index(np.argmax(img_tr), img_tr.shape)
    rmin, ymin, amin = np.unravel_index(np.argmin(img_tr), img_tr.shape)
    assert amax == 0
    assert ymax == 1
    assert rmax == 1
    assert amin < img_tr.shape[-1]/2
    
def test_invert():
    spl = Spline3D()
    coords = np.array([[0, 0, 0], [2, 1, 0], [5, 2, 3], [4, 3, 2]])
    spl.fit(coords)
    spl.make_anchors(n=5)
    
    spl_inv = spl.invert()
    spl_inv_inv = spl_inv.invert()
    
    assert_allclose(spl(), spl_inv()[::-1])
    assert_allclose(spl(der=1), -spl_inv(der=1)[::-1])
    
    assert_allclose(spl(), spl_inv_inv())
    assert_allclose(spl(der=1), spl_inv_inv(der=1))
    assert_allclose(spl(der=2), spl_inv_inv(der=2))
    