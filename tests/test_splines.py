from mtprops.utils import map_coordinates
from mtprops.components.tomogram import MtSpline
import numpy as np
from numpy.testing import assert_allclose
import impy as ip


def test_inverse_mapping():
    spl = MtSpline()
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
    spl = MtSpline()
    coords = np.array([[2, 1, 2], [2, 2, 2], [2, 3, 2], [2, 4, 2]])
    spl.fit(coords)
    
    # Cartesian
    img = ip.array(np.arange(5*6*5).reshape(5, 6, 5), dtype=np.float32, axes="zyx")
    
    crds = spl.cartesian((3, 3))
    img_tr = map_coordinates(img, crds)
    assert_allclose(img["z=1:4;y=1:5;x=1:4"], img_tr, rtol=1e-6, atol=1e-6)
    
    crds = spl.local_cartesian((3, 3), 4, u=0.5)
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
    img_tr = map_coordinates(img, crds)
    img_tr = ip.asarray(img_tr, axes="rya")
    rmax, ymax, amax = np.unravel_index(np.argmax(img_tr), img_tr.shape)
    rmin, ymin, amin = np.unravel_index(np.argmin(img_tr), img_tr.shape)
    assert amax == 0
    assert ymax == 1
    assert rmax == 1
    assert amin < img_tr.shape[-1]/2
    
    crds = spl.local_cylindrical((1, 3), 4, u=0.5)
    img_tr = map_coordinates(img, crds)
    img_tr = ip.asarray(img_tr, axes="rya")
    rmax, ymax, amax = np.unravel_index(np.argmax(img_tr), img_tr.shape)
    rmin, ymin, amin = np.unravel_index(np.argmin(img_tr), img_tr.shape)
    assert amax == 0
    assert ymax == 1
    assert rmax == 1
    assert amin < img_tr.shape[-1]/2
    
def test_invert():
    spl = MtSpline()
   
    coords = np.array([[0, 0, 0], [2, 1, 0], [5, 2, 3], [4, 3, 2]])
    spl.fit(coords)
    spl.make_anchors(n=5)
    spl.orientation = "PlusToMinus"
    
    spl_inv = spl.invert()
    spl_inv_inv = spl_inv.invert()
    
    assert_allclose(spl_inv._lims, (1, 0))
    assert_allclose(spl_inv_inv._lims, (0, 1))
    assert spl_inv.orientation == "MinusToPlus"
    assert spl_inv_inv.orientation == "PlusToMinus"
    
    assert_allclose(spl(), spl_inv()[::-1])
    assert_allclose(spl(der=1), -spl_inv(der=1)[::-1])
    assert_allclose(spl(der=2), spl_inv(der=2)[::-1])
    assert_allclose(spl(der=3), -spl_inv(der=3)[::-1])
    
    assert_allclose(spl(), spl_inv_inv())
    assert_allclose(spl(der=1), spl_inv_inv(der=1))
    assert_allclose(spl(der=2), spl_inv_inv(der=2))
    assert_allclose(spl(der=3), spl_inv_inv(der=3))

def test_clip():
    spl = MtSpline()
    spl.orientation = "PlusToMinus"
    
    coords = np.array([[0, 0, 0], [2, 1, 0], [5, 2, 3], [4, 3, 2]])
    spl.fit(coords)
    spl.orientation = "PlusToMinus"
    
    spl_c0 = spl.clip(0.2, 0.7)
    spl_c1 = spl_c0.clip(0.6, 0.4)
    assert spl_c0.orientation == "PlusToMinus"
    assert spl_c1.orientation == "MinusToPlus"
    
    assert_allclose(spl_c0._lims, (0.2, 0.7))
    assert_allclose(spl_c1._lims, (0.5, 0.4))
    
    assert_allclose(spl([0.2, 0.5, 0.7]), spl_c0([0.0, 0.6, 1.0]))
    assert_allclose(spl_c0([0.4, 0.5, 0.6]), spl_c1([1.0, 0.5, 0.0]))
    assert_allclose(spl([0.4, 0.45, 0.5]), spl_c1([1.0, 0.5, 0.0]))

def test_shift_fit():
    spl = MtSpline(scale=0.5)
   
    coords = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6]])
    spl.fit(coords)
    spl.make_anchors(n=4)
    spl.shift_fit(shifts=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]))
    spl.make_anchors(n=4)
    assert_allclose(spl(), np.array([[1, 0, 0], [1, 1, 2], [1, 2, 4], [1, 3, 6]]))

def test_dict():
    spl = MtSpline(scale=0.5)
   
    coords = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6]])
    spl.fit(coords)
    spl.orientation = "PlusToMinus"
    spl.clip(0.2, 0.8)
    
    d = spl.to_dict()
    spl_from_dict = MtSpline.from_dict(d)
    assert spl == spl_from_dict
