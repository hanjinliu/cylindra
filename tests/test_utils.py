from mtprops import utils
import pytest
import numpy as np
from numpy.testing import assert_allclose
import impy as ip

def test_ints():
    assert utils.roundint(1.) == 1
    assert utils.roundint(2.) == 2
    assert utils.roundint(1.47) == 1
    assert utils.roundint(0.52) == 1
    assert utils.roundint(-5.3) == -5
    assert utils.roundint(-4.6) == -5
    assert utils.ceilint(5.) == 5
    assert utils.ceilint(6.) == 6
    assert utils.ceilint(5.01) == 6
    assert utils.ceilint(5.9) == 6


def test_make_slice_and_pad():
    assert utils.make_slice_and_pad(4, 8, 10) == (slice(4, 8), (0, 0))
    assert utils.make_slice_and_pad(-2, 5, 10) == (slice(0, 5), (2, 0))
    assert utils.make_slice_and_pad(6, 13, 10) == (slice(6, 10), (0, 3))
    assert utils.make_slice_and_pad(-5, 14, 10) == (slice(0, 10), (5, 4))
    with pytest.raises(ValueError):
        assert utils.make_slice_and_pad(-5, -2, 10) == (slice(0, 0), (5, 0))
    with pytest.raises(ValueError):
        assert utils.make_slice_and_pad(12, 16, 10) == (slice(0, 0), (0, 6))


def test_load_a_subtomogram():
    zz, yy, xx = np.indices((40, 50, 50), dtype=np.float32)
    img = ip.asarray(zz*1e4 + yy*1e2 + xx, axes="zyx")
    
    sub = utils.load_a_subtomogram(img, (20, 25, 25), (3, 3, 3))
    assert_allclose(sub[0], np.array([[192424, 192425, 192426],
                                      [192524, 192525, 192526], 
                                      [192624, 192625, 192626]]))
    assert_allclose(sub[1], np.array([[202424, 202425, 202426],
                                      [202524, 202525, 202526], 
                                      [202624, 202625, 202626]]))
    assert_allclose(sub[2], np.array([[212424, 212425, 212426],
                                      [212524, 212525, 212526], 
                                      [212624, 212625, 212626]]))
    
    sub = utils.load_a_subtomogram(img, (1, 1, 1), (5, 5, 5))
    c = np.mean(sub)
    assert_allclose(sub[0], np.full((5, 5), c))
    assert_allclose(sub[1], np.array([[c,   c,   c,   c,   c],
                                      [c,   0,   1,   2,   3],
                                      [c, 100, 101, 102, 103],
                                      [c, 200, 201, 202, 203],
                                      [c, 300, 301, 302, 303]]))


def test_interval_divmod():
    def close(a, b):
        return np.isclose(a[0], b[0]) and a[1] == b[1]
    
    assert close(utils.interval_divmod(23, 4), (20, 5))
    assert close(utils.interval_divmod(23.7, 4), (20, 5))
    assert close(utils.interval_divmod(28.5, 5.4), (27, 5))
    assert close(utils.interval_divmod(20, 1.2), (19.2, 16))


def test_map_coordinates():
    from scipy import ndimage as ndi
    np.random.seed(0)
    img = ip.random.normal(size=(100, 100))
    
    def isclose(coords):
        return np.allclose(ndi.map_coordinates(img, coords, order=3),
                           utils.map_coordinates(img, coords, order=3))
    
    coords = np.array([[[10, 13], [11, 16], [12, 19]],
                       [[3, 32], [7, 26], [10, 18]]])
    
    assert isclose(coords)
    
    coords = np.array([[[-3, 10], [0, 2], [3, -6]],
                       [[63, 28], [10, -1], [-20, -12]]])
    
    assert isclose(coords)
    
    coords = np.array([[[-3, -6], [50, 60], [110, 120]],
                       [[120, 108], [10, 40], [-20, -12]]])
    
    assert isclose(coords)
    
    np.random.seed(None)

def test_mirror_pcc():
    np.random.seed(1234)
    with ip.SetConst(SHOW_PROGRESS=False):
        # Even number
        img = ip.random.normal(size=(128, 128))
        img_mirror = img[::-1, ::-1]
        shift1 = ip.pcc_maximum(img, img_mirror)
        shift2 = utils.mirror_pcc(img)
        assert_allclose(shift1, shift2)
        
        # Odd number
        img = ip.random.normal(size=(127, 127))
        img_mirror = img[::-1, ::-1]
        shift1 = ip.pcc_maximum(img, img_mirror)
        shift2 = utils.mirror_pcc(img)
        assert_allclose(shift1, shift2)
    np.random.seed()