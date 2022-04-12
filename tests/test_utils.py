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
    
    sub = utils.crop_tomogram(img, (20, 25, 25), (3, 3, 3))
    assert_allclose(sub[0], np.array([[192424, 192425, 192426],
                                      [192524, 192525, 192526], 
                                      [192624, 192625, 192626]]))
    assert_allclose(sub[1], np.array([[202424, 202425, 202426],
                                      [202524, 202525, 202526], 
                                      [202624, 202625, 202626]]))
    assert_allclose(sub[2], np.array([[212424, 212425, 212426],
                                      [212524, 212525, 212526], 
                                      [212624, 212625, 212626]]))
    
    sub = utils.crop_tomogram(img, (1, 1, 1), (5, 5, 5))
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
    yy, xx = np.indices((100, 100))
    img = ip.asarray(np.sin(yy/4)*np.sin(xx/3))
    
    def isclose(coords):
        return assert_allclose(
            ndi.map_coordinates(img.value, coords, order=3),
            utils.map_coordinates(img, coords, order=3),
            rtol=1e-6,
            atol=1e-6,
        )
    
    coords = np.array([[[10, 13], [11, 16], [12, 19]],
                       [[3, 32], [7, 26], [10, 18]]])
    
    isclose(coords)
        
    coords = np.array([[[-3, 10], [0.9, 2], [3, -6]],
                       [[63, 28], [10, -1], [-20, -12]]])
    
    isclose(coords)
    
    coords = np.array([[[-3, -6], [50, 60.2], [110, 120]],
                       [[120, 108], [10, 40], [-20, -12]]])
    
    isclose(coords)

def test_multi_map_coordinates():
    from scipy import ndimage as ndi
    yy, xx = np.indices((100, 100))
    img = ip.asarray(np.sin(yy/4)*np.sin(xx/3))
    
    coords = np.array([[[10, 13], [11, 16], [12, 19]],
                       [[3, 32], [7, 26], [10, 18]]], dtype=np.float64
                      )
    all_coords = np.stack([coords + i for i in range(20)], axis=0)
    
    out0 = [ndi.map_coordinates(img.value, crds, order=3) for crds in all_coords]
    
    out1 = sum([list(utils.multi_map_coordinates(img, crds, order=3))
               for crds in np.split(all_coords, 4)], start=[])
    for a, b in zip(out0, out1):
        assert_allclose(a, b)
    
    out1 = sum([list(utils.multi_map_coordinates(img, crds, order=3))
               for crds in np.split(all_coords, [3, 7, 12, 17])], start=[])
    for a, b in zip(out0, out1):
        assert_allclose(a, b)
    

def test_mirror_pcc():
    np.random.seed(1234)
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

def test_mt_pad():
    arr = \
    [[a00, a01, a02, a03],
     [a10, a11, a12, a13],
     [a20, a21, a22, a23],
     [a30, a31, a32, a33],
     [a40, a41, a42, a43]] = \
    np.array(
        [[1, 2, 3, 4],
         [5, 5, 5, 5],
         [2, 2, 3, 3],
         [6, 6, 6, 6],
         [4, 3, 2, 1]]
    )
        
    out = utils.pad_sheared_edges(arr, (1, 2), start=2)
    assert_allclose(
        [[a32, a33, a10, a11, a12, a13, a10, a11],
         [a22, a23, a00, a01, a02, a03, a20, a21],
         [a12, a13, a10, a11, a12, a13, a30, a31],
         [a02, a03, a20, a21, a22, a23, a40, a41],
         [a12, a13, a30, a31, a32, a33, a30, a31],
         [a22, a23, a40, a41, a42, a43, a20, a21],
         [a32, a33, a30, a31, a32, a33, a10, a11]],
        out
    )
    
    out = utils.pad_sheared_edges(arr, (2, 2), start=1)
    assert_allclose(
        [[a32, a33, a20, a21, a22, a23, a10, a11],
         [a22, a23, a10, a11, a12, a13, a00, a01],
         [a12, a13, a00, a01, a02, a03, a10, a11],
         [a02, a03, a10, a11, a12, a13, a20, a21],
         [a12, a13, a20, a21, a22, a23, a30, a31],
         [a22, a23, a30, a31, a32, a33, a40, a41],
         [a32, a33, a40, a41, a42, a43, a30, a31],
         [a42, a43, a30, a31, a32, a33, a20, a21],
         [a32, a33, a20, a21, a22, a23, a10, a11]],
        out
    )
