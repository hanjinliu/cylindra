import impy as ip
import numpy as np
import pytest
from IPython.display import display
from numpy.testing import assert_allclose

from cylindra.components import CylSpline
from cylindra.utils import map_coordinates


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_inverse_mapping(mode):
    spl = CylSpline(extrapolate=mode)
    coords = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    spl = spl.fit(coords)
    coords = np.array([[1, 1.5, 0], [0, 1.5, 1], [-1, 1.5, 0], [2, 1.5, 3]])

    crds_spl = spl.cartesian_to_world(coords)

    answer = np.array([[1, 1.5, 0], [0, 1.5, -1], [-1, 1.5, 0], [2, 1.5, -3]])

    assert_allclose(crds_spl, answer, rtol=1e-6, atol=1e-6)

    coords = np.array(
        [
            [1, 1.5, 0],
            [1, 1.5, np.pi / 4],
            [1, 1.5, np.pi / 2],
            [2, 1.5, np.pi],
            [2, 1.5, np.pi * 1.5],
        ]
    )

    crds_spl = spl.cylindrical_to_world(coords)

    answer = np.array(
        [
            [0, 1.5, -1],
            [np.sqrt(2) / 2, 1.5, -np.sqrt(2) / 2],
            [1, 1.5, 0],
            [0, 1.5, 2],
            [-2, 1.5, 0],
        ]
    )

    assert_allclose(crds_spl, answer, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_coordinate_transformation(mode):
    spl = CylSpline(extrapolate=mode)
    coords = np.array([[2, 1, 2], [2, 2, 2], [2, 3, 2], [2, 4, 2]])
    spl = spl.fit(coords)

    # Cartesian
    img = ip.array(np.arange(5 * 6 * 5).reshape(5, 6, 5), dtype=np.float32, axes="zyx")

    crds = spl.cartesian((3, 3))
    img_tr = map_coordinates(img, crds)
    assert_allclose(img["z=1:4;y=1:5;x=1:4"], img_tr, rtol=1e-6, atol=1e-6)

    crds = spl.local_cartesian((3, 3), 4, u=0.5)
    img_tr = map_coordinates(img, crds)
    assert_allclose(img["z=1:4;y=1:5;x=1:4"], img_tr, rtol=1e-6, atol=1e-6)

    # Cylindrical
    img = ip.zeros((5, 4, 5), dtype=np.float32, axes="zyx")
    img["z=2;y=2;x=3"] = 1
    img["z=4;y=2;x=0"] = -1

    # -1  0  0  0  0  z
    #  0  0  0  0  0  ^
    #  0  0  * +1  0  |
    #  0  0  0  0  0
    #  0  0  0  0  0 -> x

    crds = spl.cylindrical((1, 4))
    img_tr = map_coordinates(img, crds)
    img_tr = ip.asarray(img_tr, axes="rya")
    rmax, ymax, amax = np.unravel_index(np.argmax(img_tr), img_tr.shape)
    rmin, ymin, amin = np.unravel_index(np.argmin(img_tr), img_tr.shape)
    assert (amax, ymax, rmax) == (0, 1, 0)
    assert 1 < amin < img_tr.shape[-1] / 2
    assert ymin == 1
    assert rmin == 1

    crds = spl.local_cylindrical((1, 4), 4, u=0.5)
    img_tr = map_coordinates(img, crds)
    img_tr = ip.asarray(img_tr, axes="rya")
    rmax, ymax, amax = np.unravel_index(np.argmax(img_tr), img_tr.shape)
    rmin, ymin, amin = np.unravel_index(np.argmin(img_tr), img_tr.shape)
    assert (amax, ymax, rmax) == (0, 1, 0)
    assert 1 < amin < img_tr.shape[-1] / 2
    assert ymin == 1
    assert rmin == 1


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_invert(mode):
    spl = CylSpline(extrapolate=mode)

    coords = np.array([[0, 0, 0], [2, 1, 0], [5, 2, 3], [4, 3, 2]])
    spl = spl.fit(coords)
    spl.make_anchors(n=5)
    spl.orientation = "PlusToMinus"

    spl_inv = spl.invert()
    spl_inv_inv = spl_inv.invert()

    assert_allclose(spl_inv._lims, (1, 0))
    assert_allclose(spl_inv_inv._lims, (0, 1))
    assert spl_inv.orientation == "MinusToPlus"
    assert spl_inv_inv.orientation == "PlusToMinus"

    assert_allclose(spl(), spl_inv()[::-1], rtol=1e-6, atol=1e-6)
    assert_allclose(spl(der=1), -spl_inv(der=1)[::-1], rtol=1e-6, atol=1e-6)
    assert_allclose(spl(der=2), spl_inv(der=2)[::-1], rtol=1e-6, atol=1e-6)
    assert_allclose(spl(der=3), -spl_inv(der=3)[::-1], rtol=1e-6, atol=1e-6)

    assert_allclose(spl(), spl_inv_inv(), rtol=1e-6, atol=1e-6)
    assert_allclose(spl(der=1), spl_inv_inv(der=1), rtol=1e-6, atol=1e-6)
    assert_allclose(spl(der=2), spl_inv_inv(der=2), rtol=1e-6, atol=1e-6)
    assert_allclose(spl(der=3), spl_inv_inv(der=3), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_clip(mode):
    spl = CylSpline(extrapolate=mode)
    spl.orientation = "PlusToMinus"

    coords = np.array([[0, 0, 0], [2, 1, 0], [5, 2, 3], [4, 3, 2]])
    spl = spl.fit(coords)
    spl.orientation = "PlusToMinus"

    spl0 = spl.clip(0.2, 0.7)
    spl1 = spl0.clip(0.6, 0.4)
    assert spl0.orientation == "PlusToMinus"
    assert spl1.orientation == "MinusToPlus"

    assert_allclose(spl0._lims, (0.2, 0.7))
    assert_allclose(spl1._lims, (0.5, 0.4))

    assert_allclose(spl([0.2, 0.5, 0.7]), spl0([0.0, 0.6, 1.0]))
    assert_allclose(spl0([0.4, 0.5, 0.6]), spl1([1.0, 0.5, 0.0]))
    assert_allclose(spl([0.4, 0.45, 0.5]), spl1([1.0, 0.5, 0.0]))


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_extend(mode):
    spl = CylSpline(extrapolate=mode)
    coords = np.array([[0, 0, 0], [2, 1, 0], [5, 2, 3], [6, 3, 4]])
    spl = spl.fit(coords)
    spl.orientation = "PlusToMinus"

    spl0 = spl.clip(-0.8, 1.2)  # relative length = 2.0
    spl1 = spl0.clip(-0.2, 1.1)  # relative length = 2.6

    assert_allclose(spl0._lims, (-0.8, 1.2))
    assert_allclose(spl1._lims, (-1.2, 1.4))


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_shift_fit(mode):
    spl = CylSpline(extrapolate=mode)

    coords = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6]])
    spl = spl.fit(coords)
    spl.make_anchors(n=4)
    spl = spl.shift(shifts=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]))
    spl.make_anchors(n=4)
    assert_allclose(
        spl(),
        np.array([[1, 0, 0], [1, 1, 2], [1, 2, 4], [1, 3, 6]]),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_dict(mode):
    spl = CylSpline(extrapolate=mode)

    coords = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6]])
    spl = spl.fit(coords)
    spl.orientation = "PlusToMinus"
    spl = spl.clip(0.2, 0.8)

    d = spl.to_dict()
    spl_from_dict = CylSpline.from_dict(d)
    assert spl.close_to(spl_from_dict)


@pytest.mark.parametrize("radius", [0.5, 2.0, 4.0, 10.0])
@pytest.mark.parametrize("mode", ["linear", "default"])
def test_curvature(radius, mode):
    spl = CylSpline(extrapolate=mode)
    u = np.linspace(0, 2 * np.pi, 100)
    coords = np.stack([np.zeros(100), radius * np.sin(u), radius * np.cos(u)], axis=1)
    spl = spl.fit(coords, err_max=0)
    spl.make_anchors(n=100)
    cr = spl.curvature_radii()
    cr_mean = np.mean(cr)
    assert (cr_mean / radius - 1) ** 2 < 1e-4
    assert np.std(cr) / cr_mean < 1e-3


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_translate(mode):
    spl = CylSpline(extrapolate=mode)
    spl = spl.fit([[3, 2, 1], [4, 6, 7], [5, 2, 3], [9, 5, 6]])
    ds = np.array([3, 1, -2])
    spl_trans = spl.translate(ds)
    assert_allclose(
        spl_trans.partition(100),
        ds + spl.partition(100),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["linear", "default"])
def test_extrapolate_map(mode):
    spl = CylSpline(extrapolate=mode)
    spl = spl.fit([[3, 2, 1], [4, 6, 7], [5, 2, 3], [9, 5, 6]])
    sl0 = [-0.5, 0, 0.5, 1, 1.5]
    sl1 = [0, 0.5, 1]

    for der in [0, 1, 2]:
        assert_allclose(spl.map(sl0, der=der)[1:4], spl.map(sl1, der=der))
        assert_allclose(spl.map(0.3, der=der), spl.map(0.3, der=der))
        spl.map(-0.5, der=der)
        spl.map(1.5, der=der)


def test_update_props():
    spl = CylSpline()
    spl = spl.fit([[3, 2, 1], [4, 6, 7], [5, 2, 3], [9, 5, 6]])
    spl.update_props(npf=13, orientation="PlusToMinus")


def test_resample():
    spl = CylSpline()
    spl = spl.fit([[3, 2, 1], [4, 6, 7], [5, 2, 3], [9, 5, 6]]).clip(0.1, 0.9)
    spl0 = spl.resample(0.3)
    assert spl.length() == pytest.approx(spl0.length(), rel=1e-2)


def test_with_x():
    spl = CylSpline().fit([[3, 2, 1], [4, 6, 7], [5, 2, 3], [9, 5, 6]])
    spl.with_extrapolation("linear").with_config({"fit_width": 33})
    display(spl.config)
