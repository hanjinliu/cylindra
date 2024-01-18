import numpy as np
from acryo import Molecules
from numpy.testing import assert_allclose

from cylindra.components import CylSpline
from cylindra.cylmeasure import RegionProfiler


def test_length_and_width():
    image = np.zeros((5, 3), dtype=np.float32)
    label = np.array(
        [[0, 0, 2], [1, 0, 2], [1, 1, 2], [1, 1, 2], [0, 1, 0]],
        dtype=np.uint32,
    )
    reg = RegionProfiler.from_arrays(image, label, 1)
    result = reg.calculate(["length", "width"])
    assert_allclose(result["length"], [4, 4])
    assert_allclose(result["width"], [2, 1])


def test_length_and_width_at_boundary():
    image = np.zeros((5, 4), dtype=np.float32)
    label = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [2, 2, 2, 2],
        ],
        dtype=np.uint32,
    )
    reg = RegionProfiler.from_arrays(image, label, 1)
    result = reg.calculate(["length", "width"])
    assert_allclose(result["length"], [2, 1])
    assert_allclose(result["width"], [3, 4])


def test_length_and_width_non_regular():
    features = {
        "nth": [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4],
        "pf-id": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0],
        "values": [0] * 21,
        "labels": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0],
    }
    mole = Molecules(np.zeros((21, 3), dtype=np.float32), features=features)
    spl = CylSpline(config={"rise_sign": 1})
    spl.props.update_glob(npf=4, start=1, radius=1)
    reg = RegionProfiler.from_components(mole, spl, "values", "labels")
    result = reg.calculate(["length", "width"])
    assert_allclose(result["length"], [2, 1])
    assert_allclose(result["width"], [3, 4])


def test_intensity():
    image = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
        ],
        dtype=np.float32,
    )
    label = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 2, 0, 0],
            [2, 2, 2, 2],
        ],
        dtype=np.uint32,
    )
    data0 = [1, 2, 5, 5, 6]
    data1 = [5, 5, 6, 7, 8]
    reg = RegionProfiler.from_arrays(image, label, 1)
    result = reg.calculate(["mean", "std", "max", "min", "sum", "median"])
    assert_allclose(result["mean"], [np.mean(data0), np.mean(data1)])
    assert_allclose(result["std"], [np.std(data0), np.std(data1)])
    assert_allclose(result["max"], [np.max(data0), np.max(data1)])
    assert_allclose(result["min"], [np.min(data0), np.min(data1)])
    assert_allclose(result["sum"], [np.sum(data0), np.sum(data1)])
    assert_allclose(result["median"], [np.median(data0), np.median(data1)])


def test_area():
    image = np.zeros((5, 4), dtype=np.float32)
    label = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [2, 2, 2, 2],
        ],
        dtype=np.uint32,
    )
    reg = RegionProfiler.from_arrays(image, label, 1)
    result = reg.calculate(["area"])
    assert_allclose(result["area"], [5, 4])
