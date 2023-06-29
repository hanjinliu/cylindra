import pytest
import numpy as np
from numpy.testing import assert_array_equal
from cylindra import cylfilters
from acryo import Molecules
from cylindra.const import MoleculesHeader as Mole


def _2d_array_to_input(arr: np.ndarray) -> Molecules:
    nth, pf = np.stack(np.indices(arr.shape), axis=0).reshape(2, -1)
    mole = Molecules(
        np.zeros((nth.size, 3), dtype=np.float32),
        features={Mole.nth: nth, Mole.pf: pf, "value": arr.ravel()},
    )
    return mole


@pytest.mark.parametrize(
    "nrise, ans",
    [
        (0, [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2], [0, 0, 2, 2]]),
        (1, [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 2], [0, 0, 2, 2]]),
        (-1, [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
    ]
)  # fmt: skip
def test_label(nrise, ans):
    data = np.array(
        [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]]
    )
    mole = _2d_array_to_input(data)
    assert_array_equal(
        cylfilters.label(mole, "value", nrise).to_numpy().reshape(data.shape),
        np.array(ans),
    )


@pytest.mark.parametrize(
    "nrise, ans",
    [
        (0, [[1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 2], [2, 1, 2, 2]]),
        (1, [[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 2], [0, 1, 2, 2]]),
        (-1, [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [2, 0, 1, 2], [0, 1, 2, 2]])
    ]
)  # fmt: skip
def test_max_filter(nrise, ans):
    data = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2]]
    )
    mole = _2d_array_to_input(data)
    assert_array_equal(
        cylfilters.max_filter(mole, [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "value", nrise)
        .to_numpy()
        .astype(np.int32)
        .reshape(data.shape),
        np.array(ans),
    )
