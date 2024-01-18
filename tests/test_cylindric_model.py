import numpy as np
import pytest
from numpy.testing import assert_allclose

from cylindra.components import CylinderModel, Spline
from cylindra.components import indexer as Idx


def test_cylindric_model_construction():
    # Test the constructor
    model = CylinderModel(
        shape=(10, 10),
        tilts=(0.1, 0.1),
        intervals=(2.0, 2 * np.pi / 10),
        radius=1.2,
        offsets=(3.0, 5.0),
    )
    assert model.shape == (10, 10)
    assert model.tilts == (0.1, 0.1)
    assert model.intervals == (2.0, np.pi / 5)
    assert model.radius == 1.2
    assert model.offsets == (3.0, 5.0)
    assert model.displace.shape == (10, 10, 3)

    # Test the replace method
    assert model.replace(tilts=(0.2, 0.2)).tilts == (0.2, 0.2)
    assert model.replace(intervals=(4.0, np.pi / 5)).intervals == (4.0, np.pi / 5)
    assert model.replace(radius=2.0).radius == 2.0
    assert model.replace(offsets=(1.0, 1.0)).offsets == (1.0, 1.0)


def test_repr():
    model = CylinderModel(
        shape=(10, 10),
        tilts=(0.1, 0.1),
        intervals=(2.0, 2 * np.pi / 10),
        radius=1.2,
        offsets=(3.0, 5.0),
    )
    repr(model)
    repr(Idx[10:20, 3:7])


def test_monomer_creation():
    model = CylinderModel(
        shape=(10, 8),
        tilts=(0.1, 0.1),
        intervals=(2.0, 2 * np.pi / 8),
        radius=1.2,
    )

    spl = Spline.line([0, 0, 0], [0, 25, 0])
    mole = model.to_molecules(spl)
    pos_y = mole.pos[:, 1].reshape(10, 8)

    dy = np.diff(pos_y, axis=0)
    assert_allclose(dy, 2.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ["aslice", "expected"],
    [
        (slice(2, 5), [Idx[40:50, 2:5]]),
        (slice(2, 15), [Idx[40:50, 2:10], Idx[43:53, 0:5]]),
        (slice(12, 15), [Idx[43:53, 2:5]]),
        (slice(-3, 2), [Idx[37:47, 7:10], Idx[40:50, 0:2]]),
        (slice(-15, -6), [Idx[34:44, 5:10], Idx[37:47, 0:4]]),
        (slice(-15, -11), [Idx[34:44, 5:9]]),
    ],
)
def test_cylindric_slice(aslice: slice, expected: list[tuple[slice, slice]]):
    idx = Idx[40:50, aslice]
    resolved = idx.get_resolver(3).resolve_slices((100, 10))
    assert resolved == expected


def test_expand():
    model = CylinderModel(
        shape=(10, 8),
        tilts=(0.1, 0.1),
        intervals=(2.0, 1.0),
        radius=1.2,
    ).expand(0.5, Idx[4:7, :])

    spl = Spline.line([0, 0, 0], [0, 25, 0])
    mole = model.to_molecules(spl)
    pos_y = mole.pos[:, 1].reshape(10, 8)

    dy = np.diff(pos_y, axis=0)
    print(np.round(dy, 3))
    # NOTE: After diff, 3:6 (not 4:7) are expanded
    assert_allclose(dy[:3], 2.0, rtol=1e-6, atol=1e-6)
    assert_allclose(dy[3:6], 2.5, rtol=1e-6, atol=1e-6)
    assert_allclose(dy[6:], 2.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("idx", [Idx[0:3, :], Idx[4:7, :], Idx[8:10, :]])
def test_alleviate_works(idx):
    model = CylinderModel(
        shape=(10, 8),
        tilts=(0.1, 0.1),
        intervals=(2.0, 1.0),
        radius=1.2,
    ).expand(0.5, idx)

    shifted = np.zeros((10, 8), dtype=bool)
    shifted[idx] = True

    model.alleviate(shifted)
