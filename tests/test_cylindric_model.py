import numpy as np
from numpy.testing import assert_allclose
import pytest
from cylindra.components import CylinderModel, Spline, indexer as Idx
from cylindra._cpp_ext import CylinderGeometry

def test_cylindric_model_construction():
    # Test the constructor
    model = CylinderModel(
        shape=(10, 10), 
        tilts=(0.1, 0.1), 
        interval=2.,
        radius=1.2,
        offsets=(3., 5.),
    )
    assert model.shape == (10, 10)
    assert model.tilts == (0.1, 0.1)
    assert model.intervals == (2., np.pi / 5)
    assert model.radius == 1.2
    assert model.offsets == (3., 5.)
    assert model.displace.shape == (10, 10, 3)

    # Test the replace method
    assert model.replace(tilts=(0.2, 0.2)).tilts == (0.2, 0.2)
    assert model.replace(interval=4.).intervals == (4., np.pi / 5)
    assert model.replace(radius=2.0).radius == 2.0
    assert model.replace(offsets=(1., 1.)).offsets == (1., 1.)

def test_monomer_creation():
    model = CylinderModel(
        shape=(10, 8), 
        tilts=(0.1, 0.1), 
        interval=2.,
        radius=1.2,
    )
    
    spl = Spline.line([0, 0, 0], [0, 25, 0])
    mole = model.to_molecules(spl)
    pos_y = mole.pos[:, 1].reshape(10, 8)
    
    dy = np.diff(pos_y, axis=0)
    assert_allclose(dy, 2., rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize(
    ["aslice", "expected"],
    [
        (slice(2, 5), [Idx[40:50, 2:5]]),
        (slice(2, 15), [Idx[40:50, 2:10], Idx[43:53, 0:5]]),
        (slice(12, 15), [Idx[43:53, 2:5]]),
        (slice(-3, 2), [Idx[37:47, 7:10], Idx[40:50, 0:2]]),
        (slice(-15, -6), [Idx[34:44, 5:10], Idx[37:47, 0:4]]),
        (slice(-15, -11), [Idx[34:44, 5:9]]),
    ]
)
def test_cylindric_slice(aslice: slice, expected: list[tuple[slice, slice]]):
    idx = Idx[40:50, aslice]
    resolved = idx.get_resolver(3).resolve_slices((100, 10))
    assert resolved == expected

def test_expand():
    model = CylinderModel(
        shape=(10, 8), 
        tilts=(0.1, 0.1), 
        interval=2.,
        radius=1.2,
    ).expand(0.5, Idx[4:7, :])
    
    spl = Spline.line([0, 0, 0], [0, 25, 0])
    mole = model.to_molecules(spl)
    pos_y = mole.pos[:, 1].reshape(10, 8)
    
    dy = np.diff(pos_y, axis=0)
    # NOTE: After diff, 3:6 (not 4:7) are expanded
    assert_allclose(dy[:3], 2., rtol=1e-6, atol=1e-6)
    assert_allclose(dy[3:6], 2.5, rtol=1e-6, atol=1e-6)
    assert_allclose(dy[6:], 2., rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize(
    "idx", [Idx[0:3, :], Idx[4:7, :], Idx[8:10, :]]
)
def test_alleviate_works(idx):
    model = CylinderModel(
        shape=(10, 8), 
        tilts=(0.1, 0.1), 
        interval=2.,
        radius=1.2,
    ).expand(0.5, idx)
    
    shifted = np.zeros((10, 8), dtype=bool)
    shifted[idx] = True

    model.alleviate(shifted, 1)
    model.alleviate(shifted, 2)

@pytest.mark.parametrize(
    ["nrise", "index", "source"],
    [(3, (0, 0), []),
     (3, (0, 3), [(-1, -1), (0, 2)]),
     (3, (1, 0), [(0, 0)]),
     (3, (2, 0), [(1, 0)]),
     (3, (3, 0), [(2, 0), (0, 4)]),
     (3, (5, 2), [(4, 2), (5, 1)]),
     (3, (9, 4), [(8, 4), (9, 3)]),
     (-3, (0, 0), [(-1, -1), (0, 1)]),
     (-3, (0, 4), []),
     (-3, (1, 4), [(0, 4)]),
     (-3, (2, 4), [(1, 4)]),
     (-3, (3, 4), [(2, 4), (0, 0)]),
     (-3, (5, 2), [(4, 2), (5, 3)]),
     (-3, (9, 0), [(8, 0), (9, 1)]),
     ]
)
def test_cylinder_source_forward(
    nrise: int,
    index: tuple[int, int],
    source: list[tuple[int, int]],
):
    geometry = CylinderGeometry(10, 5, nrise)
    assert geometry.source_forward(*index) == source


@pytest.mark.parametrize(
    ["nrise", "index", "source"],
    [(3, (0, 0), [(1, 0), (0, 1)]),
     (3, (3, 4), [(4, 4), (6, 0)]),
     (3, (5, 2), [(6, 2), (5, 3)]),
     (3, (9, 4), []),
     (3, (9, 3), [(-1, -1), (9, 4)]),
     (3, (8, 4), [(9, 4)]),
     (3, (7, 4), [(8, 4)]),
     (3, (6, 4), [(7, 4), (9, 0)]),
     (-3, (0, 0), [(1, 0), (3, 4)]),
     (-3, (3, 4), [(4, 4), (3, 3)]),
     (-3, (5, 2), [(6, 2), (5, 1)]),
     (-3, (9, 0), []),
     (-3, (9, 1), [(-1, -1), (9, 0)]),
     (-3, (8, 0), [(9, 0)]),
     (-3, (7, 0), [(8, 0)]),
     (-3, (6, 0), [(7, 0), (9, 4)]),
     ]
)
def test_cylinder_source_backward(
    nrise: int,
    index: tuple[int, int],
    source: list[tuple[int, int]],
):
    geometry = CylinderGeometry(10, 5, nrise)
    assert geometry.source_backward(*index) == source

