import numpy as np
from numpy.testing import assert_allclose
from cylindra.components import CylindricModel, Spline

def test_cylindric_model_construction():
    # Test the constructor
    model = CylindricModel(
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
    model = CylindricModel(
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

def test_expand():
    model = CylindricModel(
        shape=(10, 8), 
        tilts=(0.1, 0.1), 
        interval=2.,
        radius=1.2,
    ).expand(0.5, start=4, stop=7)
    
    spl = Spline.line([0, 0, 0], [0, 25, 0])
    mole = model.to_molecules(spl)
    pos_y = mole.pos[:, 1].reshape(10, 8)
    
    dy = np.diff(pos_y, axis=0)
    # NOTE: After diff, 3:6 (not 4:7) are expanded
    assert_allclose(dy[:3], 2., rtol=1e-6, atol=1e-6)
    assert_allclose(dy[3:6], 2.5, rtol=1e-6, atol=1e-6)
    assert_allclose(dy[6:], 2., rtol=1e-6, atol=1e-6)
