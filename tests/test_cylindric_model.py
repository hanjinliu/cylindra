import numpy as np
from numpy.testing import assert_allclose
from mtprops.components import CylindricModel, Spline
import pytest

def test_cylindric_model_construction():
    # Test the constructor
    model = CylindricModel(
        shape=(10, 10), 
        tilts=(0.1, 0.1), 
        intervals=(2., 3.),
        radius=1.2,
        offsets=(3., 5.),
    )
    assert model.shape == (10, 10)
    assert model.tilts == (0.1, 0.1)
    assert model.intervals == (2., 3.)
    assert model.radius == 1.2
    assert model.offsets == (3., 5.)
    assert model.displace.shape == (10, 10, 3)

    # Test the replace method
    assert model.replace(tilts=(0.2, 0.2)).tilts == (0.2, 0.2)
    assert model.replace(intervals=(2., 2.)).intervals == (2., 2.)
    assert model.replace(radius=2.0).radius == 2.0
    assert model.replace(offsets=(1., 1.)).offsets == (1., 1.)

def test_monomer_creation():
    model = CylindricModel(
        shape=(10, 10), 
        tilts=(0.1, 0.1), 
        intervals=(2., 3.),
        radius=1.2,
    )
    
    spl = Spline.line([0, 0, 0], [0, 25, 0])
    mole = model.to_molecules(spl)
    pos_y = mole.pos[:, 1].reshape(10, 10)
    
    dy = np.diff(pos_y, axis=0)
    assert_allclose(dy, 2., rtol=1e-6, atol=1e-6)
