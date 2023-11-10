import numpy as np
from numpy.testing import assert_allclose
from cylindra.cylmeasure import LatticeParameters
from cylindra.components import CylSpline, CylinderModel, indexer as Idx


def test_interval():
    spl = CylSpline.line([0, 0, 0], [10, 10, 10])
    model = CylinderModel((5, 4), intervals=(1, np.pi / 2), radius=1).expand(
        0.1, Idx[2:4, :]
    )
    mole = model.to_molecules(spl)
    ans = np.ones((5, 4))
    ans[1:3, :] += 0.1

    result = LatticeParameters("spacing").calculate(mole, spl)
    assert_allclose(result[:-4], ans.ravel()[:-4], rtol=1e-6)
